//! Li et al. (2024) "EconAgent" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で LLM 駆動マクロ経済 ABM を実行する．
//! `sweep`     : エージェント数 × 税率スケール × LLM モデル (× 政策レジーム) を走査し，
//!               最終/平均マクロ指標を `sweep_summary.csv` に集計する．
//! `reproduce` : 論文の headline マクロ動態 (インフレ/失業/GDP) と Phillips 曲線
//!               (インフレ vs 失業; 負相関)・Okun の法則 (GDP 成長 vs 失業変化;
//!               負相関) を一括再現し，観測相関を期待符号アンカーと突合する．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use econagent_simulation::config::{parse_regime, Config, LlmSettings, PolicyRegime};
use econagent_simulation::metrics::{mean, okun_correlation, phillips_correlation};
use econagent_simulation::simulation::{
    ensure_output_dir, mock_decision_client, run, run_with_client, save_metrics, save_run_metadata,
    SimulationResult,
};

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "econagent",
    about = "Li et al. (2024) EconAgent: LLM-Empowered Agents for Simulating Macroeconomic Activities — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で LLM 駆動マクロ経済 ABM を実行する．
    Run(RunArgs),
    /// エージェント数 × 税率スケール × LLM モデルを走査し，マクロ指標を集計する．
    Sweep(SweepArgs),
    /// 論文 headline (Phillips 曲線 / Okun の法則 / マクロ動態) を一括再現する．
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// 家計数 N．
    #[arg(long, default_value_t = 100)]
    n_agents: usize,

    /// シミュレーション月数 (240 = 20 年)．
    #[arg(long, default_value_t = 240)]
    months: usize,

    /// 記憶長 L (直近 L か月の会話プール)．
    #[arg(long, default_value_t = 1)]
    memory_length: usize,

    /// 政策レジーム (progressive / proportional / none)．
    #[arg(long, default_value = "progressive")]
    regime: String,

    /// 累進税率スケール (基準税率の倍率)．
    #[arg(long, default_value_t = 1.0)]
    tax_scale: f64,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0; 論文は 1.0 近傍)．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (既定 .llm_cache/cache.json)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// カンマ区切りのエージェント数リスト．
    #[arg(long, default_value = "50,100,300")]
    n_agents_values: String,

    /// 税率スケールの最小値．
    #[arg(long, default_value_t = 0.5)]
    tax_scale_min: f64,

    /// 税率スケールの最大値．
    #[arg(long, default_value_t = 1.5)]
    tax_scale_max: f64,

    /// 税率スケールの刻み幅．
    #[arg(long, default_value_t = 0.5)]
    tax_scale_step: f64,

    /// カンマ区切りの LLM モデルリスト (sweep のラベル兼 OLLAMA_MODEL 上書き)．
    #[arg(long, default_value = "llama3.2:latest")]
    llm_models: String,

    /// 政策レジーム (sweep では単一固定; progressive / proportional / none)．
    #[arg(long, default_value = "progressive")]
    regime: String,

    /// シミュレーション月数．
    #[arg(long, default_value_t = 240)]
    months: usize,

    /// 記憶長 L．
    #[arg(long, default_value_t = 1)]
    memory_length: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// 家計数 N (論文標準 100)．
    #[arg(long, default_value_t = 100)]
    n_agents: usize,

    /// シミュレーション月数 (240 = 20 年; --quick で 60 に縮約)．
    #[arg(long, default_value_t = 240)]
    months: usize,

    /// 記憶長 L．
    #[arg(long, default_value_t = 1)]
    memory_length: usize,

    /// 乱数シード基点 (シナリオごとに派生)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (ライブ時のみ使用)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,

    /// ライブ LLM の代わりに scripted mock を使う (オフライン検証・CI 用)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// 短縮再現 (months=60; CI スモーク用)．
    #[arg(long, default_value_t = false)]
    quick: bool,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// `sweep_summary.csv` の 1 行 (条件ごとの最終/平均マクロ指標)．
#[derive(serde::Serialize)]
struct SweepRow {
    n_agents: usize,
    tax_scale: f64,
    llm_model: String,
    regime: String,
    run: usize,
    seed: u64,
    final_month: usize,
    /// 3 年目以降 (month >= 36) の平均インフレ率．
    mean_inflation_3y: f64,
    /// 3 年目以降の平均失業率．
    mean_unemployment_3y: f64,
    /// 最終月の名目 GDP．
    final_gdp: f64,
    /// 最終月の貯蓄 Gini 係数．
    final_gini: f64,
    /// 全期間の平均労働傾向．
    mean_work_propensity: f64,
    /// 全期間の平均消費傾向．
    mean_consume_propensity: f64,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    n_agents_values: Vec<usize>,
    tax_scales: Vec<f64>,
    llm_models: Vec<String>,
    regime: String,
    months: usize,
    memory_length: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// 派生シードのラベルに使う文字列ハッシュ (explicit identity)．
fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// 税率スケール列を [min, max] step 刻みで生成する (浮動小数誤差を丸めで吸収)．
fn tax_scale_range(min: f64, max: f64, step: f64) -> Vec<f64> {
    if step <= 0.0 || max < min {
        return vec![min];
    }
    let mut out = Vec::new();
    let n = ((max - min) / step).round() as i64;
    for k in 0..=n {
        // 浮動小数誤差を 6 桁丸めで吸収する．
        out.push(((min + step * k as f64) * 1e6).round() / 1e6);
    }
    out
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let regime = parse_regime(&args.regime).unwrap_or_else(|e| panic!("{}", e));

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    let cfg = Config {
        n_agents: args.n_agents,
        months: args.months,
        memory_length: args.memory_length,
        regime,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        output_dir: output_dir.clone(),
        ..Config::default()
    };

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&cfg.output_dir);

    println!("=== Li et al. (2024) EconAgent マクロ経済 再現実験 ===");
    println!(
        "N: {} | months: {} | memory L: {} | regime: {} | tax_scale: {}",
        cfg.n_agents,
        cfg.months,
        cfg.memory_length,
        cfg.regime.label(),
        cfg.tax_scale,
    );
    println!(
        "LLM: temp={} llm_seed={} cache={} | seed: {:?}",
        cfg.llm.temperature, cfg.llm.seed, args.cache_path, cfg.seed
    );
    println!("出力先: {}", cfg.output_dir);
    println!("-------------------------------------------------");

    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json (pretty-print JSON; socsim_results::write_json に委譲)．
    {
        let path = format!("{}/config.json", cfg.output_dir);
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
    }

    // latest シンボリックリンクを再作成する (best-effort; 従来同様エラーは無視)．
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    if let Some(last) = result.metrics_history.last() {
        println!(
            "最終 GDP: {:.1} | 失業率: {:.3} | インフレ: {:.3} | Gini: {:.3}",
            last.nominal_gdp, last.unemployment_rate, last.inflation_rate, last.gini_savings
        );
    }
    println!(
        "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("メトリクス → {}/metrics.csv", cfg.output_dir);
    println!("LLM メタ   → {}/run_metadata.json", cfg.output_dir);
    println!("設定       → {}/config.json", cfg.output_dir);
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let regime: PolicyRegime = parse_regime(&args.regime).unwrap_or_else(|e| panic!("{}", e));
    let n_agents_values: Vec<usize> = split_csv(&args.n_agents_values)
        .iter()
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("不正な n_agents: {s}"))
        })
        .collect();
    let tax_scales = tax_scale_range(args.tax_scale_min, args.tax_scale_max, args.tax_scale_step);
    let llm_models = split_csv(&args.llm_models);

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = n_agents_values.len() * tax_scales.len() * llm_models.len() * args.runs;

    println!("=== Li et al. (2024) EconAgent パラメータスイープ ===");
    println!(
        "N: {} 種 | tax_scale: {} 種 | model: {} 種 | regime: {} | 試行: {} | 合計: {} 実行",
        n_agents_values.len(),
        tax_scales.len(),
        llm_models.len(),
        regime.label(),
        args.runs,
        n_total,
    );
    println!("出力先: {}", sweep_dir);
    println!("-----------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &n_agents in &n_agents_values {
        for &tax_scale in &tax_scales {
            for model in &llm_models {
                // モデル名で OLLAMA_MODEL を上書きする (sweep の LLM モデル走査)．
                std::env::set_var("OLLAMA_MODEL", model);
                for run_idx in 0..args.runs {
                    let seed = socsim_core::derive_seed(
                        args.seed,
                        &[
                            n_agents as u64,
                            label_hash(&format!("{tax_scale}")),
                            label_hash(model),
                            run_idx as u64,
                        ],
                    );

                    let cfg = Config {
                        n_agents,
                        months: args.months,
                        memory_length: args.memory_length,
                        regime,
                        tax_scale,
                        seed: Some(seed),
                        llm: LlmSettings {
                            temperature: args.llm_temperature,
                            seed: args.llm_seed,
                            cache_path: Some(args.cache_path.clone()),
                        },
                        output_dir: sweep_dir.clone(),
                        ..Config::default()
                    };

                    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                    let row = summarize(&result, n_agents, tax_scale, model, regime, run_idx, seed);
                    summary_rows.push(row);
                    done += 1;
                }
                println!(
                    "[{}/{}] N={} tax_scale={} model={} 完了 ({} 試行)",
                    done, n_total, n_agents, tax_scale, model, args.runs,
                );
            }
        }
    }

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            n_agents_values: n_agents_values.clone(),
            tax_scales: tax_scales.clone(),
            llm_models: llm_models.clone(),
            regime: regime.label().to_string(),
            months: args.months,
            memory_length: args.memory_length,
            runs: args.runs,
            seed: args.seed,
            llm_temperature: args.llm_temperature,
            llm_seed: args.llm_seed,
        };
        let path = format!("{}/sweep_config.json", sweep_dir);
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_sweep", timestamp));

    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("税率スケール別の平均 Gini (再分配が不平等に与える影響):");
    for &tax_scale in &tax_scales {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| (r.tax_scale - tax_scale).abs() < 1e-9)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg_gini = mean(&rows.iter().map(|r| r.final_gini).collect::<Vec<_>>());
        let avg_u = mean(
            &rows
                .iter()
                .map(|r| r.mean_unemployment_3y)
                .collect::<Vec<_>>(),
        );
        println!(
            "  tax_scale={:<4} → Ginī = {:.3} | ū(3y+) = {:.3}",
            tax_scale, avg_gini, avg_u
        );
    }
    println!("-----------------------------------------------------------");
    println!("サマリ → {}/sweep_summary.csv", sweep_dir);
    println!("設定   → {}/sweep_config.json", sweep_dir);
}

/// 1 実行結果を sweep の 1 行に集約する．
fn summarize(
    result: &econagent_simulation::simulation::SimulationResult,
    n_agents: usize,
    tax_scale: f64,
    model: &str,
    regime: PolicyRegime,
    run_idx: usize,
    seed: u64,
) -> SweepRow {
    let m = &result.metrics_history;
    // 3 年目以降 (month >= 36) のスライス．
    let after_3y: Vec<&econagent_simulation::metrics::MacroMetrics> =
        m.iter().filter(|r| r.month >= 36).collect();
    let mean_infl = mean(
        &after_3y
            .iter()
            .map(|r| r.inflation_rate)
            .collect::<Vec<_>>(),
    );
    let mean_unemp = mean(
        &after_3y
            .iter()
            .map(|r| r.unemployment_rate)
            .collect::<Vec<_>>(),
    );
    let mean_work = mean(&m.iter().map(|r| r.avg_work_propensity).collect::<Vec<_>>());
    let mean_consume = mean(
        &m.iter()
            .map(|r| r.avg_consume_propensity)
            .collect::<Vec<_>>(),
    );
    let (final_gdp, final_gini) = m
        .last()
        .map(|r| (r.nominal_gdp, r.gini_savings))
        .unwrap_or((0.0, 0.0));

    SweepRow {
        n_agents,
        tax_scale,
        llm_model: model.to_string(),
        regime: regime.label().to_string(),
        run: run_idx,
        seed,
        final_month: result.final_month,
        mean_inflation_3y: mean_infl,
        mean_unemployment_3y: mean_unemp,
        final_gdp,
        final_gini,
        mean_work_propensity: mean_work,
        mean_consume_propensity: mean_consume,
        cache_hit_rate: result.metadata.cache_hit_rate(),
    }
}

// ---------------------------------------------------------------------------
// reproduce (論文 headline: Phillips 曲線 / Okun の法則 / マクロ動態)
// ---------------------------------------------------------------------------

/// `reproduce_summary.json` の 1 シナリオ行 (政策レジームごとのマクロ動態)．
#[derive(serde::Serialize)]
struct ReproduceScenario {
    /// シナリオ名 (= 政策レジームラベル)．
    name: String,
    regime: String,
    /// 3 年目以降 (month ≥ 36) の平均インフレ率．
    mean_inflation_3y: f64,
    /// 3 年目以降の平均失業率．
    mean_unemployment_3y: f64,
    /// 最終月の名目 GDP．
    final_gdp: f64,
    /// 最終月の貯蓄 Gini 係数．
    final_gini: f64,
    /// Phillips 曲線の Pearson 相関 (失業率 vs インフレ率; 負が期待)．
    phillips_r: Option<f64>,
    /// Okun の法則の Pearson 相関 (失業率変化 vs GDP 成長率; 負が期待)．
    okun_r: Option<f64>,
    /// 実行月数．
    final_month: usize,
    /// この結果を保存したサブディレクトリ (Python の図生成入力)．
    results_subdir: String,
}

/// `reproduce_summary.json` のアンカー判定行．
///
/// 数値帯アンカー (`target_lo..=target_hi` に `observed` が入るか) と，符号アンカー
/// (Phillips/Okun が負か) の両方を表現する．符号アンカーは `target_hi = 0` を上限と
/// した «負であること» の帯で表す．
#[derive(serde::Serialize)]
struct ReproduceAnchor {
    name: String,
    paper_value: String,
    observed: f64,
    target_lo: f64,
    target_hi: f64,
    pass: bool,
}

/// `reproduce_summary.json` のルート．
#[derive(serde::Serialize)]
struct ReproduceSummary {
    command: &'static str,
    paper: &'static str,
    mock: bool,
    quick: bool,
    months: usize,
    n_agents: usize,
    scenarios: Vec<ReproduceScenario>,
    anchors: Vec<ReproduceAnchor>,
    n_pass: usize,
    n_anchors: usize,
}

/// 1 設定を実行する (`mock` なら scripted mock，さもなくばライブ LLM)．
///
/// mock 時は cache_path を None に倒して in-memory cache を使い，ディスク保存を
/// 抑止する (ライブ LLM 呼び出し 0)．
fn run_one(cfg: &Config, mock: bool) -> Result<SimulationResult, String> {
    if mock {
        let mock_cfg = Config {
            llm: LlmSettings {
                cache_path: None,
                ..cfg.llm.clone()
            },
            ..cfg.clone()
        };
        run_with_client(&mock_cfg, mock_decision_client())
    } else {
        run(cfg)
    }
}

fn cmd_reproduce(args: ReproduceArgs) {
    let months = if args.quick { 60 } else { args.months };

    let ts = timestamp();
    let out_dir = format!("{}/{}_reproduce", args.output_dir, ts);
    ensure_output_dir(&out_dir);
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    println!("=== Li et al. (2024) EconAgent 論文 headline 一括再現 ===");
    println!(
        "N: {} | months: {} | memory L: {} | mock: {} | quick: {}",
        args.n_agents, months, args.memory_length, args.mock, args.quick,
    );
    println!("出力先: {}", out_dir);
    println!("-------------------------------------------------");

    // 政策レジーム 3 種を再現シナリオとする (論文標準 = progressive)．
    // proportional / none は再分配が不平等・動態に与える影響の対照群．
    let scenarios_spec: [PolicyRegime; 3] = [
        PolicyRegime::Progressive,
        PolicyRegime::Proportional,
        PolicyRegime::None,
    ];

    let mut scenarios: Vec<ReproduceScenario> = Vec::new();

    for regime in scenarios_spec {
        let name = regime.label().to_string();
        let subdir = format!("{}/{}", out_dir, name);
        ensure_output_dir(&subdir);
        let seed = socsim_core::derive_seed(args.seed, &[label_hash(&name)]);
        let cfg = Config {
            n_agents: args.n_agents,
            months,
            memory_length: args.memory_length,
            regime,
            seed: Some(seed),
            llm: LlmSettings {
                temperature: args.llm_temperature,
                seed: args.llm_seed,
                cache_path: Some(args.cache_path.clone()),
            },
            output_dir: subdir.clone(),
            ..Config::default()
        };

        let result = run_one(&cfg, args.mock).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

        save_metrics(&result.metrics_history, &subdir);
        save_run_metadata(&result, &cfg, &subdir);
        let path = format!("{}/config.json", subdir);
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");

        let m = &result.metrics_history;
        let after_3y: Vec<&econagent_simulation::metrics::MacroMetrics> =
            m.iter().filter(|r| r.month >= 36).collect();
        let mean_infl = mean(
            &after_3y
                .iter()
                .map(|r| r.inflation_rate)
                .collect::<Vec<_>>(),
        );
        let mean_unemp = mean(
            &after_3y
                .iter()
                .map(|r| r.unemployment_rate)
                .collect::<Vec<_>>(),
        );
        let (final_gdp, final_gini) = m
            .last()
            .map(|r| (r.nominal_gdp, r.gini_savings))
            .unwrap_or((0.0, 0.0));

        scenarios.push(ReproduceScenario {
            name: name.clone(),
            regime: name,
            mean_inflation_3y: mean_infl,
            mean_unemployment_3y: mean_unemp,
            final_gdp,
            final_gini,
            phillips_r: phillips_correlation(m),
            okun_r: okun_correlation(m),
            final_month: result.final_month,
            results_subdir: regime.label().to_string(),
        });
    }

    // headline は progressive シナリオ (論文標準)．
    let base = scenarios
        .iter()
        .find(|s| s.regime == "progressive")
        .expect("progressive シナリオが存在する");

    // --- アンカー判定 ---
    let mut anchors: Vec<ReproduceAnchor> = Vec::new();
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: f64| {
        anchors.push(ReproduceAnchor {
            name: name.to_string(),
            paper_value: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && obs <= hi,
        });
    };

    // Phillips 曲線: インフレ vs 失業の負相関 (headline)．[-1, 0) 帯．
    let phillips = base.phillips_r.unwrap_or(f64::NAN);
    push(
        "Phillips curve: corr(unemployment, inflation) < 0",
        "negative",
        phillips,
        -1.0,
        -1e-6,
    );
    // Okun の法則: GDP 成長 vs 失業変化の負相関 (headline)．[-1, 0) 帯．
    let okun = base.okun_r.unwrap_or(f64::NAN);
    push(
        "Okun's law: corr(Δunemployment, GDP growth) < 0",
        "negative",
        okun,
        -1.0,
        -1e-6,
    );
    // インフレ率: 有限かつ有界であること (爆発・発散の検出)．
    // 帯は «オフライン mock の健全域» であり論文のタイトな ±5% 帯ではない:
    // ローカル proxy 意思決定は gpt-3.5-turbo と異なり水準は一致しない (設計どおり)．
    // headline は符号 (Phillips/Okun 負) であり，水準は定性アンカーに留める．
    push(
        "mean inflation (3y+) bounded (offline-mock sanity, not paper level)",
        "moderate (paper ~±5%)",
        base.mean_inflation_3y,
        -0.5,
        1.0,
    );
    // 失業率: 有界域 (爆発検出)．論文は 2%〜12%; mock は proxy として広めに取る．
    push(
        "mean unemployment (3y+) bounded in [0.0, 0.30]",
        "0.02-0.12 (paper)",
        base.mean_unemployment_3y,
        0.0,
        0.30,
    );

    let n_pass = anchors.iter().filter(|a| a.pass).count();
    let n_anchors = anchors.len();

    println!("シナリオ:");
    for s in &scenarios {
        let ph = s
            .phillips_r
            .map(|r| format!("{r:.3}"))
            .unwrap_or_else(|| "n/a".into());
        let ok = s
            .okun_r
            .map(|r| format!("{r:.3}"))
            .unwrap_or_else(|| "n/a".into());
        println!(
            "  [{:<12}] π̄(3y+)={:.3} ū(3y+)={:.3} GDP={:.1} Gini={:.3} | Phillips r={} Okun r={}",
            s.name, s.mean_inflation_3y, s.mean_unemployment_3y, s.final_gdp, s.final_gini, ph, ok,
        );
    }
    println!("-------------------------------------------------");
    for a in &anchors {
        let hi = if a.target_hi.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.3}", a.target_hi)
        };
        println!(
            "[{}] {:<52} obs={:.4} target=[{:.3},{}] paper={}",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
            a.paper_value,
        );
    }
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, n_anchors);

    let summary = ReproduceSummary {
        command: "reproduce",
        paper: "Li et al. (2024) EconAgent — headline macro dynamics: Phillips curve (inflation vs \
                unemployment, negative) and Okun's law (GDP growth vs unemployment change, negative)",
        mock: args.mock,
        quick: args.quick,
        months,
        n_agents: args.n_agents,
        scenarios,
        anchors,
        n_pass,
        n_anchors,
    };
    let path = format!("{}/reproduce_summary.json", out_dir);
    write_json(&summary, &path).expect("reproduce_summary.json の書き込みに失敗");

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_reproduce", ts));

    println!("サマリ → {}/reproduce_summary.json", out_dir);
    println!("各シナリオの metrics.csv / run_metadata.json / config.json を各サブディレクトリに保存しました．");
    println!("図 (マクロ時系列 + Phillips/Okun 散布) は `uv run econagent-tools reproduce` で生成できます．");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
