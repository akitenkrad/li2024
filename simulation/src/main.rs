//! Li et al. (2024) "EconAgent" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で LLM 駆動マクロ経済 ABM を実行する．
//! `sweep`     : エージェント数 × 税率スケール × LLM モデルを走査する．
//!               親 run 1 本 + (セル × 試行) ごとの子 run．
//! `reproduce` : 論文の headline マクロ動態 (インフレ/失業/GDP) と Phillips 曲線
//!               (インフレ vs 失業; 負相関)・Okun の法則 (GDP 成長 vs 失業変化;
//!               負相関) を一括再現する．親 run 1 本 + 政策レジームごとの子 run で，
//!               観測値は親の sweep スコープ指標に，論文の報告値は `reference.csv` に
//!               入る．
//!
//! サブコマンド 1 回が runvault の run 1 本になる．出力の置き場と同一性 (run ディレ
//! クトリ・`config.json`・`metrics.csv`) は runvault が持つので，ここではタイム
//! スタンプ付きディレクトリも `latest` symlink も作らない．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};

use econagent_simulation::config::{parse_regime, Config, LlmSettings, PolicyRegime};
use econagent_simulation::llm::{build_live_client, EconClient};
use econagent_simulation::metrics::mean;
use econagent_simulation::record::{self, RunSummary, DOMAIN, EXPERIMENT, REPO_ID, SWEEP_SCOPE};
use econagent_simulation::simulation::{
    mock_decision_client, run_with_client_observed, SimulationResult,
};
use socsim_llm::LlmClient;

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
    /// Development run: write it under results/_scratch/ so it is never synced to the vault.
    #[arg(long, global = true)]
    scratch: bool,

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
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

/// スイープ親 run の実験条件 (グリッド定義そのもの)．
#[derive(serde::Serialize)]
struct SweepParameters {
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

/// `reproduce` 親 run の実験条件 (全シナリオ共通の設定)．
///
/// `mock` を持つのは，scripted mock とライブ LLM が別の世界を回すためである
/// (子 run 側は `run.json` の `llm` ブロックが `provider = "mock"` を持つ)．
#[derive(serde::Serialize)]
struct ReproduceParameters {
    n_agents: usize,
    months: usize,
    memory_length: usize,
    seed: u64,
    mock: bool,
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

/// mock は永続キャッシュを持たないので `cache_path` を落とす．
fn llm_settings(temperature: f32, seed: u64, cache_path: &str, mock: bool) -> LlmSettings {
    LlmSettings {
        temperature,
        seed,
        cache_path: (!mock).then(|| cache_path.to_string()),
    }
}

/// LLM キャッシュの置き場を用意する (ライブ実行のみ; mock は in-memory)．
fn ensure_cache_dir(cfg: &Config) {
    if let Some(parent) = cfg
        .llm
        .cache_path
        .as_deref()
        .and_then(|path| Path::new(path).parent())
    {
        let _ = fs::create_dir_all(parent);
    }
}

/// LLM クライアントを 1 本組む．
///
/// `run.json` の `llm` ブロックに書くモデル名と endpoint は，実際に応答する
/// バックエンドから採らないと意味を持たないので，組み立ては `Run::start` より前に
/// 置く．
fn build_client(cfg: &Config, mock: bool) -> EconClient {
    if mock {
        mock_decision_client()
    } else {
        build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"))
    }
}

/// クライアントが名乗ったモデル/endpoint から `llm` ブロックを組む．
fn llm_block_of(client: &EconClient, temperature: f32) -> runvault::Llm {
    record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        temperature,
    )
}

/// LLM 呼び出しの要約を 1 行で出す (旧 `run_metadata.json` のコンソール表示)．
fn print_llm_summary(result: &SimulationResult) {
    println!(
        "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs, scratch: bool) {
    let regime = parse_regime(&args.regime).unwrap_or_else(|e| panic!("{}", e));

    // シードを実体化してから記録する．--seed 省略時にシミュレーション側で
    // rand::random に落とすと，実際に使われたシードがどこにも残らない．
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);

    let cfg = Config {
        n_agents: args.n_agents,
        months: args.months,
        memory_length: args.memory_length,
        regime,
        tax_scale: args.tax_scale,
        seed: Some(seed),
        llm: llm_settings(args.llm_temperature, args.llm_seed, &args.cache_path, false),
        ..Config::default()
    };
    ensure_cache_dir(&cfg);

    // クライアントは run を開始する前に組む (`llm` ブロックのため)．
    let client = build_client(&cfg, false);
    let llm = llm_block_of(&client, cfg.llm.temperature);

    let parameters = cfg.to_run_config_json();
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .scratch(scratch)
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(seed)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

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
        "LLM: temp={} llm_seed={} cache={} | seed: {}",
        cfg.llm.temperature, cfg.llm.seed, args.cache_path, seed
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    // 進捗の 1 単位は 1 か月．費用がそこにあり，1 か月は全家計に労働と消費の
    // 決定を尋ねる — その 1 つ 1 つがモデル呼び出しになる．run の試行は 1 本
    // しかないので，試行を単位にすると最初の行から最後の行まで何も出ない．
    // このモデルには早期停止 (`request_stop`) が無く必ず months か月まわるので，
    // 分母は正確である．
    let mut stage = rv.stage("months", cfg.months);
    let result = run_with_client_observed(&cfg, client, |_| stage.tick())
        .unwrap_or_else(|e| panic!("実行に失敗: {}", e));
    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();
    record::log_simulation(&mut rv, &result);

    if let Some(last) = result.metrics_history.last() {
        println!(
            "最終 GDP: {:.1} | 失業率: {:.3} | インフレ: {:.3} | Gini: {:.3}",
            last.nominal_gdp, last.unemployment_rate, last.inflation_rate, last.gini_savings
        );
    }
    print_llm_summary(&result);

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("メトリクス → {}/metrics.csv", dir.display());
    println!("設定       → {}/config.json", dir.display());
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs, scratch: bool) {
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

    let n_total = n_agents_values.len() * tax_scales.len() * llm_models.len() * args.runs;

    let sweep_parameters = SweepParameters {
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

    // 親 run: グリッド定義そのものを parameters に持つ．個別条件の指標は書かない
    // (どれも子 run の月次指標から組み直せる)．親は 1 本のシミュレーションではない
    // ので master_seed を名乗らず，base seed は /parameters.seed と seed_pointers
    // 経由で execution_hash に残る．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .scratch(scratch)
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

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
    println!("出力先: {}", parent.dir().display());
    println!("-----------------------------------------------------------");

    // コンソールの «税率スケール別» 表のためだけに集約を持つ (ディスクには書かない)．
    let mut cells: Vec<(f64, RunSummary)> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &n_agents in &n_agents_values {
        // stage は家計数ごとに 1 つ．その下の税率スケールとモデルは全部この 1 つに
        // 収める — 条件ごとに開け直すと小さな 100% がグリッドの数だけ並ぶだけで，
        // 掃引全体のどこにいるかが読めなくなる．
        //
        // 家計数で «だけ» 分けるのは，1 か月の費用を変える要因のうち時間的に
        // 偏っているのがそれだけだからである．1 か月は全家計に決定を尋ねるので
        // 費用は N に比例し，既定の `--n-agents-values 50,100,300` は 6 倍違う．
        // しかも家計数は最も外のループなので，1 つの stage にまとめると前半
        // (N=50) から外挿した見積もりが後半 (N=300) を数分の 1 に言う．実測でも
        // 同じ 3 か月の区間が N=5 で 10s，N=10 で 21s だった．
        //
        // モデルも 1 か月の費用を変えるが，`--llm-models` は最も内側のループなので
        // stage の中で均等に入れ替わる．混ざり方が最初から最後まで一様なら平均の
        // 速度は代表値として正しく，見積もりは狂わない — 偏っている家計数とは
        // 事情が違う．税率スケールは税の式を変えるだけで仕事の量を変えない．
        //
        // 比を重みで書くことはできない — 呼び出し 1 回の所要はプロンプト長・
        // エンドポイント・キャッシュの当たり方で決まり，走らせる前には分からない．
        let mut stage = parent.stage(
            &format!("N={n_agents}"),
            tax_scales.len() * llm_models.len() * args.runs * args.months,
        );
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
                        llm: llm_settings(
                            args.llm_temperature,
                            args.llm_seed,
                            &args.cache_path,
                            false,
                        ),
                        ..Config::default()
                    };
                    ensure_cache_dir(&cfg);

                    let client = build_client(&cfg, false);
                    let llm = llm_block_of(&client, cfg.llm.temperature);

                    // 子は «セル 1 点の試行 1 本» — 手で回した `run` と同じ形の
                    // parameters を書くので，同じ条件なら config_hash が一致する．
                    // 名前を `run` にしないのは，`runvault path --subcommand run` が
                    // 手で回した run と掃引の子のどちらを返すか分からなくなるため．
                    let parameters = cfg.to_run_config_json();
                    let mut child = Run::start(
                        RunOptions::new(EXPERIMENT, "sweep-point")
                            .scratch(scratch)
                            .repo_id(REPO_ID)
                            .domain(DOMAIN)
                            .results_root(&args.output_dir)
                            .parameters(&parameters)
                            .expect("runvault: 子 run の parameters の組み立てに失敗")
                            .seed_pointers(["/seed"])
                            .master_seed(seed)
                            .replicate_index(run_idx as u64)
                            .lineage(Lineage {
                                sweep_id: Some(sweep_id.clone()),
                                parent_run_uid: Some(parent_run_uid.clone()),
                                ..Default::default()
                            })
                            .llm(llm)
                            .replication(record::replication()),
                    )
                    .expect("runvault: 子 run の開始に失敗");

                    let result = run_with_client_observed(&cfg, client, |_| stage.tick())
                        .unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                    record::log_simulation(&mut child, &result);
                    child.finish().expect("runvault: 子 run の完了に失敗");

                    cells.push((tax_scale, RunSummary::from_history(&result.metrics_history)));
                    done += 1;
                }
                println!(
                    "[{}/{}] N={} tax_scale={} model={} 完了 ({} 試行)",
                    done, n_total, n_agents, tax_scale, model, args.runs,
                );
            }
        }
        // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
        // 食い違うダイジェストを持つことになる．
        stage.close();
    }

    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("税率スケール別の平均 Gini (再分配が不平等に与える影響):");
    for &tax_scale in &tax_scales {
        let rows: Vec<&RunSummary> = cells
            .iter()
            .filter(|(s, _)| (s - tax_scale).abs() < 1e-9)
            .map(|(_, r)| r)
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
            "  tax_scale={:<4} → Ginī = {:.3} | ū(3y+) = {:.3}",
            tax_scale, avg_gini, avg_u
        );
    }

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");
    println!("-----------------------------------------------------------");
    println!("スイープ定義 → {}/config.json", dir.display());
    println!("各セルの試行は子 run (subcommand=sweep-point) の metrics.csv にあります");
}

// ---------------------------------------------------------------------------
// reproduce (論文 headline: Phillips 曲線 / Okun の法則 / マクロ動態)
// ---------------------------------------------------------------------------

/// 論文の報告値 1 つ．`reference.csv` の 1 行になる．
///
/// 入るのは **論文が印字した数だけ** である．インフレ率 −5%〜5%・失業率 2%〜12% は
/// 論文が «帯» として報告した範囲で 1 つの数ではないので (`reference.csv` は
/// name → 1 値)，コンソールとドキュメントに残す．本実装が置いたオフライン mock の
/// 健全域 ([`ANCHORS`] の `lo`/`hi`) は論文の主張ではないのでもちろん記録しない．
struct PaperValue {
    /// 観測側の指標名と揃える (差分がそのまま取れる)．
    name: &'static str,
    value: f64,
    /// `research.targets[]` の target_id．
    target_id: &'static str,
    source: &'static str,
}

/// 論文 Fig. 3 が報告した Pearson 相関 (headline = 論文標準の累進課税シナリオ)．
const PAPER_VALUES: [PaperValue; 2] = [
    PaperValue {
        name: "phillips_r_progressive",
        value: -0.619,
        target_id: "fig3",
        source: "Li et al. (2024), Figure 3 — Phillips curve, Pearson r = -0.619 (p < 0.01)",
    },
    PaperValue {
        name: "okun_r_progressive",
        value: -0.918,
        target_id: "fig3",
        source: "Li et al. (2024), Figure 3 — Okun's law, Pearson r = -0.918 (p < 0.001)",
    },
];

/// 本実装が置いたアンカー帯 (論文の主張ではないので記録せず表示だけする)．
struct Anchor {
    label: &'static str,
    /// 論文が報告した内容 (表示のみ)．
    paper: &'static str,
    /// headline シナリオの [`RunSummary`] のどの量を見るか．
    metric: &'static str,
    lo: f64,
    hi: f64,
}

const ANCHORS: [Anchor; 4] = [
    // Phillips 曲線: インフレ vs 失業の負相関 (headline)．[-1, 0) 帯．
    Anchor {
        label: "Phillips curve: corr(unemployment, inflation) < 0",
        paper: "r = -0.619 (p < 0.01)",
        metric: "phillips_r",
        lo: -1.0,
        hi: -1e-6,
    },
    // Okun の法則: GDP 成長 vs 失業変化の負相関 (headline)．[-1, 0) 帯．
    Anchor {
        label: "Okun's law: corr(Δunemployment, GDP growth) < 0",
        paper: "r = -0.918 (p < 0.001)",
        metric: "okun_r",
        lo: -1.0,
        hi: -1e-6,
    },
    // インフレ率: 有限かつ有界であること (爆発・発散の検出)．
    // 帯は «オフライン mock の健全域» であり論文のタイトな ±5% 帯ではない:
    // ローカル proxy 意思決定は gpt-3.5-turbo と異なり水準は一致しない (設計どおり)．
    // headline は符号 (Phillips/Okun 負) であり，水準は定性アンカーに留める．
    Anchor {
        label: "mean inflation (3y+) bounded (offline-mock sanity, not paper level)",
        paper: "-5%..5% (paper)",
        metric: "mean_inflation_3y",
        lo: -0.5,
        hi: 1.0,
    },
    // 失業率: 有界域 (爆発検出)．論文は 2%〜12%; mock は proxy として広めに取る．
    Anchor {
        label: "mean unemployment (3y+) bounded in [0.0, 0.30]",
        paper: "2%..12% (paper)",
        metric: "mean_unemployment_3y",
        lo: 0.0,
        hi: 0.30,
    },
];

fn cmd_reproduce(args: ReproduceArgs, scratch: bool) {
    let months = if args.quick { 60 } else { args.months };

    let parent_parameters = ReproduceParameters {
        n_agents: args.n_agents,
        months,
        memory_length: args.memory_length,
        seed: args.seed,
        mock: args.mock,
        llm_temperature: args.llm_temperature,
        llm_seed: args.llm_seed,
    };

    // 親 run: 全シナリオ共通の設定を parameters に持ち，シナリオをまたいだ判定の
    // 対象になる量を sweep スコープの指標として書く．論文の報告値は reference.csv に
    // 入る．親は単一の master_seed を持たない (シナリオごとの子が派生シードを持つ)．
    let mut parent = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .scratch(scratch)
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parent_parameters)
            .expect("runvault: reproduce の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: reproduce 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: reproduce 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Li et al. (2024) EconAgent 論文 headline 一括再現 ===");
    println!(
        "N: {} | months: {} | memory L: {} | mock: {} | quick: {}",
        args.n_agents, months, args.memory_length, args.mock, args.quick,
    );
    println!("出力先: {}", parent.dir().display());
    println!("-------------------------------------------------");

    // 政策レジーム 3 種を再現シナリオとする (論文標準 = progressive)．
    // proportional / none は再分配が不平等・動態に与える影響の対照群．
    // レジームごとに世界を別々に回す (シードもシナリオ名から派生する) ので，
    // シナリオ 1 つが子 run 1 本になる．
    let scenarios_spec: [PolicyRegime; 3] = [
        PolicyRegime::Progressive,
        PolicyRegime::Proportional,
        PolicyRegime::None,
    ];

    let mut summaries: Vec<(&'static str, RunSummary)> = Vec::new();

    // レジームは税の式を変えるだけで，1 か月あたりの家計数もモデル呼び出し数も
    // 変えない．よって 3 シナリオを 1 つの stage で通して数える．早期停止が
    // 無いので分母は正確である．
    let mut stage = parent.stage("months", scenarios_spec.len() * months);

    for (idx, regime) in scenarios_spec.into_iter().enumerate() {
        let name = regime.label();
        let seed = socsim_core::derive_seed(args.seed, &[label_hash(name)]);
        let cfg = Config {
            n_agents: args.n_agents,
            months,
            memory_length: args.memory_length,
            regime,
            seed: Some(seed),
            llm: llm_settings(
                args.llm_temperature,
                args.llm_seed,
                &args.cache_path,
                args.mock,
            ),
            ..Config::default()
        };
        ensure_cache_dir(&cfg);

        let client = build_client(&cfg, args.mock);
        let llm = llm_block_of(&client, cfg.llm.temperature);

        let parameters = cfg.to_run_config_json();
        let mut child = Run::start(
            RunOptions::new(EXPERIMENT, "reproduce-scenario")
                .scratch(scratch)
                .repo_id(REPO_ID)
                .domain(DOMAIN)
                .results_root(&args.output_dir)
                .parameters(&parameters)
                .expect("runvault: 子 run の parameters の組み立てに失敗")
                .seed_pointers(["/seed"])
                .master_seed(seed)
                .replicate_index(idx as u64)
                .lineage(Lineage {
                    sweep_id: Some(sweep_id.clone()),
                    parent_run_uid: Some(parent_run_uid.clone()),
                    ..Default::default()
                })
                .llm(llm)
                .replication(record::replication()),
        )
        .expect("runvault: 子 run の開始に失敗");

        let result = run_with_client_observed(&cfg, client, |_| stage.tick())
            .unwrap_or_else(|e| panic!("実行に失敗 ({name}): {e}"));
        record::log_simulation(&mut child, &result);
        child.finish().expect("runvault: 子 run の完了に失敗");

        summaries.push((name, RunSummary::from_history(&result.metrics_history)));
    }

    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    // --- 親の sweep スコープ指標 (シナリオごとの判定対象) ---
    //
    // 条件は値ではなく «どの数か» を指す名前なので，`phillips_r_progressive` の
    // ように名前に畳む．最終 GDP・最終 Gini・月ごとの値は子 run の metrics.csv が
    // 正本なので親には重ねない．相関は標本不足や分散 0 のとき «定義できない» ので，
    // `None` の行は書かない (欠測を 0 で埋めない)．
    let mut observed: Vec<(String, f64)> = Vec::new();
    for (name, s) in &summaries {
        observed.push((format!("mean_inflation_3y_{name}"), s.mean_inflation_3y));
        observed.push((
            format!("mean_unemployment_3y_{name}"),
            s.mean_unemployment_3y,
        ));
        if let Some(r) = s.phillips_r {
            observed.push((format!("phillips_r_{name}"), r));
        }
        if let Some(r) = s.okun_r {
            observed.push((format!("okun_r_{name}"), r));
        }
    }
    let observed_refs: Vec<(&str, f64)> = observed.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    parent
        .log_metrics(SWEEP_SCOPE, &observed_refs)
        .expect("reproduce 親の集約指標の記録に失敗");

    // --- 論文の報告値 (reference.csv) ---
    for pv in &PAPER_VALUES {
        parent
            .log_reference(pv.name, pv.value)
            .scope(SWEEP_SCOPE)
            .target(pv.target_id)
            .source(pv.source)
            .send()
            .unwrap_or_else(|e| panic!("論文値 {} の記録に失敗: {e}", pv.name));
    }

    // --- コンソール出力 ---
    println!("シナリオ:");
    for (name, s) in &summaries {
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
            name, s.mean_inflation_3y, s.mean_unemployment_3y, s.final_gdp, s.final_gini, ph, ok,
        );
    }

    // headline は progressive シナリオ (論文標準)．
    let base = summaries
        .iter()
        .find(|(name, _)| *name == "progressive")
        .map(|(_, s)| s)
        .expect("progressive シナリオが存在する");

    println!("-------------------------------------------------");
    let by_metric = |metric: &str| -> f64 {
        match metric {
            "phillips_r" => base.phillips_r.unwrap_or(f64::NAN),
            "okun_r" => base.okun_r.unwrap_or(f64::NAN),
            "mean_inflation_3y" => base.mean_inflation_3y,
            "mean_unemployment_3y" => base.mean_unemployment_3y,
            other => panic!("未知のアンカー指標: {other}"),
        }
    };
    let mut n_pass = 0usize;
    for a in &ANCHORS {
        let obs = by_metric(a.metric);
        let pass = obs >= a.lo && obs <= a.hi;
        if pass {
            n_pass += 1;
        }
        println!(
            "[{}] {:<52} obs={:.4} band=[{:.3},{:.3}] paper={}",
            if pass { "PASS" } else { "OFF " },
            a.label,
            obs,
            a.lo,
            a.hi,
            a.paper,
        );
    }
    println!("-------------------------------------------------");
    println!(
        "{}/{} アンカーが in-band (帯は本実装の設定)",
        n_pass,
        ANCHORS.len()
    );
    println!("論文値との差 (Fig. 3 の Pearson 相関):");
    for pv in &PAPER_VALUES {
        if let Some((_, obs)) = observed.iter().find(|(n, _)| n == pv.name) {
            println!(
                "  {:<26} obs={:+.4} paper={:+.4} diff={:+.4}",
                pv.name,
                obs,
                pv.value,
                obs - pv.value
            );
        }
    }

    let dir = parent
        .finish()
        .expect("runvault: reproduce 親 run の完了に失敗");
    println!("-------------------------------------------------");
    println!("集約   → {}/metrics.csv (scope=sweep)", dir.display());
    println!("論文値 → {}/reference.csv", dir.display());
    println!("各シナリオの月次指標は子 run (subcommand=reproduce-scenario) にあります．");
    println!(
        "図 (マクロ時系列 + Phillips/Okun 散布) は `uv run econagent-tools reproduce` で生成できます．"
    );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let scratch = cli.scratch;
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args, scratch),
        Commands::Sweep(args) => cmd_sweep(args, scratch),
        Commands::Reproduce(args) => cmd_reproduce(args, scratch),
    }
}
