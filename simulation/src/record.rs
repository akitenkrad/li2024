//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` でも `sweep` / `reproduce` の子でも同一なので，
//! ここ 1 箇所で組み立てる．月次のマクロ指標，run 全体を 1 つの値で表す指標，LLM
//! ブロック，条件 1 本ぶんの集約もここに集める．
//!
//! # 旧 `metrics.csv` の列をどこへ置くか
//!
//! 旧 `metrics.csv` は `month,year,nominal_gdp,…` の wide 形式で，1 行が 1 か月
//! だった．runvault の `metrics.csv` は `run_uid,step,step_unit,scope,name,value`
//! なので，時間軸は `step` に，残りの数は 1 つずつ指標になる．
//!
//! - `month` は時間軸そのもの (`step`)．値としては書かない．
//! - `year` は書かない．`FiscalRewardMechanism` が `year = month / 12` で埋めて
//!   いる導出値で，`month` から一意に決まる．2 箇所に置くと食い違う余地ができる
//!   だけで情報は増えない．
//! - 残る 9 列はいずれも系列 ID を持たない «経済全体の量» なので，`scope=run` の
//!   ステップ指標になる (名前は旧列名のまま)．家計は個別に観測していないので
//!   `observation` / `terminal` は書かない — 予算を使い切る以外の終わり方が無く，
//!   終端の時刻は run スコープの `final_month` が言い尽くす．

use runvault::{Llm, Replication, Run, Target, Work};

use crate::metrics::{mean, okun_correlation, phillips_correlation, MacroMetrics};
use crate::simulation::SimulationResult;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "econagent";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "li2024";
/// 分野．家計の初期賃金 (パレート分布)・年齢・就業 Bernoulli ドロー・市場調整の
/// 一様乱数がいずれも乱数駆動で `master_seed` が要るので `simulation`．意思決定は
/// LLM が担うが，測っているのはモデルの安全性ではなくマクロ経済の定型化された事実
/// なので `llm-safety` ではない．LLM 側の同一性は `run.json` の `llm` ブロックが持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// このモデルの 1 刻みは «経済の 1 か月» — 全家計が知覚・意思決定し，労働/財市場が
/// 清算され，財政 (課税・再分配) と金融 (Taylor ルール) が回り，記憶が更新される
/// までの 1 巡である．語彙 (`schema/v1/vocabulary.toml`) に `month` は無く，1 巡を
/// 表す語は `round`．同じく LLM 駆動で «暦の 1 単位 = 1 巡» の zhao2024 / gao2023 と
/// 揃える．
const T_UNIT: &str = "round";

/// 月次指標と run 全体の指標の粒度．いずれも経済全体の集約なので `run`．
const SCOPE: &str = "run";

/// `reproduce` の親が持つ «条件をまたいだ集約» の粒度．
pub const SWEEP_SCOPE: &str = "sweep";

/// この再現実験が対象としている論文．
///
/// `run` も `sweep` / `reproduce` の子も同じ対象を持つ — 掃引は税率スケール・
/// エージェント数・モデルを変えて定型化された事実の頑健性を見るためのもので，
/// 別の対象を持たない．
pub fn replication() -> Replication {
    let mut work = Work::arxiv("2310.10436")
        .title(
            "EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic \
             Activities",
        )
        .year(2024)
        .source_version("acl-2024");
    // vault 側の同定にも使えるよう paper-id も残す (work_id は arXiv 側)．
    work.paper_id = Some("P00001795".to_string());
    Replication::new(work)
        .target(Target::figure("fig2", "Figure 2"))
        .target(Target::figure("fig3", "Figure 3"))
        .target(Target::claim(
            "macro-regularities-emerge-from-llm-decisions",
            "Classical macroeconomic regularities — the Phillips curve and Okun's law — emerge \
             from LLM agents' monthly work and consumption decisions without pre-defined rules",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/li2024/設計書.md")
}

// --------------------------------------------------------------------------- //
// LLM ブロック
// --------------------------------------------------------------------------- //

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める．推測しているのは分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.2:latest` のような動くエイリアスであることが
/// 多い．socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // 知覚プロンプトとリフレクションプロンプトは家計ごとに組み立てられ，固定の
        // system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// --------------------------------------------------------------------------- //
// シミュレーション 1 本ぶんの記録
// --------------------------------------------------------------------------- //

/// シミュレーション 1 本ぶんを run へ書く (`run` と掃引/再現の子で共通)．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    log_monthly(run, &result.metrics_history);
    log_run_scope(run, result);
}

/// 月次のマクロ指標を `metrics.csv` に書く (1 か月 1 回)．
///
/// `month` は `step` として渡すので値には現れない．`year` は書かない (`month` から
/// 導出される)．率 (`unemployment_rate` / `inflation_rate`) は分母が 0 のときに
/// «定義できない» が，[`crate::metrics`] の関数はどちらも 0.0 を返す — 関数が返した
/// 0 は欠測ではないのでそのまま記録する．
fn log_monthly(run: &mut Run, history: &[MacroMetrics]) {
    for m in history {
        run.log_metrics_at(
            m.month as u64,
            T_UNIT,
            SCOPE,
            &[
                ("nominal_gdp", m.nominal_gdp),
                ("price", m.price),
                ("interest_rate", m.interest_rate),
                ("unemployment_rate", m.unemployment_rate),
                ("inflation_rate", m.inflation_rate),
                ("gini_savings", m.gini_savings),
                ("avg_work_propensity", m.avg_work_propensity),
                ("avg_consume_propensity", m.avg_consume_propensity),
                ("redistribution", m.redistribution),
            ],
        )
        .unwrap_or_else(|e| panic!("month {} の指標の記録に失敗: {e}", m.month));
    }
}

/// run 全体を 1 つの値で表す指標．
///
/// `final_month` は socsim エンジンが数えた完了ステップ数で，旧 `sweep_summary.csv`
/// の `final_month` と同じ値である．実行時間は `status.json` の `duration_sec` が
/// 正本なので指標にはしない．`n_units` (観測主体の数) は書かない — 家計を個別に
/// 観測していないので，このモデルに «観測主体» は無い．
///
/// LLM の呼び出し数と cache-hit は旧 `run_metadata.json` から移した．model /
/// endpoint / temperature は数ではないので `run.json` の `llm` ブロックが持つ．
fn log_run_scope(run: &mut Run, result: &SimulationResult) {
    let calls = result.metadata.total();
    let mut values: Vec<(&str, f64)> = vec![
        ("final_month", result.final_month as f64),
        ("llm_calls", calls as f64),
        ("llm_cache_hits", result.metadata.cache_hits() as f64),
    ];
    // 呼び出しが 1 本も無いときの cache-hit 率は «0» ではなく «定義できない»．
    // 欠測を 0 で埋めず，率の行そのものを書かない．
    if calls > 0 {
        values.push(("llm_cache_hit_rate", result.metadata.cache_hit_rate()));
    }
    run.log_metrics(SCOPE, &values)
        .expect("run スコープの指標の記録に失敗");
}

// --------------------------------------------------------------------------- //
// 1 本ぶんの集約 (コンソール表示と reproduce 親の集約に使う)
// --------------------------------------------------------------------------- //

/// シミュレーション 1 本を 1 行に集約した値．
///
/// ディスクには «この構造体» としては書かない．旧 `sweep_summary.csv` の 1 行に
/// 相当するが，どの値も子 run の月次指標から組み直せるので，掃引ではコンソール表示に
/// だけ使い，Python 側は子 run から同じ表を作る (同じ数を 2 箇所に置かない)．
/// `reproduce` では判定の対象になる 4 つだけが親の `scope=sweep` 指標に載る．
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// 3 年目以降 (month ≥ 36) の平均インフレ率．
    pub mean_inflation_3y: f64,
    /// 3 年目以降の平均失業率．
    pub mean_unemployment_3y: f64,
    /// 最終月の名目 GDP．
    pub final_gdp: f64,
    /// 最終月の貯蓄 Gini 係数．
    pub final_gini: f64,
    /// 全期間の平均労働傾向．
    pub mean_work_propensity: f64,
    /// 全期間の平均消費傾向．
    pub mean_consume_propensity: f64,
    /// Phillips 曲線の Pearson 相関 (失業率 vs インフレ率; 負が期待)．
    pub phillips_r: Option<f64>,
    /// Okun の法則の Pearson 相関 (失業率変化 vs GDP 成長率; 負が期待)．
    pub okun_r: Option<f64>,
}

impl RunSummary {
    /// 月次指標の履歴から集約する．
    pub fn from_history(m: &[MacroMetrics]) -> Self {
        let after_3y: Vec<&MacroMetrics> = m.iter().filter(|r| r.month >= 36).collect();
        let (final_gdp, final_gini) = m
            .last()
            .map(|r| (r.nominal_gdp, r.gini_savings))
            .unwrap_or((0.0, 0.0));
        RunSummary {
            mean_inflation_3y: mean(
                &after_3y
                    .iter()
                    .map(|r| r.inflation_rate)
                    .collect::<Vec<_>>(),
            ),
            mean_unemployment_3y: mean(
                &after_3y
                    .iter()
                    .map(|r| r.unemployment_rate)
                    .collect::<Vec<_>>(),
            ),
            final_gdp,
            final_gini,
            mean_work_propensity: mean(
                &m.iter().map(|r| r.avg_work_propensity).collect::<Vec<_>>(),
            ),
            mean_consume_propensity: mean(
                &m.iter()
                    .map(|r| r.avg_consume_propensity)
                    .collect::<Vec<_>>(),
            ),
            phillips_r: phillips_correlation(m),
            okun_r: okun_correlation(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runvault::meta::TargetKind;

    #[test]
    fn the_work_id_agrees_with_the_arxiv_id() {
        let research: runvault::meta::Research = replication().into();
        let work = research.work.expect("再現実験なので work がある");
        assert_eq!(work.work_id, "arxiv:2310.10436");
        assert_eq!(work.paper_id.as_deref(), Some("P00001795"));
    }

    #[test]
    fn the_targets_are_the_two_figures_and_the_headline_claim() {
        let research: runvault::meta::Research = replication().into();
        assert_eq!(research.targets.len(), 3);
        assert!(matches!(research.targets[0].kind, TargetKind::Figure));
        assert!(matches!(research.targets[1].kind, TargetKind::Figure));
        assert!(matches!(research.targets[2].kind, TargetKind::Claim));
    }

    #[test]
    fn a_run_that_reproduces_the_paper_passes_the_research_checks() {
        let research: runvault::meta::Research = replication().into();
        runvault::verify::check_research(&research).expect("research の検査に失敗");
    }

    #[test]
    fn the_provider_comes_from_the_endpoint() {
        assert_eq!(llm_block("m", "mock://scripted", 0.0).provider, "mock");
        assert_eq!(
            llm_block("m", "https://api.openai.com/v1", 0.0).provider,
            "openai"
        );
        assert_eq!(
            llm_block("m", "http://localhost:11434", 0.0).provider,
            "ollama"
        );
    }
}
