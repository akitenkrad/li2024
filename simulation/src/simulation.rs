//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で家計初期化 (賃金
//!   パレート分布・年齢分布) の init RNG を，`derive_seed(root, &[1])` で engine
//!   RNG (= 就業 Bernoulli ドロー・市場調整の一様乱数) を派生する．bit 単位で
//!   再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き
//!   Ollama→OpenAI フォールバッククライアントに閉じ込め，`temperature=0`/`seed`
//!   固定 + プロンプト→応答キャッシュで擬似決定論化する．モデル・endpoint・温度は
//!   `run.json` の `llm` ブロックへ，呼び出し数と cache-hit は run スコープの指標へ
//!   落ちる ([`crate::record`])．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;

use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{SequentialScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};

use crate::config::Config;
use crate::llm::EconClient;
use crate::mechanisms::{
    FiscalRewardMechanism, LlmDecisionMechanism, MacroPolicyMechanism, MarketClearingMechanism,
    MemoryUpdateMechanism, SharedClient, SharedMetadata, SharedMetrics,
};
use crate::metrics::MacroMetrics;
use crate::world::{EconWorld, Household, MacroEnv};

/// 家計初期化用 RNG ラベル (賃金パレート分布・年齢分布)．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン (= 就業ドロー・市場調整一様乱数) 用 RNG ラベル．
const RNG_ENGINE: u64 = 1;

/// 2018 U.S. 成人年齢分布の簡易近似 (区間と相対重み)．
///
/// 18-29, 30-44, 45-59, 60-79 の 4 区間に重み付けし，区間内は一様にサンプリング
/// する (決定論的 init RNG)．
const AGE_BANDS: [(u32, u32, f64); 4] = [
    (18, 29, 0.22),
    (30, 44, 0.28),
    (45, 59, 0.27),
    (60, 79, 0.23),
];

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// 月次マクロ指標の履歴 (metrics.csv の行)．
    pub metrics_history: Vec<MacroMetrics>,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名 (run_metadata 用)．
    pub llm_model: String,
    /// LLM endpoint (run_metadata 用; primary)．
    pub llm_endpoint: String,
    /// 実行した月数 (= 完了ステップ数)．
    pub final_month: usize,
}

/// パレート分布から時給をサンプリングする．
///
/// 逆関数法: `x = x_min / U^(1/α)`，`U ~ Uniform(0,1]`．`α` が小さいほど裾が重く
/// 不平等が大きい．
fn sample_pareto_wage(rng: &mut SimRng, x_min: f64, alpha: f64) -> f64 {
    let u: f64 = rng.gen_range(1e-9..=1.0);
    x_min / u.powf(1.0 / alpha)
}

/// 年齢分布から年齢をサンプリングする (決定論的 init RNG)．
fn sample_age(rng: &mut SimRng) -> u32 {
    let total: f64 = AGE_BANDS.iter().map(|(_, _, w)| w).sum();
    let mut x: f64 = rng.gen_range(0.0..total);
    for &(lo, hi, w) in AGE_BANDS.iter() {
        if x < w {
            return rng.gen_range(lo..=hi);
        }
        x -= w;
    }
    AGE_BANDS[AGE_BANDS.len() - 1].1
}

/// 世界状態を初期化する (家計生成 + マクロ環境)．
///
/// 賃金はパレート分布，年齢は 2018 U.S. 近似分布から init RNG で決定論的に
/// サンプリングする (socsim コア層)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> EconWorld {
    let (brackets, rates) = cfg.tax_schedule();

    let mut households: BTreeMap<AgentId, Household> = BTreeMap::new();
    for i in 0..cfg.n_agents as u64 {
        let wage = sample_pareto_wage(rng, cfg.wage_min, cfg.wage_pareto_alpha);
        let age = sample_age(rng);
        households.insert(AgentId(i), Household::new(wage, age, cfg.init_savings));
    }

    let env = MacroEnv {
        price: cfg.init_price,
        interest_rate: cfg.natural_rate + cfg.target_inflation,
        inventory: 0.0,
        productivity: cfg.productivity,
        redistribution: 0.0,
        tax_brackets: brackets,
        tax_rates: rates,
        price_history: vec![cfg.init_price], // n=0 年の基準物価．
        year_price_buffer: Vec::new(),
        year_unemp_buffer: Vec::new(),
        natural_rate: cfg.natural_rate,
        target_inflation: cfg.target_inflation,
        natural_unemployment: cfg.natural_unemployment,
        alpha_pi: cfg.alpha_pi,
        alpha_u: cfg.alpha_u,
        alpha_w: cfg.alpha_w,
        alpha_p: cfg.alpha_p,
    };

    EconWorld {
        clock: SimClock::new(cfg.months as u64),
        households,
        env,
    }
}

// クライアントを中で組む入口 (旧 `run`) は置かない．`run.json` の `llm` ブロックに
// 書くモデル名と endpoint は，実際に応答するバックエンドから採らないと意味を持たない
// ので，クライアントは `Run::start` より前に呼び出し側が組む．中で組む入口が残って
// いると，`llm` ブロックを埋めないまま記録できてしまう．

/// オフライン検証用の scripted 意思決定クライアント (ライブ LLM 不要)．
///
/// `reproduce`・統合テスト・サンドボックスで実 LLM を呼ばずにマクロ動態を生成する
/// ための決定論的 mock．前月の失業率と，直近に確定した前年の年次インフレ率だけ
/// から，全家計に共通する労働傾向 p^w と消費傾向 p^c を返す:
///
/// - 失業が高いほど **労働意欲を上げ** (職を取りに行く)．
/// - 失業が高いほど・インフレが高いほど **消費を絞る** (予備的貯蓄 + 実質購買力の
///   低下)．値は下限 0.05 で打ち切る．
///
/// 就業は p^w の Bernoulli ドローで物価と独立し，年次インフレ率は対になる月より
/// 遅れて確定するため，Phillips 相関は 0 付近のノイズになり，符号は検証対象に
/// しない．一方，名目 GDP はほぼ就業者数 × 物価なので，Okun の強い負相関は
/// 検証できる．同一 seed なら税制レジームはマクロ集計値を変えず，Gini だけを
/// 変える．プロンプトに対して純関数的なので，cache なしでも bit 決定論的である．
/// リフレクションプロンプト (JSON 要求を含まない) には短い所感を返す．
pub fn mock_decision_client() -> EconClient {
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    let backend = ScriptedClient::new("mock-econagent", |prompt: &str| {
        if !prompt.contains("Answer with JSON only") {
            // リフレクション: 短い所感 (指標には影響しない)．
            return "The economy felt stable; I will keep working and consume moderately."
                .to_string();
        }
        let u = parse_marker(prompt, "Unemployment rate: ").unwrap_or(0.1);
        let infl = parse_marker(prompt, "Annual inflation: ").unwrap_or(0.0);
        // 労働: 0.78 + 0.4·u (高失業ほど職を取りに行く)．
        let work = (0.78 + 0.40 * u).clamp(0.0, 1.0);
        // 消費: 0.30 − 0.5·u − 0.6·infl (高失業/高インフレで消費を絞る安定化応答)．
        // ベースを抑えて需要≈供給に保ち暴走インフレを避ける．
        let consume = (0.30 - 0.50 * u - 0.60 * infl).clamp(0.05, 1.0);
        format!("{{\"work\": {work:.4}, \"consume\": {consume:.4}}}")
    });
    wrap_client(backend, PromptCache::in_memory())
}

/// 知覚プロンプトの «{marker}0.123» 形式の行から数値を読む (mock 用)．
///
/// 先頭の `-` (負のインフレ) も許容する．
fn parse_marker(prompt: &str, marker: &str) -> Option<f64> {
    let idx = prompt.find(marker)?;
    let rest = &prompt[idx + marker.len()..];
    let num: String = rest
        .chars()
        .enumerate()
        .take_while(|(i, c)| c.is_ascii_digit() || *c == '.' || (*i == 0 && *c == '-'))
        .map(|(_, c)| c)
        .collect();
    num.parse::<f64>().ok()
}

/// 与えられた [`EconClient`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．LLM クライアントはメカニズムと
/// `Rc<RefCell<…>>` で共有し，実行後にキャッシュ保存・メタデータ集計に使う．
pub fn run_with_client(cfg: &Config, client: EconClient) -> Result<SimulationResult, String> {
    run_with_client_observed(cfg, client, |_| {})
}

/// The same, calling `on_month` once for every simulated month.
///
/// The callback is where a caller counts its progress. A month is the unit
/// because it is the unit the cost is in: one month asks every household for
/// its work and consumption decision, and each of those is a model call. A
/// trial of `months` months would be a single tick, and `run` has exactly one
/// trial — it would say nothing at all between its first line and its last.
///
/// It is given the month number rather than nothing so a caller can report
/// against the clock rather than against its own tally.
pub fn run_with_client_observed(
    cfg: &Config,
    client: EconClient,
    mut on_month: impl FnMut(usize),
) -> Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    // 初期世界 (root から派生した init RNG; 決定論的 socsim コア層)．
    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    // LLM モデル/endpoint をメタデータ用に控える．
    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    // クライアント・メタデータ・月次指標バッファを共有する．
    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_metrics: SharedMetrics = Rc::new(RefCell::new(Vec::new()));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(SequentialScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(MacroPolicyMechanism))
        .add_mechanism(Box::new(LlmDecisionMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
        )))
        .add_mechanism(Box::new(MarketClearingMechanism))
        .add_mechanism(Box::new(FiscalRewardMechanism::new(Rc::clone(
            &shared_metrics,
        ))))
        .add_mechanism(Box::new(MemoryUpdateMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            cfg.memory_length,
        )))
        .build();

    let mut final_month = 0usize;
    sim.run_observed(|report| {
        final_month = report.t as usize;
        on_month(final_month);
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    // キャッシュを保存 (cache_path 指定時; in-memory はスキップ)．
    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let metrics_history = shared_metrics.borrow().clone();
    let metadata = shared_meta.borrow().clone();

    Ok(SimulationResult {
        metrics_history,
        metadata,
        llm_model,
        llm_endpoint,
        final_month,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PolicyRegime;
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    fn scripted_client() -> EconClient {
        // 全家計に労働傾向 0.8・消費傾向 0.4 を返す mock．
        let backend = ScriptedClient::new("mock-llama3.2", |_prompt: &str| {
            "{\"work\": 0.8, \"consume\": 0.4}".to_string()
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn test_config() -> Config {
        Config {
            n_agents: 8,
            months: 24,
            memory_length: 1,
            regime: PolicyRegime::Progressive,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn scripted_run_produces_monthly_metrics() {
        let cfg = test_config();
        let result = run_with_client(&cfg, scripted_client()).unwrap();
        assert_eq!(result.metrics_history.len(), cfg.months);
        assert_eq!(result.final_month, cfg.months);
    }

    #[test]
    fn core_is_deterministic_given_mock() {
        let cfg = test_config();
        let a = run_with_client(&cfg, scripted_client()).unwrap();
        let b = run_with_client(&cfg, scripted_client()).unwrap();
        let ga: Vec<f64> = a.metrics_history.iter().map(|m| m.nominal_gdp).collect();
        let gb: Vec<f64> = b.metrics_history.iter().map(|m| m.nominal_gdp).collect();
        assert_eq!(ga, gb, "同一シードは完全再現すべき");
    }
}
