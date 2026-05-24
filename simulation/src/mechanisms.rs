//! socsim フレームワーク上の EconAgent マクロ経済メカニズム (5 Mechanism × 6 phase)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! 就業 Bernoulli ドロー・市場調整の一様乱数を `ctx.rng` (ChaCha20) で行い，上層
//! (非決定的 LLM レイヤ) は [`EconClient`] (キャッシュ付き Ollama→OpenAI
//! フォールバック) 越しの意思決定・四半期リフレクションを行う．
//!
//! 論文の月次ステップ (知覚 → 意思決定 → 市場更新 → 財政 → 金融 → 記憶更新) を
//! 6-phase へ割り当てる:
//!
//! | Mechanism | Phase | 役割 |
//! |-----------|-------|------|
//! | [`MacroPolicyMechanism`]  | Environment  | 年初の Taylor ルール利子率更新・物価/在庫繰越・前月再分配額の確定 |
//! | [`LlmDecisionMechanism`]  | Decision     | LLM 駆動の個体意思決定 → p^w/p^c，就業 Bernoulli ドロー (唯一の毎月 LLM 所在) |
//! | [`MarketClearingMechanism`] | Interaction | 生産/在庫・総需要・需給不均衡に応じた賃金/物価調整 |
//! | [`FiscalRewardMechanism`] | Reward       | 累進課税・均等再分配・貯蓄更新 + 月次マクロ指標の集計・記録 |
//! | [`MemoryUpdateMechanism`] | PostStep     | 記憶プール維持・四半期リフレクション (LLM)・就業繰越・終了判定 |
//!
//! LLM クライアントと呼び出しメタデータは `Rc<RefCell<…>>` で共有し，run ドライバ
//! が実行後にキャッシュ保存・メタデータ集計に使う (engine はメカニズムを所有する
//! ため，共有参照で取り出す)．集計済みの月次指標も共有バッファ経由でドライバへ渡す．

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;

use socsim_core::{Mechanism, Phase, Result, SocsimError, StepContext, WorldState};
use socsim_llm::MetadataCollector;

use crate::config::LlmSettings;
use crate::llm::{llm_config, EconClient};
use crate::metrics::{gini, inflation_rate, mean, nominal_gdp, unemployment_rate, MacroMetrics};
use crate::prompts;
use crate::world::{snap_probability, EconWorld, HOURS_PER_MONTH};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<EconClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;
/// 共有 月次指標バッファ (ドライバが run 後に CSV へ書き出す)．
pub type SharedMetrics = Rc<RefCell<Vec<MacroMetrics>>>;

// =========================================================================== //
// 1. MacroPolicyMechanism (Environment)
// =========================================================================== //

/// 年初に金融政策 (Taylor ルール) で利子率を更新し，前月再分配額を引き継ぐ
/// (`Environment` フェーズ)．
///
/// 物価・在庫はワールドに状態として残るので明示的な繰越は不要．年次境界
/// (`month % 12 == 0` かつ `month > 0`) で Taylor ルールを適用する．
pub struct MacroPolicyMechanism;

impl Mechanism<EconWorld> for MacroPolicyMechanism {
    fn name(&self) -> &str {
        "macro_policy"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, EconWorld>) -> Result<()> {
        let month = ctx.world.month();
        // 年初 (1 年目末以降の各年初) に Taylor ルールで利子率を更新する．
        if month > 0 && month.is_multiple_of(12) {
            let env = &ctx.world.env;
            let pi = inflation_rate(&env.price_history);
            // 直近年の平均失業率 (前年バッファの平均)．未蓄積なら自然失業率を使う．
            let u = if env.year_unemp_buffer.is_empty() {
                env.natural_unemployment
            } else {
                mean(&env.year_unemp_buffer)
            };
            let r = env.natural_rate
                + env.target_inflation
                + env.alpha_pi * (pi - env.target_inflation)
                + env.alpha_u * (env.natural_unemployment - u);
            ctx.world.env.interest_rate = r.max(0.0);
            // 新年度開始: 失業バッファをリセットする (物価バッファは Reward 側で管理)．
            ctx.world.env.year_unemp_buffer.clear();
        }
        Ok(())
    }
}

// =========================================================================== //
// 2. LlmDecisionMechanism (Decision)
// =========================================================================== //

/// LLM 駆動の個体意思決定 (`Decision` フェーズ; 唯一の毎月 LLM 所在)．
///
/// 各家計の知覚プロンプト (プロフィール + 経済変数 + 記憶) を構築して LLM へ
/// 問い合わせ，労働傾向 p^w・消費傾向 p^c を取得する．就業 l_i ~ Bernoulli(p^w_i)
/// を `ctx.rng` で確定する (決定論的 socsim コア層)．
pub struct LlmDecisionMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
}

impl LlmDecisionMechanism {
    pub fn new(client: SharedClient, metadata: SharedMetadata, settings: LlmSettings) -> Self {
        LlmDecisionMechanism {
            client,
            metadata,
            settings,
        }
    }
}

impl Mechanism<EconWorld> for LlmDecisionMechanism {
    fn name(&self) -> &str {
        "llm_decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, EconWorld>) -> Result<()> {
        // 知覚プロンプトに使う現在のマクロ指標 (前月までで確定した値)．
        let n = ctx.world.n();
        let n_emp_prev = ctx
            .world
            .households
            .values()
            .filter(|h| h.employed_prev)
            .count();
        let unemployment = unemployment_rate(n_emp_prev, n);
        let inflation = inflation_rate(&ctx.world.env.price_history);

        // 全家計の意思決定をスナップショット的に確定する (同期更新セマンティクス)．
        // agent_ids はソート済み → 決定論的なドロー順序．
        let ids = ctx.world.agent_ids();
        let env_snapshot = ctx.world.env.clone();
        for id in ids {
            let household = ctx
                .world
                .households
                .get(&id)
                .expect("household exists")
                .clone();
            let prompt =
                prompts::decision_prompt(&household, &env_snapshot, unemployment, inflation);
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client
                    .complete(&prompt, &llm_config(&self.settings))
                    .map_err(|e| {
                        SocsimError::Mechanism(format!("decision LLM call failed: {e}"))
                    })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let (p_work, p_consume) = prompts::parse_decision(&text);

            // 就業 Bernoulli ドロー (決定論的 socsim コア層; ctx.rng を消費)．
            let employed = ctx.rng.gen_range(0.0..1.0) < p_work;

            if let Some(h) = ctx.world.households.get_mut(&id) {
                h.p_work = snap_probability(p_work);
                h.p_consume = snap_probability(p_consume);
                h.employed = employed;
            }
        }
        Ok(())
    }
}

// =========================================================================== //
// 3. MarketClearingMechanism (Interaction)
// =========================================================================== //

/// 労働市場マッチングと財市場クリアリング (`Interaction` フェーズ)．
///
/// 生産・在庫 `G += Σ_j l_j · 168 · A`，総需要 `D = Σ_j p^c_j s_j / P`，需給不均衡
/// `φ̄ = (D − G) / max(D, G)` に応じて賃金 `w_i ← w_i(1+φ_i)`・物価 `P ← P(1+φ_P)`
/// を調整する．`φ_i, φ_P` は `sign(φ̄)·U(0, α|φ̄|)` (一様ドローは `ctx.rng`)．
pub struct MarketClearingMechanism;

impl Mechanism<EconWorld> for MarketClearingMechanism {
    fn name(&self) -> &str {
        "market_clearing"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, EconWorld>) -> Result<()> {
        let a = ctx.world.env.productivity;
        let price = ctx.world.env.price.max(1e-9);

        // 生産: 就業者がそれぞれ 168 × A 単位を生産し在庫に積む．
        let production: f64 = ctx
            .world
            .households
            .values()
            .filter(|h| h.employed)
            .map(|_| HOURS_PER_MONTH * a)
            .sum();
        ctx.world.env.inventory += production;

        // 総需要: D = Σ p^c_j s_j / P (消費傾向 × 貯蓄を物価で実質化)．
        let demand: f64 = ctx
            .world
            .households
            .values()
            .map(|h| h.p_consume * h.savings / price)
            .sum();

        let supply = ctx.world.env.inventory;
        let denom = demand.max(supply).max(1e-9);
        let phi_bar = (demand - supply) / denom; // ∈ [-1, 1]
        let sign = if phi_bar >= 0.0 { 1.0 } else { -1.0 };

        let alpha_w = ctx.world.env.alpha_w;
        let alpha_p = ctx.world.env.alpha_p;

        // 賃金調整 (家計ごとに独立な一様ドロー; ctx.rng → 決定論的)．
        let ids = ctx.world.agent_ids();
        for id in ids {
            let phi_i = sign * ctx.rng.gen_range(0.0..=(alpha_w * phi_bar.abs()));
            if let Some(h) = ctx.world.households.get_mut(&id) {
                h.wage = (h.wage * (1.0 + phi_i)).max(0.0);
            }
        }

        // 物価調整 (単一の一様ドロー)．
        let phi_p = sign * ctx.rng.gen_range(0.0..=(alpha_p * phi_bar.abs()));
        ctx.world.env.price = (ctx.world.env.price * (1.0 + phi_p)).max(1e-6);

        Ok(())
    }
}

// =========================================================================== //
// 4. FiscalRewardMechanism (Reward)
// =========================================================================== //

/// 政府の累進課税・均等再分配・貯蓄更新 + 月次マクロ指標の集計・記録
/// (`Reward` フェーズ)．
///
/// 各家計の税前所得 `z_i` に累進税 `T(z_i)` を課し，税収を均等再分配 (`z^r =
/// (1/N) Σ_j T(z_j)`)．消費 `\hat c_i = p^c_i · s_i` を控除して貯蓄を更新:
/// `s_i ← s_i − \hat c_i + (z_i − T(z_i)) + z^r`．その後，名目 GDP・失業率・
/// インフレ率・Gini を計算し共有バッファへ push する．年次境界では物価/失業を
/// 年次集約する．
pub struct FiscalRewardMechanism {
    metrics: SharedMetrics,
}

impl FiscalRewardMechanism {
    pub fn new(metrics: SharedMetrics) -> Self {
        FiscalRewardMechanism { metrics }
    }
}

/// 累進税 T(z): 区分境界 `brackets` (末尾 ∞) と限界税率 `rates` で所得 `z` の税額を
/// 計算する．`brackets.len() == rates.len() + 1` を仮定する．
pub fn progressive_tax(z: f64, brackets: &[f64], rates: &[f64]) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let mut tax = 0.0;
    for k in 0..rates.len() {
        let lo = brackets[k];
        let hi = brackets[k + 1];
        if z > lo {
            let taxable = z.min(hi) - lo;
            tax += rates[k] * taxable;
        }
    }
    tax
}

impl Mechanism<EconWorld> for FiscalRewardMechanism {
    fn name(&self) -> &str {
        "fiscal_reward"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, EconWorld>) -> Result<()> {
        let n = ctx.world.n();
        let brackets = ctx.world.env.tax_brackets.clone();
        let rates = ctx.world.env.tax_rates.clone();
        let price = ctx.world.env.price.max(1e-9);

        // --- 課税: 各家計の z_i と T(z_i) を計算 ---
        let ids = ctx.world.agent_ids();
        let mut taxes: Vec<f64> = Vec::with_capacity(n);
        let mut incomes: Vec<f64> = Vec::with_capacity(n);
        for &id in &ids {
            let h = ctx.world.households.get(&id).expect("exists");
            let z = h.pretax_income();
            let t = progressive_tax(z, &brackets, &rates);
            incomes.push(z);
            taxes.push(t);
        }
        let total_tax: f64 = taxes.iter().sum();
        let redistribution = if n > 0 { total_tax / n as f64 } else { 0.0 };

        // --- 消費・貯蓄更新 + 在庫消化 ---
        let mut consumed_units = 0.0;
        for (idx, &id) in ids.iter().enumerate() {
            if let Some(h) = ctx.world.households.get_mut(&id) {
                let z = incomes[idx];
                let t = taxes[idx];
                // 名目消費 = 消費傾向 × 貯蓄．実消費単位 = 名目消費 / P．
                let nominal_consumption = h.p_consume * h.savings.max(0.0);
                let real_units = nominal_consumption / price;
                // 税引後所得 + 再分配 − 消費 を貯蓄へ反映．
                h.savings = h.savings - nominal_consumption + (z - t) + redistribution;
                h.last_consumption = nominal_consumption;
                h.last_tax = t;
                consumed_units += real_units;
            }
        }
        // 在庫から実消費分を消化 (下限 0)．
        ctx.world.env.inventory = (ctx.world.env.inventory - consumed_units).max(0.0);
        ctx.world.env.redistribution = redistribution;

        // --- 年次集約 (物価・失業) ---
        let n_emp = ctx.world.households.values().filter(|h| h.employed).count();
        let u_month = unemployment_rate(n_emp, n);
        ctx.world.env.year_price_buffer.push(ctx.world.env.price);
        ctx.world.env.year_unemp_buffer.push(u_month);

        let month = ctx.world.month();
        // 年末 (各年の 12 番目の月; month % 12 == 11) に年次平均物価を確定する．
        if (month + 1).is_multiple_of(12) {
            let avg_price = mean(&ctx.world.env.year_price_buffer);
            ctx.world.env.price_history.push(avg_price);
            ctx.world.env.year_price_buffer.clear();
        }

        // --- 月次マクロ指標 ---
        let gdp = nominal_gdp(n_emp, HOURS_PER_MONTH, ctx.world.env.productivity, price);
        let inflation = inflation_rate(&ctx.world.env.price_history);
        let savings: Vec<f64> = ctx
            .world
            .households
            .values()
            .map(|h| h.savings.max(0.0))
            .collect();
        let p_works: Vec<f64> = ctx.world.households.values().map(|h| h.p_work).collect();
        let p_consumes: Vec<f64> = ctx.world.households.values().map(|h| h.p_consume).collect();

        let metric = MacroMetrics {
            month: month as usize,
            year: (month / 12) as usize,
            nominal_gdp: gdp,
            price: ctx.world.env.price,
            interest_rate: ctx.world.env.interest_rate,
            unemployment_rate: u_month,
            inflation_rate: inflation,
            gini_savings: gini(&savings),
            avg_work_propensity: mean(&p_works),
            avg_consume_propensity: mean(&p_consumes),
            redistribution,
        };
        self.metrics.borrow_mut().push(metric.clone());
        // ドライバ観測用に最新指標を scratch にも置く．
        ctx.scratch.insert("nominal_gdp", metric.nominal_gdp);
        ctx.scratch
            .insert("unemployment_rate", metric.unemployment_rate);

        Ok(())
    }
}

// =========================================================================== //
// 5. MemoryUpdateMechanism (PostStep)
// =========================================================================== //

/// 記憶モジュール更新 (`PostStep` フェーズ)．
///
/// 直近 L か月の会話プールを維持し，四半期末 (`(month+1) % 3 == 0`) に
/// リフレクションを生成 (LLM)．就業状態を翌月へ繰越す．終了判定は engine の
/// クロック (`t_max`) に委ねる．
pub struct MemoryUpdateMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    memory_length: usize,
}

impl MemoryUpdateMechanism {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        memory_length: usize,
    ) -> Self {
        MemoryUpdateMechanism {
            client,
            metadata,
            settings,
            memory_length: memory_length.max(1),
        }
    }
}

impl Mechanism<EconWorld> for MemoryUpdateMechanism {
    fn name(&self) -> &str {
        "memory_update"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, EconWorld>) -> Result<()> {
        let month = ctx.world.month();
        let is_quarter_end = (month + 1).is_multiple_of(3);

        let ids = ctx.world.agent_ids();
        for &id in &ids {
            // メモリへ当月の所感を追記し，直近 L 件に切り詰める．
            if let Some(h) = ctx.world.households.get_mut(&id) {
                let entry = format!(
                    "month {}: {}, worked={}, consumed={:.2}, savings={:.2}",
                    month,
                    if h.employed { "employed" } else { "unemployed" },
                    h.employed,
                    h.last_consumption,
                    h.savings,
                );
                h.memory.push(entry);
                if h.memory.len() > self.memory_length {
                    let excess = h.memory.len() - self.memory_length;
                    h.memory.drain(0..excess);
                }
                // 就業状態を翌月へ繰越す．
                h.employed_prev = h.employed;
            }
        }

        // 四半期末にリフレクションを生成する (LLM)．
        if is_quarter_end {
            for &id in &ids {
                let household = ctx.world.households.get(&id).expect("exists").clone();
                let prompt = prompts::reflection_prompt(&household);
                let text = {
                    let mut client = self.client.borrow_mut();
                    let resp = client
                        .complete(&prompt, &llm_config(&self.settings))
                        .map_err(|e| {
                            SocsimError::Mechanism(format!("reflection LLM call failed: {e}"))
                        })?;
                    self.metadata.borrow_mut().record(resp.metadata.clone());
                    resp.text
                };
                if let Some(h) = ctx.world.households.get_mut(&id) {
                    h.reflection = text.trim().to_string();
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progressive_tax_brackets() {
        // 2018 US 単身者 brackets の一部で検算．
        let brackets = vec![0.0, 9_700.0, 39_475.0, f64::INFINITY];
        let rates = vec![0.10, 0.12, 0.22];
        // z = 0 → 0
        assert!((progressive_tax(0.0, &brackets, &rates) - 0.0).abs() < 1e-9);
        // z = 9700 → 970
        assert!((progressive_tax(9_700.0, &brackets, &rates) - 970.0).abs() < 1e-6);
        // z = 20000 → 970 + 0.12*(20000-9700) = 970 + 1236 = 2206
        assert!((progressive_tax(20_000.0, &brackets, &rates) - 2206.0).abs() < 1e-6);
    }

    #[test]
    fn proportional_tax_single_bracket() {
        let brackets = vec![0.0, f64::INFINITY];
        let rates = vec![0.2];
        assert!((progressive_tax(1000.0, &brackets, &rates) - 200.0).abs() < 1e-9);
    }

    #[test]
    fn zero_tax_regime() {
        let brackets = vec![0.0, f64::INFINITY];
        let rates = vec![0.0];
        assert!((progressive_tax(5000.0, &brackets, &rates) - 0.0).abs() < 1e-9);
    }
}
