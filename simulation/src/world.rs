//! socsim フレームワーク上の EconAgent マクロ経済シミュレーションの世界状態．
//!
//! エージェント = 移動する空間主体ではなく，市場を通じて相互作用する固定の家計
//! (worker-consumer) である．したがって `socsim-grid` (`GridIndex` / `CellGrid`)
//! は採用せず，家計属性を `BTreeMap<AgentId, Household>` に，マクロ環境を単一の
//! [`MacroEnv`] ブロックに保持する．`agent_ids()` は `BTreeMap` キー (昇順) を
//! そのまま返し決定論を担保する (socsim コア層)．
//!
//! `#[derive(Clone)]` でスナップショット (save/resume) と感度分析の比較実験に
//! 対応する．

use std::collections::BTreeMap;

use socsim_core::{AgentId, SimClock, WorldState};

/// 1 家計 (worker-consumer) の状態．
#[derive(Clone, Debug)]
pub struct Household {
    /// 時給 w_i (パレート分布で初期化)．
    pub wage: f64,
    /// 貯蓄 s_i．
    pub savings: f64,
    /// 前月の就業状態．
    pub employed_prev: bool,
    /// 当月の就業状態 l_i ∈ {0,1} (Bernoulli(p_work) で確定)．
    pub employed: bool,
    /// 年齢 (2018 U.S. 分布で初期化)．
    pub age: u32,
    /// 前月の実消費 \hat c_i．
    pub last_consumption: f64,
    /// 前月の支払税 T(z_i)．
    pub last_tax: f64,
    /// 直近の労働傾向 p^w_i ∈ [0,1] (LLM 出力)．
    pub p_work: f64,
    /// 直近の消費傾向 p^c_i ∈ [0,1] (LLM 出力)．
    pub p_consume: f64,
    /// 直近 L か月の会話プール (知覚プロンプトへ注入)．
    pub memory: Vec<String>,
    /// 四半期リフレクション (LLM が要約; 知覚プロンプトへ注入)．
    pub reflection: String,
}

impl Household {
    /// 初期賃金・年齢・初期貯蓄から家計状態を作る．
    pub fn new(wage: f64, age: u32, savings: f64) -> Self {
        Household {
            wage,
            savings,
            employed_prev: false,
            employed: false,
            age,
            last_consumption: 0.0,
            last_tax: 0.0,
            p_work: 0.0,
            p_consume: 0.0,
            memory: Vec::new(),
            reflection: String::new(),
        }
    }

    /// 当月の税前所得 z_i = (就業なら w_i × 168，非就業なら 0)．
    pub fn pretax_income(&self) -> f64 {
        if self.employed {
            self.wage * HOURS_PER_MONTH
        } else {
            0.0
        }
    }
}

/// 就業者 1 人の 1 月あたり労働時間 (論文: 168 時間)．
pub const HOURS_PER_MONTH: f64 = 168.0;

/// マクロ環境ブロック (市場・政策・集計)．
#[derive(Clone, Debug)]
pub struct MacroEnv {
    /// 物価 P．
    pub price: f64,
    /// 利子率 r (Taylor ルール; 年次更新)．
    pub interest_rate: f64,
    /// 在庫 G (財市場の供給ストック)．
    pub inventory: f64,
    /// 生産性 A．
    pub productivity: f64,
    /// 前月の一人当たり再分配額 z^r．
    pub redistribution: f64,
    /// 累進税区分境界 b_k (末尾に ∞ を含む; 長さ = 税率数 + 1)．
    pub tax_brackets: Vec<f64>,
    /// 区分ごとの限界税率 τ_k．
    pub tax_rates: Vec<f64>,
    /// 年次平均物価 \bar P_n (インフレ率算出用; 各年 1 値)．
    pub price_history: Vec<f64>,
    /// 当年 (進行中) の月次物価バッファ (年末に平均して price_history へ)．
    pub year_price_buffer: Vec<f64>,
    /// 当年 (進行中) の月次失業率バッファ (年末に平均して年次失業率へ)．
    pub year_unemp_buffer: Vec<f64>,

    // --- 政策パラメータ ---
    /// 自然利子率 r_n．
    pub natural_rate: f64,
    /// 目標インフレ率 π^t．
    pub target_inflation: f64,
    /// 自然失業率 u_n．
    pub natural_unemployment: f64,
    /// Taylor ルールのインフレ係数 α_π．
    pub alpha_pi: f64,
    /// Taylor ルールの失業係数 α_u．
    pub alpha_u: f64,
    /// 賃金調整の最大変化率 α_w．
    pub alpha_w: f64,
    /// 物価調整の最大変化率 α_P．
    pub alpha_p: f64,
}

/// EconAgent マクロ経済シミュレーションの世界状態．
#[derive(Clone)]
pub struct EconWorld {
    /// シミュレーションクロック (1 tick = 1 か月)．
    pub clock: SimClock,
    /// 各家計の状態 (ソート済みキー)．
    pub households: BTreeMap<AgentId, Household>,
    /// マクロ環境ブロック．
    pub env: MacroEnv,
}

impl EconWorld {
    /// 家計数 N．
    pub fn n(&self) -> usize {
        self.households.len()
    }

    /// 現在月 (0 始まり)．
    ///
    /// socsim エンジンはステップ先頭で `tick()` するため，クロックは 1..=t_max を
    /// 走る．本モデルは月を 0 始まり (0..t_max) で扱うので `t() - 1` を返す．
    pub fn month(&self) -> u64 {
        self.clock.t().saturating_sub(1)
    }

    /// 当年内の月インデックス (0..=11)．年次境界判定に使う．
    pub fn month_in_year(&self) -> u64 {
        self.month() % 12
    }
}

impl WorldState for EconWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap のキーはソート済み．契約 (sorted) を明示する → 決定論．
        self.households.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

/// 確率を [0,1] にクランプし 0.02 刻みグリッドに丸める (論文: 0〜1 を 0.02 刻み)．
pub fn snap_probability(p: f64) -> f64 {
    let clamped = p.clamp(0.0, 1.0);
    (clamped / 0.02).round() * 0.02
}
