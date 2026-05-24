//! シミュレーション設定．
//!
//! Li et al. (2024) EconAgent のコアモデル (LLM 駆動マクロ経済 ABM) と感度分析
//! パラメータを保持する [`Config`] と，その JSON シリアライズ表現を定義する．
//! 税制スケジュール・金融/財政政策パラメータ・記憶長・LLM 設定をここに集約する．

use serde::Serialize;

// --------------------------------------------------------------------------- //
// 政策レジーム
// --------------------------------------------------------------------------- //

/// 財政政策レジーム (課税・再分配の方式)．
///
/// 論文標準は累進課税 + 均等再分配．感度分析のため比例課税・課税なしも列挙する．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyRegime {
    /// 累進課税 + 均等再分配 (2018 U.S. 連邦税; 論文標準)．
    Progressive,
    /// 比例課税 (フラットタックス) + 均等再分配．
    Proportional,
    /// 課税なし (再分配なし)．
    None,
}

impl PolicyRegime {
    pub fn label(&self) -> &'static str {
        match self {
            PolicyRegime::Progressive => "progressive",
            PolicyRegime::Proportional => "proportional",
            PolicyRegime::None => "none",
        }
    }
}

/// 文字列から [`PolicyRegime`] をパースする．
pub fn parse_regime(s: &str) -> Result<PolicyRegime, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "progressive" | "prog" => Ok(PolicyRegime::Progressive),
        "proportional" | "flat" | "prop" => Ok(PolicyRegime::Proportional),
        "none" | "off" => Ok(PolicyRegime::None),
        _ => Err(format!(
            "不正な政策レジーム: \"{}\" (progressive / proportional / none)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// 税制スケジュール (2018 U.S. 連邦税; 単身者)
// --------------------------------------------------------------------------- //

/// 2018 U.S. 連邦所得税 (単身者) の課税区分境界 (年額ドル)．
///
/// 区分は `[b_1, b_2, …]`．`b_1 = 0` から始まり，最後の区分は実質的に上限なし
/// (本実装では `f64::INFINITY` を末尾に補う)．
pub const US2018_BRACKETS: [f64; 7] = [
    0.0, 9_700.0, 39_475.0, 84_200.0, 160_725.0, 204_100.0, 510_300.0,
];

/// 2018 U.S. 連邦所得税 (単身者) の限界税率 (区分ごと)．
pub const US2018_RATES: [f64; 7] = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37];

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// プロバイダ優先順位は «Ollama 第一 → OpenAI フォールバック» 固定．モデル・
/// ホスト・API キーは環境変数で渡す (`OLLAMA_HOST` / `OLLAMA_MODEL` /
/// `OPENAI_API_KEY` / `OPENAI_MODEL`)．`temperature`/`seed` で擬似決定論化する．
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// 生成温度 (既定 0.0; 再現性のため．論文は 1.0 近傍)．
    pub temperature: f32,
    /// 生成シード (バックエンドへ渡す; Ollama は honour，OpenAI は best-effort)．
    pub seed: u64,
    /// プロンプト→応答キャッシュの保存先 (None なら in-memory)．
    pub cache_path: Option<String>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None,
        }
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
///
/// 既定値は論文 §5 の標準設定 (N=100, 240 か月=20 年, 累進課税 + 再分配) に近い．
#[derive(Debug, Clone)]
pub struct Config {
    /// エージェント数 N (家計数)．
    pub n_agents: usize,
    /// シミュレーション月数 t_max (240 = 20 年)．
    pub months: usize,
    /// 記憶長 L (直近 L か月の会話プールを保持; 論文既定 1)．
    pub memory_length: usize,

    // --- 政策・税制 ---
    /// 財政政策レジーム．
    pub regime: PolicyRegime,
    /// 累進税率スケール (基準税率 US2018_RATES の倍率; 1.0 が基準)．
    pub tax_scale: f64,

    // --- マクロ環境の初期値・生産 ---
    /// 生産性 A (就業者 1 人 1 月あたり 168 時間 × A 単位を生産)．
    pub productivity: f64,
    /// 初期物価 P_0．
    pub init_price: f64,
    /// パレート分布による時給初期化のスケール (最小時給)．
    pub wage_min: f64,
    /// パレート分布の形状パラメータ α (小さいほど不平等)．
    pub wage_pareto_alpha: f64,
    /// 初期貯蓄 (全家計共通の初期値)．
    pub init_savings: f64,

    // --- 金融政策 (Taylor ルール) ---
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

    // --- 市場調整 (賃金・物価) ---
    /// 賃金調整の最大変化率 α_w．
    pub alpha_w: f64,
    /// 物価調整の最大変化率 α_P．
    pub alpha_p: f64,

    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
    /// 結果出力ディレクトリ．
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_agents: 100,
            months: 240,
            memory_length: 1,
            regime: PolicyRegime::Progressive,
            tax_scale: 1.0,
            productivity: 1.0,
            init_price: 1.0,
            wage_min: 10.0,
            wage_pareto_alpha: 1.5,
            init_savings: 0.0,
            natural_rate: 0.01,
            target_inflation: 0.02,
            natural_unemployment: 0.04,
            alpha_pi: 0.5,
            alpha_u: 0.5,
            alpha_w: 0.05,
            alpha_p: 0.10,
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// 設定された政策レジーム・スケールに応じた «税区分境界, 限界税率» を返す．
    ///
    /// 末尾に `f64::INFINITY` を補った境界列 (長さ = 税率数 + 1) を返すので，
    /// `T(z)` は `brackets[k]..brackets[k+1]` を区分 `k` として走査できる．
    pub fn tax_schedule(&self) -> (Vec<f64>, Vec<f64>) {
        match self.regime {
            PolicyRegime::Progressive => {
                let mut brackets: Vec<f64> = US2018_BRACKETS.to_vec();
                brackets.push(f64::INFINITY);
                let rates: Vec<f64> = US2018_RATES
                    .iter()
                    .map(|r| (r * self.tax_scale).clamp(0.0, 1.0))
                    .collect();
                (brackets, rates)
            }
            PolicyRegime::Proportional => {
                // フラットタックス: 単一区分 [0, ∞)．基準は 22% × scale．
                let brackets = vec![0.0, f64::INFINITY];
                let rate = (0.22 * self.tax_scale).clamp(0.0, 1.0);
                (brackets, vec![rate])
            }
            PolicyRegime::None => {
                // 課税なし: 税率 0．
                let brackets = vec![0.0, f64::INFINITY];
                (brackets, vec![0.0])
            }
        }
    }
}

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_agents: usize,
    pub months: usize,
    pub memory_length: usize,
    pub regime: String,
    pub tax_scale: f64,
    pub productivity: f64,
    pub init_price: f64,
    pub wage_min: f64,
    pub wage_pareto_alpha: f64,
    pub init_savings: f64,
    pub natural_rate: f64,
    pub target_inflation: f64,
    pub natural_unemployment: f64,
    pub alpha_pi: f64,
    pub alpha_u: f64,
    pub alpha_w: f64,
    pub alpha_p: f64,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_agents: self.n_agents,
            months: self.months,
            memory_length: self.memory_length,
            regime: self.regime.label().to_string(),
            tax_scale: self.tax_scale,
            productivity: self.productivity,
            init_price: self.init_price,
            wage_min: self.wage_min,
            wage_pareto_alpha: self.wage_pareto_alpha,
            init_savings: self.init_savings,
            natural_rate: self.natural_rate,
            target_inflation: self.target_inflation,
            natural_unemployment: self.natural_unemployment,
            alpha_pi: self.alpha_pi,
            alpha_u: self.alpha_u,
            alpha_w: self.alpha_w,
            alpha_p: self.alpha_p,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
}
