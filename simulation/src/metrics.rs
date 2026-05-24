//! 評価指標 (論文 §4.3 §6)．
//!
//! 月次マクロ指標 — 名目 GDP・インフレ率・失業率・Gini 係数・平均労働傾向・
//! 平均消費傾向 — を計算する．Phillips 曲線 (インフレ vs 失業) と Okun の法則
//! (GDP 成長 vs 失業変化) の Pearson 相関は Python (scipy) 側で `metrics.csv`
//! から計算するため，ここでは時系列の構成要素のみを提供する．

use serde::Serialize;

/// 名目 GDP: 当月に生産された名目付加価値 = Σ_j (就業者の名目生産)．
///
/// 就業者 1 人あたり `168 × A` 単位を物価 `P` で評価する．
pub fn nominal_gdp(n_employed: usize, hours_per_month: f64, productivity: f64, price: f64) -> f64 {
    n_employed as f64 * hours_per_month * productivity * price
}

/// 失業率 u = (非就業者数) / N ∈ [0,1] (当月)．
pub fn unemployment_rate(n_employed: usize, n_total: usize) -> f64 {
    if n_total == 0 {
        return 0.0;
    }
    1.0 - (n_employed as f64 / n_total as f64)
}

/// インフレ率 π = (\bar P_n − \bar P_{n-1}) / \bar P_{n-1}．
///
/// 年次平均物価が 2 値以上あるとき直近 2 年から算出する．1 年目は定義不可なので
/// `0.0` を返す．
pub fn inflation_rate(price_history: &[f64]) -> f64 {
    let n = price_history.len();
    if n < 2 {
        return 0.0;
    }
    let prev = price_history[n - 2];
    let curr = price_history[n - 1];
    if prev.abs() < 1e-12 {
        return 0.0;
    }
    (curr - prev) / prev
}

/// Gini 係数 G = Σ_i Σ_j |x_i − x_j| / (2 N Σ_i x_i) ∈ [0,1]．
///
/// 負値や総和 0 のときは 0 を返す (定義不可)．
pub fn gini(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let total: f64 = values.iter().sum();
    if total.abs() < 1e-12 {
        return 0.0;
    }
    let mut sum_abs_diff = 0.0;
    for &xi in values {
        for &xj in values {
            sum_abs_diff += (xi - xj).abs();
        }
    }
    let g = sum_abs_diff / (2.0 * n as f64 * total);
    g.clamp(0.0, 1.0)
}

/// 全家計の平均値 (空なら 0)．労働傾向・消費傾向の平均に使う．
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// 1 月分のマクロ指標 (metrics.csv の 1 行; long-format ではなく wide 1 行)．
///
/// Python 側で long-format へ溶かす想定はせず，各列を持つ wide 形式で出力する
/// (指標数が少なく時系列解析が素直なため)．`metrics.csv` は «month, year, ...»
/// の月次行を縦に並べた表になる．
#[derive(Debug, Clone, Serialize)]
pub struct MacroMetrics {
    /// 月 t (0 始まり)．
    pub month: usize,
    /// 年 n (= month / 12)．
    pub year: usize,
    /// 名目 GDP (当月)．
    pub nominal_gdp: f64,
    /// 物価 P (当月末)．
    pub price: f64,
    /// 利子率 r (当月適用)．
    pub interest_rate: f64,
    /// 失業率 u (当月) ∈ [0,1]．
    pub unemployment_rate: f64,
    /// インフレ率 π (直近年次; 年初に更新され当年内は一定)．
    pub inflation_rate: f64,
    /// 貯蓄分布の Gini 係数 ∈ [0,1]．
    pub gini_savings: f64,
    /// 平均労働傾向 \bar p^w ∈ [0,1]．
    pub avg_work_propensity: f64,
    /// 平均消費傾向 \bar p^c ∈ [0,1]．
    pub avg_consume_propensity: f64,
    /// 当月の一人当たり再分配額 z^r．
    pub redistribution: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unemployment_in_range() {
        assert!((unemployment_rate(0, 10) - 1.0).abs() < 1e-12);
        assert!((unemployment_rate(10, 10) - 0.0).abs() < 1e-12);
        assert!((unemployment_rate(5, 10) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gini_of_equal_is_zero() {
        let eq = vec![100.0, 100.0, 100.0, 100.0];
        assert!(gini(&eq) < 1e-9);
    }

    #[test]
    fn gini_of_extreme_inequality_near_one() {
        // 1 人だけが全資産を持つ → N が大きいほど Gini → 1．
        let mut v = vec![0.0; 99];
        v.push(100.0);
        let g = gini(&v);
        assert!(g > 0.95, "極端な不平等で Gini はほぼ 1 (got {g})");
    }

    #[test]
    fn inflation_from_history() {
        let h = vec![1.0, 1.05];
        assert!((inflation_rate(&h) - 0.05).abs() < 1e-9);
        assert_eq!(inflation_rate(&[1.0]), 0.0);
    }

    #[test]
    fn nominal_gdp_scales_with_employment() {
        let g1 = nominal_gdp(10, 168.0, 1.0, 1.0);
        let g2 = nominal_gdp(20, 168.0, 1.0, 1.0);
        assert!((g2 - 2.0 * g1).abs() < 1e-9);
    }
}
