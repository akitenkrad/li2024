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

/// Pearson 相関係数 r = Σ(x−x̄)(y−ȳ) / √(Σ(x−x̄)² Σ(y−ȳ)²)．
///
/// `reproduce` の Phillips 曲線 (失業率 vs インフレ率) と Okun の法則 (失業率変化
/// vs GDP 成長率) の符号検証に使う．長さが揃わない・標本 < 2・どちらかの分散が 0
/// のときは `None` を返す (定義不可)．Python (scipy.stats.pearsonr) 側の値と一致
/// させるための独立実装で，LLM 非依存・決定論的．
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-12 || vy < 1e-12 {
        return None;
    }
    Some(cov / (vx * vy).sqrt())
}

/// Phillips 曲線の相関: 失業率 (x) vs インフレ率 (y)．
///
/// 3 年目以降 (month ≥ 36) の月次系列を使い，過渡期を除く (論文の定常域に対応)．
/// 該当データが 2 点未満なら全期間にフォールバックする．負の相関 (失業 ↑ で
/// インフレ ↓) が headline．
pub fn phillips_correlation(metrics: &[MacroMetrics]) -> Option<f64> {
    let mut sub: Vec<&MacroMetrics> = metrics.iter().filter(|m| m.month >= 36).collect();
    if sub.len() < 2 {
        sub = metrics.iter().collect();
    }
    let u: Vec<f64> = sub.iter().map(|m| m.unemployment_rate).collect();
    let pi: Vec<f64> = sub.iter().map(|m| m.inflation_rate).collect();
    pearson(&u, &pi)
}

/// Okun の法則の相関: 失業率変化 Δu (x) vs 名目 GDP 成長率 (y)．
///
/// 月次差分から Δu と GDP 成長率を作り Pearson 相関を取る．負の相関 (失業 ↑ で
/// 成長 ↓) が headline．GDP=0 の月は成長率が定義不可なので除外する．
pub fn okun_correlation(metrics: &[MacroMetrics]) -> Option<f64> {
    if metrics.len() < 3 {
        return None;
    }
    let mut du: Vec<f64> = Vec::new();
    let mut growth: Vec<f64> = Vec::new();
    for w in metrics.windows(2) {
        let prev = &w[0];
        let curr = &w[1];
        if prev.nominal_gdp.abs() < 1e-12 {
            continue;
        }
        du.push(curr.unemployment_rate - prev.unemployment_rate);
        growth.push((curr.nominal_gdp - prev.nominal_gdp) / prev.nominal_gdp);
    }
    pearson(&du, &growth)
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

    #[test]
    fn pearson_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [4.0, 3.0, 2.0, 1.0];
        let r = pearson(&x, &y).unwrap();
        assert!((r + 1.0).abs() < 1e-9, "完全な負の相関は r=-1 (got {r})");
    }

    #[test]
    fn pearson_degenerate_is_none() {
        // 分散 0・標本不足は None．
        assert!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
        assert!(pearson(&[1.0], &[1.0]).is_none());
        assert!(pearson(&[1.0, 2.0], &[1.0]).is_none());
    }

    /// month/year/unemployment/inflation/gdp だけを持つ簡易行を作る．
    fn row(month: usize, u: f64, infl: f64, gdp: f64) -> MacroMetrics {
        MacroMetrics {
            month,
            year: month / 12,
            nominal_gdp: gdp,
            price: 1.0,
            interest_rate: 0.0,
            unemployment_rate: u,
            inflation_rate: infl,
            gini_savings: 0.0,
            avg_work_propensity: 0.0,
            avg_consume_propensity: 0.0,
            redistribution: 0.0,
        }
    }

    #[test]
    fn phillips_negative_on_synthetic_data() {
        // 失業 ↑ で インフレ ↓ の合成系列 (month >= 36 を確保)．
        let mut m = Vec::new();
        for k in 0..48 {
            let u = 0.02 + 0.001 * k as f64;
            let infl = 0.05 - 0.0008 * k as f64;
            m.push(row(k, u, infl, 1000.0 - 2.0 * k as f64));
        }
        let r = phillips_correlation(&m).unwrap();
        assert!(r < 0.0, "Phillips は負の相関 (got {r})");
    }

    #[test]
    fn okun_negative_on_synthetic_data() {
        // 失業上昇月は GDP 下降，失業下降月は GDP 上昇 → Δu と成長率は負相関．
        let mut m = Vec::new();
        let mut gdp = 1000.0;
        let mut u = 0.05;
        for k in 0..40 {
            let up = k % 2 == 0;
            u += if up { 0.01 } else { -0.01 };
            gdp *= if up { 0.97 } else { 1.03 };
            m.push(row(k, u, 0.0, gdp));
        }
        let r = okun_correlation(&m).unwrap();
        assert!(r < 0.0, "Okun は負の相関 (got {r})");
    }
}
