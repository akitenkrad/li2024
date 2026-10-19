//! Li et al. (2024) EconAgent マクロ経済 ABM の統合テスト．
//!
//! **ライブ LLM を一切必要としない**: socsim-llm の `mock::ScriptedClient` で
//! 決定論的に意思決定を駆動し，以下を検証する:
//! ・マクロ会計 (累進課税・均等再分配・貯蓄更新・名目 GDP・Gini) の妥当性
//! ・市場クリアリングの算術 (生産・需給・賃金/物価調整)
//! ・就業 Bernoulli ドローの seed 決定論性
//! ・指標プラミング (metrics.csv 行が月数ぶん生成される)

use econagent_simulation::config::{Config, PolicyRegime};
use econagent_simulation::llm::{wrap_client, EconClient};
use econagent_simulation::mechanisms::progressive_tax;
use econagent_simulation::metrics::{okun_correlation, phillips_correlation};
use econagent_simulation::simulation::{mock_decision_client, run_with_client};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

/// 全家計に固定の {work, consume} を返す mock クライアント．
fn scripted(work: f64, consume: f64) -> EconClient {
    let reply = format!("{{\"work\": {work}, \"consume\": {consume}}}");
    let backend = ScriptedClient::new("mock-model", move |prompt: &str| {
        if prompt.contains("Answer with JSON only") {
            reply.clone()
        } else {
            "Reflection: steady as she goes.".to_string()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

fn base_config() -> Config {
    Config {
        n_agents: 10,
        months: 24,
        memory_length: 1,
        regime: PolicyRegime::Progressive,
        seed: Some(7),
        ..Config::default()
    }
}

// --------------------------------------------------------------------------- //
// メトリクス配線: 月数ぶんの行が生成される
// --------------------------------------------------------------------------- //

#[test]
fn produces_one_metric_row_per_month() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(0.8, 0.4)).unwrap();
    assert_eq!(result.metrics_history.len(), cfg.months);
    assert_eq!(result.final_month, cfg.months);
    // 月インデックスは 0..months で連続している．
    for (i, m) in result.metrics_history.iter().enumerate() {
        assert_eq!(m.month, i);
    }
}

// --------------------------------------------------------------------------- //
// マクロ指標の妥当性: 失業率・Gini ∈ [0,1]，GDP は非負
// --------------------------------------------------------------------------- //

#[test]
fn macro_indicators_are_in_sane_ranges() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(0.8, 0.4)).unwrap();
    let mut saw_nonzero_gdp = false;
    for m in &result.metrics_history {
        assert!(
            (0.0..=1.0).contains(&m.unemployment_rate),
            "失業率は [0,1] (got {})",
            m.unemployment_rate
        );
        assert!(
            (0.0..=1.0).contains(&m.gini_savings),
            "Gini は [0,1] (got {})",
            m.gini_savings
        );
        assert!(m.nominal_gdp >= 0.0, "GDP は非負");
        if m.nominal_gdp > 0.0 {
            saw_nonzero_gdp = true;
        }
    }
    assert!(saw_nonzero_gdp, "高労働傾向なら GDP は非ゼロになるはず");
}

// --------------------------------------------------------------------------- //
// 雇用と GDP の関係: work=1.0 → 全員就業 → 失業率 ≈ 0，work=0.0 → 失業率 = 1
// --------------------------------------------------------------------------- //

#[test]
fn no_work_means_full_unemployment_and_zero_gdp() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(0.0, 0.0)).unwrap();
    for m in &result.metrics_history {
        assert!(
            (m.unemployment_rate - 1.0).abs() < 1e-9,
            "誰も働かない → 失業率 1"
        );
        assert!(m.nominal_gdp.abs() < 1e-9, "誰も働かない → GDP 0");
    }
}

#[test]
fn full_work_means_low_unemployment() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(1.0, 0.3)).unwrap();
    // work=1.0 → Bernoulli(1.0) は常に true → 全員就業 → 失業率 0．
    for m in &result.metrics_history {
        assert!(m.unemployment_rate.abs() < 1e-9, "全員働く → 失業率 0");
        assert!(m.nominal_gdp > 0.0, "全員働く → GDP > 0");
    }
}

// --------------------------------------------------------------------------- //
// 累進課税の算術検算 (2018 US 単身者 brackets)
// --------------------------------------------------------------------------- //

#[test]
fn progressive_tax_matches_hand_calc() {
    let cfg = base_config();
    let (brackets, rates) = cfg.tax_schedule();
    // z = 50000: 10%*9700 + 12%*(39475-9700) + 22%*(50000-39475)
    //          = 970 + 3573 + 2315.5 = 6858.5
    let t = progressive_tax(50_000.0, &brackets, &rates);
    assert!(
        (t - 6858.5).abs() < 1e-3,
        "累進税の手計算と一致すべき (got {t})"
    );
}

// --------------------------------------------------------------------------- //
// 政策レジーム: 課税なしでは税が 0
// --------------------------------------------------------------------------- //

#[test]
fn none_regime_collects_no_tax() {
    let mut cfg = base_config();
    cfg.regime = PolicyRegime::None;
    let (brackets, rates) = cfg.tax_schedule();
    assert!((progressive_tax(100_000.0, &brackets, &rates)).abs() < 1e-9);

    let result = run_with_client(&cfg, scripted(0.8, 0.4)).unwrap();
    // 課税なし → 再分配も 0．
    for m in &result.metrics_history {
        assert!(m.redistribution.abs() < 1e-9, "課税なし → 再分配 0");
    }
}

// --------------------------------------------------------------------------- //
// 決定論性: 同一シード + 同一 mock → 完全再現 (socsim コア層)
// --------------------------------------------------------------------------- //

#[test]
fn core_is_deterministic_given_fixed_mock() {
    let cfg = base_config();
    let a = run_with_client(&cfg, scripted(0.6, 0.5)).unwrap();
    let b = run_with_client(&cfg, scripted(0.6, 0.5)).unwrap();
    let ga: Vec<f64> = a.metrics_history.iter().map(|m| m.nominal_gdp).collect();
    let gb: Vec<f64> = b.metrics_history.iter().map(|m| m.nominal_gdp).collect();
    let ua: Vec<f64> = a
        .metrics_history
        .iter()
        .map(|m| m.unemployment_rate)
        .collect();
    let ub: Vec<f64> = b
        .metrics_history
        .iter()
        .map(|m| m.unemployment_rate)
        .collect();
    assert_eq!(ga, gb, "同一シードは GDP を完全再現すべき");
    assert_eq!(ua, ub, "同一シードは失業率を完全再現すべき");
}

// --------------------------------------------------------------------------- //
// 異なるシード → (一般に) 異なる就業ドロー軌跡
// --------------------------------------------------------------------------- //

#[test]
fn different_seed_changes_employment_trajectory() {
    let mut cfg_a = base_config();
    cfg_a.seed = Some(1);
    let mut cfg_b = base_config();
    cfg_b.seed = Some(999);
    // p_work=0.5 で Bernoulli ドローに依存 → seed で軌跡が変わる．
    let a = run_with_client(&cfg_a, scripted(0.5, 0.4)).unwrap();
    let b = run_with_client(&cfg_b, scripted(0.5, 0.4)).unwrap();
    let ua: Vec<f64> = a
        .metrics_history
        .iter()
        .map(|m| m.unemployment_rate)
        .collect();
    let ub: Vec<f64> = b
        .metrics_history
        .iter()
        .map(|m| m.unemployment_rate)
        .collect();
    assert_ne!(ua, ub, "異なるシードは (一般に) 異なる就業軌跡を生む");
}

// --------------------------------------------------------------------------- //
// reproduce 用 mock クライアントの性質
// --------------------------------------------------------------------------- //

/// reproduce シナリオ相当の設定 (mock decision client; ライブ LLM 不要)．
fn reproduce_config() -> Config {
    Config {
        n_agents: 100,
        months: 60,
        memory_length: 1,
        regime: PolicyRegime::Progressive,
        seed: Some(42),
        ..Config::default()
    }
}

/// mock の Okun 相関は，名目 GDP = 就業者数 × 物価という恒等式に近いため，
/// N=100・60 か月では強い負相関になる．500 seed の実測は平均 -0.959，最大
/// -0.915 であり，閾値 -0.8 は N=100 専用である．
#[test]
fn mock_okun_is_strongly_negative_across_seeds() {
    for seed in 0..8 {
        let mut cfg = reproduce_config();
        cfg.seed = Some(seed);
        let result = run_with_client(&cfg, mock_decision_client()).unwrap();
        let okun = okun_correlation(&result.metrics_history).expect("Okun 相関が計算できる");
        assert!(okun < -0.8, "seed {seed}: Okun r={okun} は -0.8 未満");
    }
}

/// mock の Phillips 相関は構造的にほぼ 0 であり，500 seed では平均 +0.003，
/// 負は 49% だった．したがって符号は判定せず，値を計算・記録できることだけを見る．
#[test]
fn mock_phillips_is_computed_but_not_judged() {
    let result = run_with_client(&reproduce_config(), mock_decision_client()).unwrap();
    let phillips =
        phillips_correlation(&result.metrics_history).expect("Phillips 相関が計算できる");
    assert!(phillips.is_finite(), "Phillips 相関は有限であるべき");
}

/// mock では税が全額再分配され，全家計が同じ割合を消費するため，税制レジームは
/// 失業率・物価・名目 GDP の系列を変えない．Gini はこの不変条件の対象外である．
#[test]
fn mock_macro_aggregates_are_regime_invariant() {
    let metrics = |regime| {
        let mut cfg = reproduce_config();
        cfg.regime = regime;
        run_with_client(&cfg, mock_decision_client())
            .unwrap()
            .metrics_history
            .into_iter()
            .map(|m| (m.unemployment_rate, m.price, m.nominal_gdp))
            .collect::<Vec<_>>()
    };
    // 失業率は完全に一致する．物価と GDP は，再分配の合計を足す順序の違いで
    // 最下位ビットがずれうるので，相対誤差 1e-9 で比べる．
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);

    let progressive = metrics(PolicyRegime::Progressive);
    for other in [PolicyRegime::Proportional, PolicyRegime::None] {
        let series = metrics(other);
        assert_eq!(progressive.len(), series.len());
        for (month, (a, b)) in progressive.iter().zip(&series).enumerate() {
            assert_eq!(a.0, b.0, "失業率が月 {month} で違う ({other:?})");
            assert!(
                close(a.1, b.1),
                "物価が月 {month} で違う ({other:?}): {} vs {}",
                a.1,
                b.1
            );
            assert!(
                close(a.2, b.2),
                "GDP が月 {month} で違う ({other:?}): {} vs {}",
                a.2,
                b.2
            );
        }
    }
}

#[test]
fn mock_reproduce_macro_indicators_are_bounded() {
    let cfg = reproduce_config();
    let result = run_with_client(&cfg, mock_decision_client()).unwrap();
    for m in &result.metrics_history {
        assert!(
            m.nominal_gdp.is_finite() && m.nominal_gdp >= 0.0,
            "GDP 有限非負"
        );
        assert!(m.inflation_rate.is_finite(), "インフレ率は有限");
        assert!((0.0..=1.0).contains(&m.unemployment_rate), "失業率 ∈ [0,1]");
        assert!((0.0..=1.0).contains(&m.gini_savings), "Gini ∈ [0,1]");
    }
}

#[test]
fn mock_decision_client_is_bit_deterministic() {
    // 同一シード + 同一 mock decision client → metrics が bit 単位で一致する．
    let cfg = reproduce_config();
    let a = run_with_client(&cfg, mock_decision_client()).unwrap();
    let b = run_with_client(&cfg, mock_decision_client()).unwrap();

    let series = |r: &econagent_simulation::simulation::SimulationResult| {
        r.metrics_history
            .iter()
            .map(|m| {
                (
                    m.nominal_gdp.to_bits(),
                    m.unemployment_rate.to_bits(),
                    m.inflation_rate.to_bits(),
                    m.gini_savings.to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        series(&a),
        series(&b),
        "同一シード + 同一 mock は metrics を bit 単位で再現すべき"
    );
}
