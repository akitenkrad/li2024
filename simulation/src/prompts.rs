//! LLM プロンプト生成と応答パース．
//!
//! EconAgent の知覚モジュール (プロフィール + 経済変数 + 記憶 → 意思決定プロンプト)
//! と記憶モジュールの四半期リフレクションプロンプトを構築する．LLM 出力は
//! `{"work": p_work, "consume": p_consume}` の JSON で，両確率は [0,1]．
//!
//! 応答パースは「まず JSON として読む → 失敗時は本文中の数値 2 つを拾う」の二段
//! フォールバックで頑健化する (ローカルモデルは厳密 JSON を返さないことがある)．

use crate::world::{snap_probability, Household, MacroEnv};

/// 1 家計の知覚プロンプト (意思決定要求) を構築する．
///
/// プロフィール (年齢・時給・貯蓄・前月就業)・現在のマクロ経済変数 (物価・利子率・
/// 失業率)・直近記憶 + リフレクションを統合し，労働傾向 p^w と消費傾向 p^c を
/// JSON で答えるよう促す．プロンプト末尾の固定文 `Answer with JSON` で応答形式を
/// 安定させ，キャッシュキー (= プロンプト全文 + モデル名) を決定論化する．
pub fn decision_prompt(
    household: &Household,
    env: &MacroEnv,
    unemployment: f64,
    inflation: f64,
) -> String {
    let employment = if household.employed_prev {
        "employed last month"
    } else {
        "unemployed last month"
    };
    let mut s = String::new();
    s.push_str(
        "You are an economic agent (a worker-consumer) in a simulated macroeconomy. \
         Each month you decide (1) your willingness to work and (2) what fraction of your \
         wealth to consume. Reason like a rational but human household.\n\n",
    );
    s.push_str("## Your profile\n");
    s.push_str(&format!("- Age: {} years\n", household.age));
    s.push_str(&format!("- Hourly wage: {:.2}\n", household.wage));
    s.push_str(&format!("- Savings: {:.2}\n", household.savings));
    s.push_str(&format!("- Status: {employment}\n"));
    s.push_str(&format!(
        "- Last month consumption: {:.2}, tax paid: {:.2}\n",
        household.last_consumption, household.last_tax
    ));
    s.push_str("\n## Current economy\n");
    s.push_str(&format!("- Price level: {:.3}\n", env.price));
    s.push_str(&format!("- Interest rate: {:.3}\n", env.interest_rate));
    s.push_str(&format!("- Unemployment rate: {:.3}\n", unemployment));
    s.push_str(&format!("- Annual inflation: {:.3}\n", inflation));
    s.push_str(&format!(
        "- Per-capita redistribution last month: {:.2}\n",
        env.redistribution
    ));

    if !household.reflection.is_empty() {
        s.push_str("\n## Your recent reflection\n");
        s.push_str(&household.reflection);
        s.push('\n');
    }
    if !household.memory.is_empty() {
        s.push_str("\n## Recent months\n");
        for m in &household.memory {
            s.push_str("- ");
            s.push_str(m);
            s.push('\n');
        }
    }

    s.push_str(
        "\n## Decision\n\
         Decide your willingness to work this month and the fraction of wealth to consume. \
         Both are probabilities between 0 and 1 (higher work = more likely to take a job; \
         higher consume = spend a larger share of savings).\n\
         Answer with JSON only, e.g. {\"work\": 0.80, \"consume\": 0.40}\n",
    );
    s
}

/// 四半期リフレクションプロンプトを構築する (記憶モジュール)．
///
/// 直近の記憶プールを要約し，今後の意思決定に活かす短い所感を求める．
pub fn reflection_prompt(household: &Household) -> String {
    let mut s = String::new();
    s.push_str(
        "You are an economic agent reflecting on the past quarter. Summarize, in one or two \
         sentences, what you have learned about the economy and how it should guide your future \
         work and consumption decisions.\n\n",
    );
    s.push_str(&format!(
        "Your profile: age {}, hourly wage {:.2}, savings {:.2}.\n",
        household.age, household.wage, household.savings
    ));
    if !household.memory.is_empty() {
        s.push_str("Recent months:\n");
        for m in &household.memory {
            s.push_str("- ");
            s.push_str(m);
            s.push('\n');
        }
    }
    s.push_str("\nReflection:");
    s
}

/// LLM 応答から (p_work, p_consume) を抽出する．
///
/// 1. JSON `{"work": .., "consume": ..}` を試す．
/// 2. 失敗時は本文中に現れる最初の 2 つの浮動小数を拾う．
/// 3. それも失敗なら `(0.5, 0.5)` を返す (中立フォールバック)．
///
/// 抽出後は [`snap_probability`] で [0,1] / 0.02 刻みに正規化する．
pub fn parse_decision(text: &str) -> (f64, f64) {
    // 1) 厳密 JSON．
    if let Some((w, c)) = parse_json_decision(text) {
        return (snap_probability(w), snap_probability(c));
    }
    // 2) 本文中の数値 2 つ．
    let nums = extract_floats(text);
    if nums.len() >= 2 {
        return (snap_probability(nums[0]), snap_probability(nums[1]));
    }
    if nums.len() == 1 {
        return (snap_probability(nums[0]), 0.5);
    }
    // 3) 中立フォールバック．
    (0.5, 0.5)
}

/// `{"work": .., "consume": ..}` を serde_json で読む (キー名の揺れも許容)．
fn parse_json_decision(text: &str) -> Option<(f64, f64)> {
    // JSON 部分だけを切り出す ({ から最後の } まで)．
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    let slice = &text[start..=end];
    let v: serde_json::Value = serde_json::from_str(slice).ok()?;
    let obj = v.as_object()?;
    let work = lookup_number(obj, &["work", "p_work", "willingness_to_work", "labor"])?;
    let consume = lookup_number(
        obj,
        &["consume", "p_consume", "consumption", "consume_fraction"],
    )?;
    Some((work, consume))
}

/// JSON オブジェクトから候補キー名のいずれかで数値を引く．
fn lookup_number(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(f) = v.as_f64() {
                return Some(f);
            }
            // 文字列で来た場合もパースを試みる．
            if let Some(s) = v.as_str() {
                if let Ok(f) = s.trim().parse::<f64>() {
                    return Some(f);
                }
            }
        }
    }
    None
}

/// 文字列中の浮動小数 (符号・小数点) を順に抽出する．
fn extract_floats(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<f64>| {
        if !cur.is_empty() {
            if let Ok(f) = cur.parse::<f64>() {
                out.push(f);
            }
            cur.clear();
        }
    };
    for ch in text.chars() {
        if ch.is_ascii_digit() || ch == '.' || (ch == '-' && cur.is_empty()) {
            cur.push(ch);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_json() {
        let (w, c) = parse_decision("{\"work\": 0.8, \"consume\": 0.4}");
        assert!((w - 0.8).abs() < 1e-9);
        assert!((c - 0.4).abs() < 1e-9);
    }

    #[test]
    fn parses_json_with_prose() {
        let (w, c) =
            parse_decision("Sure! Here is my decision: {\"work\": 1.0, \"consume\": 0.0} done.");
        assert!((w - 1.0).abs() < 1e-9);
        assert!((c - 0.0).abs() < 1e-9);
    }

    #[test]
    fn falls_back_to_floats() {
        let (w, c) = parse_decision("work 0.6 and consume 0.3 roughly");
        assert!((w - 0.6).abs() < 1e-9);
        assert!((c - 0.3).abs() < 1e-9);
    }

    #[test]
    fn snaps_to_grid_and_clamps() {
        let (w, c) = parse_decision("{\"work\": 1.5, \"consume\": -0.3}");
        assert!((w - 1.0).abs() < 1e-9);
        assert!((c - 0.0).abs() < 1e-9);
    }

    #[test]
    fn neutral_fallback_when_no_numbers() {
        let (w, c) = parse_decision("I am not sure what to do.");
        assert!((w - 0.5).abs() < 1e-9);
        assert!((c - 0.5).abs() < 1e-9);
    }
}
