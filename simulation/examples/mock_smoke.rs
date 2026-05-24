//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (metrics.csv / run_metadata.json / config.json) と Python
//! 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で
//! 決定論的に意思決定を駆動し，本番 `run` と同じ writer で結果を書き出す．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;
use std::fs;

use chrono::Local;

use econagent_simulation::config::Config;
use econagent_simulation::llm::wrap_client;
use econagent_simulation::simulation::{
    ensure_output_dir, run_with_client, save_metrics, save_run_metadata,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        n_agents: 20,
        months: 36,
        memory_length: 2,
        seed: Some(42),
        output_dir: output_dir.clone(),
        ..Config::default()
    };

    // 意思決定プロンプトには JSON で {work, consume} を返す．プロンプトに含まれる
    // 失業率に応じて消費傾向をゆるく変える擬似挙動で，指標に動きを出す．
    // リフレクションプロンプト (JSON を含まない) には短い所感を返す．
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        if prompt.contains("Answer with JSON only") {
            // 高失業 (Unemployment rate: 0.5+) では消費傾向を下げる擬似挙動．
            if prompt.contains("Unemployment rate: 0.5")
                || prompt.contains("Unemployment rate: 0.6")
                || prompt.contains("Unemployment rate: 0.7")
            {
                "{\"work\": 0.85, \"consume\": 0.20}".to_string()
            } else {
                "{\"work\": 0.75, \"consume\": 0.45}".to_string()
            }
        } else {
            "The economy felt stable; I will keep working and consume moderately.".to_string()
        }
    });
    let client = wrap_client(backend, PromptCache::in_memory());

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {output_dir}");
    println!(
        "final month={} GDP={:.1} unemployment={:.3} inflation={:.3} gini={:.3}",
        result.final_month,
        last.nominal_gdp,
        last.unemployment_rate,
        last.inflation_rate,
        last.gini_savings,
    );
}
