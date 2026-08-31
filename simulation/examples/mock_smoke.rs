//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (run ディレクトリ・metrics.csv / config.json) と Python 可視化を
//! 検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で決定論的に意思
//! 決定を駆動し，本番 `run` と同じ runvault の run へ書く．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use runvault::{Run, RunOptions};

use econagent_simulation::config::Config;
use econagent_simulation::llm::wrap_client;
use econagent_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use econagent_simulation::simulation::run_with_client;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());

    let cfg = Config {
        n_agents: 20,
        months: 36,
        memory_length: 2,
        seed: Some(42),
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

    // クライアントは run を開始する前に組む (`llm` ブロックのため)．
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );
    let parameters = cfg.to_run_config_json();
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&base)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(cfg.seed.expect("mock smoke は seed を固定する"))
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, client).expect("mock run failed");
    record::log_simulation(&mut rv, &result);
    let dir = rv.finish().expect("runvault: run の完了に失敗");

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {}", dir.display());
    println!(
        "final month={} GDP={:.1} unemployment={:.3} inflation={:.3} gini={:.3}",
        result.final_month,
        last.nominal_gdp,
        last.unemployment_rate,
        last.inflation_rate,
        last.gini_savings,
    );
}
