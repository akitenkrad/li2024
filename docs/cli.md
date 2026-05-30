[English](cli.md) | [日本語](cli.ja.md)

# CLI (Rust `econagent`)

Build with `cargo build --release`; the binary is `target/release/econagent`. Three subcommands: `run`, `sweep` and `reproduce`.

## LLM environment variables

The LLM provider order is **Ollama first → OpenAI fallback**, configured by environment (never hard-coded):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint (`/api/chat`) |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model |
| `OPENAI_API_KEY` | (unset) | enables the OpenAI fallback |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model (paper used `gpt-3.5-turbo-0613`) |

If Ollama is unreachable and no OpenAI key is set, the run fails with a config error. With a warm prompt cache the model is not contacted at all.

## `run` — single simulation

```bash
cargo run --release -- run \
    --n-agents 100 --months 240 --seed 42 \
    --memory-length 1 --regime progressive --tax-scale 1.0 \
    --llm-temperature 0.0
```

| Flag | Default | Meaning |
|---|---|---|
| `--n-agents` | 100 | number of households N |
| `--months` | 240 | simulation length in months (240 = 20 years) |
| `--memory-length` | 1 | months of conversation memory L |
| `--regime` | progressive | fiscal regime: `progressive` / `proportional` / `none` |
| `--tax-scale` | 1.0 | multiplier on the base tax rates |
| `--seed` | (random) | root seed for the deterministic socsim core |
| `--llm-temperature` | 0.0 | generation temperature (0 for reproducibility) |
| `--llm-seed` | 0 | backend generation seed |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache file |
| `--output-dir` | `results` | output base directory |

Outputs to `results/{timestamp}/`: `config.json`, `metrics.csv`, `run_metadata.json`, and a `results/latest` symlink.

A quick offline-friendly smoke for the real-LLM path: `--n-agents 4 --months 12 --seed 42`.

## `sweep` — sensitivity analysis

Sweeps over agent count × tax-scale × LLM model, with custom aggregation into `sweep_summary.csv`:

```bash
cargo run --release -- sweep \
    --n-agents-values 50,100,300 \
    --tax-scale-min 0.5 --tax-scale-max 1.5 --tax-scale-step 0.5 \
    --llm-models llama3.2:latest \
    --runs 3 --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--n-agents-values` | `50,100,300` | comma-separated N values |
| `--tax-scale-min/max/step` | `0.5/1.5/0.5` | tax-scale grid |
| `--llm-models` | `llama3.2:latest` | comma-separated models (also overrides `OLLAMA_MODEL` per cell) |
| `--regime` | progressive | single fiscal regime for the sweep |
| `--months` | 240 | months per run |
| `--runs` | 3 | independent trials per condition (distinct derived seeds) |
| `--seed` | 42 | base seed; each cell derives an independent stream |

Each `sweep_summary.csv` row records per-condition aggregates: 3-year-onward mean inflation/unemployment, final GDP, final Gini, mean work/consume propensity and the cache-hit rate. Outputs to `results/{timestamp}_sweep/` with `sweep_config.json`.

## `reproduce` — paper headline (Phillips curve / Okun's law / macro dynamics)

Runs the headline macro reproduction across the three fiscal regimes (the standard `progressive` regime is the headline scenario) and checks the observed correlations against the expected negative-sign anchors:

```bash
# offline (no live LLM) — scripted mock decision client
cargo run --release -- reproduce --mock --quick --seed 42

# live LLM (Ollama first), full 240 months
OLLAMA_MODEL=llama3.2:latest cargo run --release -- reproduce --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--n-agents` | 100 | number of households N |
| `--months` | 240 | months per scenario (`--quick` shrinks to 60) |
| `--memory-length` | 1 | months of conversation memory L |
| `--seed` | 42 | base seed; each regime derives an independent stream |
| `--mock` | off | use the scripted mock decision client (offline / CI; no live LLM) |
| `--quick` | off | short reproduction (months=60) |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache (live path only) |
| `--output-dir` | `results` | output base directory |

Computes the **Phillips curve** correlation (unemployment vs inflation, expected negative) and **Okun's law** correlation (Δunemployment vs GDP growth, expected negative) from each scenario's `metrics.csv`, and writes `reproduce_summary.json` with the observed correlations and PASS/off anchors (the two negative-sign headline anchors plus bounded inflation/unemployment sanity ranges). Each scenario's `metrics.csv` / `run_metadata.json` / `config.json` are saved under `results/{timestamp}_reproduce/{regime}/`. The `--mock` path is bit-deterministic and contacts no model; the level of inflation under the offline mock is a proxy and does not match the paper's level (a local proxy ≠ the paper's `gpt-3.5-turbo`) — the headline result is the **sign** of the two correlations.

Generate the figures (macro time series + Phillips/Okun scatter with scipy `r`/`p`) with `uv run econagent-tools reproduce`.

---
*This file was generated by Claude Code.*
