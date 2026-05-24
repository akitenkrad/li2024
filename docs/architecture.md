[English](architecture.md) | [日本語](architecture.ja.md)

# Architecture

## Repository layout

```
replications/li2024/
├── Cargo.toml                  # Rust workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust crate `econagent-simulation` (bin `econagent`)
│   ├── Cargo.toml              # socsim-core + socsim-engine + socsim-llm (features=["live"])
│   ├── examples/mock_smoke.rs  # offline (no live LLM) pipeline smoke
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep
│   │   ├── lib.rs
│   │   ├── config.rs           # Config, PolicyRegime, tax schedule, LLM settings
│   │   ├── world.rs            # EconWorld (WorldState), Household, MacroEnv
│   │   ├── mechanisms.rs       # the five mechanisms over the six phases
│   │   ├── llm.rs              # socsim-llm builder (Ollama→OpenAI + cache)
│   │   ├── prompts.rs          # perception/reflection prompts + response parsing
│   │   ├── metrics.rs          # GDP / inflation / unemployment / Gini / propensities
│   │   └── simulation.rs       # init_world + run drivers + output writers
│   └── tests/integration_test.rs   # mock-driven (ScriptedClient); no live LLM
├── tools/                      # Python package `econagent-tools` (module `econagent_tools`)
│   └── src/econagent_tools/
│       ├── cli.py
│       ├── visualize.py        # macro time series + Phillips/Okun scatter (scipy)
│       ├── visualize_sweep.py  # tax-scale / N / model comparison
│       └── show_experiment_settings.py
└── docs/                       # bilingual (.md + .ja.md)
```

## Two-layer determinism

socsim's core is deterministic and LLM-free; an LLM is inherently not. The design confines the LLM to two mechanisms and pseudo-determinises it:

| Layer | Components | Reproducibility |
|---|---|---|
| Deterministic socsim core | household init (Pareto wages, age bands), employment Bernoulli draws, market-adjustment uniforms, scheduling, fiscal/monetary accounting, metrics | bit-for-bit given the seed (ChaCha20 `SimRng`) |
| Non-deterministic LLM layer | monthly work/consume decision, quarterly reflection | `socsim-llm` `CachingClient` (`hash(prompt+model)` → response) + `temperature=0` + fixed seed |

RNG streams are derived from one root seed: `derive_seed(root, &[0])` for world init, `derive_seed(root, &[1])` for the engine (employment draws, market uniforms).

## The LLM client layer (`llm.rs`)

The production client is

```
CachingClient< Box<dyn LlmClient> >
  backend = FallbackClient< OllamaClient, OpenAiClient >   // Ollama first → OpenAI fallback
  cache   = PromptCache (JSON file or in-memory)
```

The backend is type-erased to `Box<dyn LlmClient>` directly — `socsim-llm` provides `impl LlmClient for Box<T>` (issue #26), so no local newtype is needed. Tests inject a `mock::ScriptedClient` through the same `EconClient = CachingClient<Box<dyn LlmClient>>` alias. The crate depends on `socsim-llm` with `features = ["live"]` (Ollama + OpenAI backends, the `FallbackClient`). There is **no `reqwest`** dependency: socsim-llm owns the HTTP transport.

## WorldState and mechanisms

`EconWorld` holds `households: BTreeMap<AgentId, Household>` (sorted keys → deterministic `agent_ids()`) and a single `MacroEnv` block (price, interest rate, inventory, productivity, tax schedule, price/unemployment history, policy params). One tick = one month; the clock runs `1..=months`, and the model treats month indices as 0-based (`month() = t − 1`).

The five mechanisms map onto the six-phase loop:

| Mechanism | Phase | Role |
|---|---|---|
| `MacroPolicyMechanism` | Environment | year-boundary Taylor-rule interest-rate update; carry redistribution |
| `LlmDecisionMechanism` | Decision | perception prompt → LLM → `{work, consume}` JSON; employment `l_i ~ Bernoulli(p^w_i)` via `ctx.rng` (the only per-step LLM site) |
| `MarketClearingMechanism` | Interaction | production `G += Σ l_j·168·A`, demand `D = Σ p^c_j s_j / P`, imbalance `φ̄ = (D−G)/max(D,G)`, wage/price adjustment |
| `FiscalRewardMechanism` | Reward | progressive tax `T(z_i)`, equal redistribution, savings update; compute & record monthly macro metrics |
| `MemoryUpdateMechanism` | PostStep | maintain last-L-month memory pool, quarterly reflection (LLM), carry employment |

The LLM client, the metadata collector and the monthly-metrics buffer are shared with the mechanisms via `Rc<RefCell<…>>`, since the engine owns the boxed mechanisms; the run driver reads them back after the run for cache persistence, metadata and CSV output.

## Metrics

`metrics.csv` is one wide row per month: `month, year, nominal_gdp, price, interest_rate, unemployment_rate, inflation_rate, gini_savings, avg_work_propensity, avg_consume_propensity, redistribution`. The Phillips curve (inflation vs unemployment) and Okun's law (GDP growth vs unemployment change) Pearson correlations are computed in Python (scipy) from this file.

## References

- Li, N., Gao, C., Li, M., Li, Y., & Liao, Q. (2024). EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities. *Proceedings of ACL 2024*, 15523–15536. arXiv:2310.10436.
- socsim: https://github.com/akitenkrad/rs-social-simulation-tools (the `socsim-llm` crate, issues #21/#26).

---
*This file was generated by Claude Code.*
