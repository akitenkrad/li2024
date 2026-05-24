//! Li et al. (2024) "EconAgent: Large Language Model-Empowered Agents for
//! Simulating Macroeconomic Activities" (ACL 2024) の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した LLM 駆動マクロ経済 ABM の公開 API を提供する．
//! 設定 (`config`)・世界状態 (`world`)・LLM クライアント層 (`llm`)・プロンプト
//! 生成と応答パース (`prompts`)・更新メカニズム (`mechanisms`)・実行ドライバ
//! (`simulation`)・集計メトリクス (`metrics`) をモジュールとして公開し，バイナリ
//! (`econagent`) と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (家計初期化・就業 Bernoulli ドロー・市場調整一様乱数・
//! スケジューリング・メトリクス) は seed から bit 単位で決定論的である．LLM レイヤ
//! (意思決定・四半期リフレクション) は socsim の bit 再現性の **外側** にあり，
//! `socsim-llm` のキャッシュ + `temperature=0` + `seed` 固定で擬似決定論化する．
//! 詳細は `crate::llm` を参照．設計書 §4.2 は当初 `reqwest` を挙げていたが，
//! 本スイートは `socsim-llm` (issue #21/#26) に標準化したため `reqwest` は使わない．

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;
