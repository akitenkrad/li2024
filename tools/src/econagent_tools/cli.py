"""econagent-tools — Li et al. (2024) EconAgent マクロ経済シミュレーション ツール統合 CLI．

Usage:
    econagent-tools visualize [...]
    econagent-tools visualize-sweep [...]
    econagent-tools show-experiment-settings [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

`reproduce` (論文 Fig.2-6 / Table の一括再現・COVID 外的介入) は Phase 3 で実装予定 (未提供)．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示の
実体 (visualize / visualize_sweep / show_experiment_settings) は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="econagent-tools",
    description="Li et al. (2024) EconAgent マクロ経済シミュレーション 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (マクロ指標時系列 + Phillips/Okun 散布) の可視化",
            "econagent_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (税率スケール × N × モデルの指標) の可視化",
            "econagent_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
            "econagent_tools.show_experiment_settings:main",
        ),
    },
)


if __name__ == "__main__":
    main()
