"""econagent-tools show-experiment-settings — 実行結果の設定表示．

results/{timestamp}/config.json (run) または
results/{timestamp}_sweep/sweep_config.json (sweep) を読み，実行時に使われた全
パラメータを整形表示する．存在すれば run_metadata.json の LLM 情報
(モデル・endpoint・温度・seed・cache-hit 率) も併せて表示する．
`results/latest` も解決される．

Usage:
    econagent-tools show-experiment-settings
    econagent-tools show-experiment-settings --results-dir results/20260524_153000
    econagent-tools show-experiment-settings --results-dir results/latest --json

I/O・run_metadata ブロックは共有ヘルパ `socsim_tools` に委譲する (出力はバイト等価)．
run 設定テーブルは複合行 (Taylor α_π/α_u・市場 α_w/α_P) を含み汎用 `render_run_config`
の `{label}: {value}` 形に収まらないため econagent 側に残す．sweep 設定テーブルと
`--json` の `kind` フィールドも econagent 固有なので本モジュールに残す．
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import load_run_metadata, resolve_results_dir
from socsim_tools.settings import render_run_metadata


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """config.json (run) か sweep_config.json (sweep) を探す．"""
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (run) または sweep_config.json (sweep)"
    )


def render_run_config(cfg: dict, source: Path) -> str:
    """run 設定テーブルを整形する (econagent 固有; 複合行を含む)．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (run)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"エージェント数 N : {cfg.get('n_agents', '-')}")
    lines.append(f"月数 months      : {cfg.get('months', '-')}")
    lines.append(f"記憶長 L         : {cfg.get('memory_length', '-')}")
    lines.append(f"政策レジーム     : {cfg.get('regime', '-')}")
    lines.append(f"税率スケール     : {cfg.get('tax_scale', '-')}")
    lines.append(f"生産性 A         : {cfg.get('productivity', '-')}")
    lines.append(f"初期物価 P0      : {cfg.get('init_price', '-')}")
    lines.append(f"自然利子率 r_n   : {cfg.get('natural_rate', '-')}")
    lines.append(f"目標インフレ π^t : {cfg.get('target_inflation', '-')}")
    lines.append(f"自然失業率 u_n   : {cfg.get('natural_unemployment', '-')}")
    lines.append(f"Taylor α_π/α_u   : {cfg.get('alpha_pi', '-')} / {cfg.get('alpha_u', '-')}")
    lines.append(f"市場 α_w/α_P     : {cfg.get('alpha_w', '-')} / {cfg.get('alpha_p', '-')}")
    lines.append(f"シード (コア)    : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append(f"出力先           : {cfg.get('output_dir', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    """sweep 設定テーブルを整形する (econagent 固有; リスト項目を `, ` 連結する)．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"エージェント数 N : {', '.join(map(str, cfg.get('n_agents_values', [])))}")
    lines.append(f"税率スケール     : {', '.join(map(str, cfg.get('tax_scales', [])))}")
    lines.append(f"LLM モデル       : {', '.join(cfg.get('llm_models', []))}")
    lines.append(f"政策レジーム     : {cfg.get('regime', '-')}")
    lines.append(f"月数 months      : {cfg.get('months', '-')}")
    lines.append(f"記憶長 L         : {cfg.get('memory_length', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="econagent-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="実行結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open() as f:
        cfg = json.load(f)
    meta = load_run_metadata(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "run_metadata": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_run_metadata(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
