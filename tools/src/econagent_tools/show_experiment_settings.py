"""econagent-tools show-experiment-settings — run ディレクトリの設定表示．

runvault の run ディレクトリの `config.json` (封筒．条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．`run` か `sweep` か `sweep-point` か
`reproduce` か `reproduce-scenario` かは `run.json` の `subcommand` で判別する
(`sweep_config.json` はもう書かれない)．LLM 情報 (モデル・provider・温度) は
`run.json` の `llm` ブロック，呼び出し数と cache-hit 率は `metrics.csv` の run
スコープ指標から採る (移行前の `run_metadata.json` はこの 2 つに分かれた)．

legacy の `results/<timestamp>/config.json` (flat) と
`results/<timestamp>_sweep/sweep_config.json` も `--results-dir` に直接渡せば読める．

run ディレクトリのパスは次で取れる:
    runvault path --experiment econagent --latest --subcommand run --standalone
    runvault path --experiment econagent --latest --subcommand sweep
    runvault path --experiment econagent --latest --subcommand reproduce

Usage:
    uv run econagent-tools show-experiment-settings
    uv run econagent-tools show-experiment-settings --results-dir "$(runvault path --experiment econagent --latest --subcommand sweep)"
    uv run econagent-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from runvault.read import config_parameters, load_run_meta, runvault_path

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "econagent"

# config キー → 表示ラベル (右コロン位置を揃えるためパディング済み)．
# `run` の条件と，`sweep` / `reproduce` の親が持つ格子の両方を並べる — 親と子で
# キーがほとんど重ならないので 1 つの表で足りる．
FIELD_LABELS = {
    # run (掃引/再現の子も同じ形)
    "n_agents": "エージェント数 N ",
    "months": "月数 months      ",
    "memory_length": "記憶長 L         ",
    "regime": "政策レジーム     ",
    "tax_scale": "税率スケール     ",
    "productivity": "生産性 A         ",
    "init_price": "初期物価 P0      ",
    "wage_min": "最小時給         ",
    "wage_pareto_alpha": "パレート α       ",
    "init_savings": "初期貯蓄         ",
    "natural_rate": "自然利子率 r_n   ",
    "target_inflation": "目標インフレ π^t ",
    "natural_unemployment": "自然失業率 u_n   ",
    "alpha_pi": "Taylor α_π       ",
    "alpha_u": "Taylor α_u       ",
    "alpha_w": "市場 α_w         ",
    "alpha_p": "市場 α_P         ",
    # sweep 親
    "n_agents_values": "N (格子)         ",
    "tax_scales": "税率スケール格子 ",
    "llm_models": "LLM モデル (格子)",
    "runs": "試行数 runs      ",
    # reproduce 親
    "mock": "mock 駆動        ",
    # 共通
    "seed": "シード (コア)    ",
    "llm_temperature": "LLM 温度         ",
    "llm_seed": "LLM seed         ",
    # legacy の flat config.json だけが持つ
    "output_dir": "出力先 (legacy)  ",
}

# run スコープ指標 → 表示ラベル．
SCOPE_LABELS = {
    "final_month": "完了ステップ数   ",
    "llm_calls": "呼び出し総数     ",
    "llm_cache_hits": "cache-hit        ",
}


def _fmt(value: object) -> str:
    """リストは `, ` 連結，それ以外はそのまま．"""
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def resolve_results_dir(path_like: str) -> Path:
    """シンボリックリンク (legacy の `results/latest`) を実体に解決する．"""
    p = Path(path_like)
    if p.is_symlink():
        return Path(os.path.realpath(p))
    return p


def load_config(results_dir: Path) -> tuple[dict, Path, str]:
    """実験条件と，それがどのサブコマンドのものかを返す．

    runvault の config.json は封筒で，条件は `parameters` の下にある．どのサブコマンド
    かは run.json が答える．legacy は flat な config.json (`command` を持つ) か
    `sweep_config.json`．
    """
    params = config_parameters(results_dir, required=False)
    if params is not None:
        meta = load_run_meta(results_dir, required=False)
        if meta is not None:
            kind = str(meta.get("subcommand", "run"))
        else:
            kind = "sweep" if params.get("command") == "sweep" else "run"
        return params, results_dir / "config.json", kind

    sweep_cfg = results_dir / "sweep_config.json"
    if sweep_cfg.exists():
        with sweep_cfg.open() as f:
            return json.load(f), sweep_cfg, "sweep"

    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (runvault の封筒 / legacy の flat) "
        f"または sweep_config.json (legacy の sweep)"
    )


def scope_metrics(results_dir: Path, scope: str) -> dict[str, float]:
    """`metrics.csv` の «step を持たない» 行を scope で絞って読む．

    `runvault.read.run_scope_metrics` は scope 列を見ないので，再現の親が持つ
    «シナリオをまたいだ集約» (scope=sweep) と子の «run 全体の値» (scope=run) が
    混ざる．ここでは分けて表示するので scope で絞る．
    """
    path = results_dir / "metrics.csv"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "scope" not in reader.fieldnames:
            return {}  # legacy の wide な metrics.csv
        for row in reader:
            if row["scope"] == scope and row["step"] == "":
                out[row["name"]] = float(row["value"])
    return out


def render_config(cfg: dict, source: Path, kind: str) -> str:
    """設定テーブルを整形する．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"実行設定 ({kind})")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    for field, label in FIELD_LABELS.items():
        if field in cfg:
            lines.append(f"{label}: {_fmt(cfg[field])}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm(meta: dict | None, scoped: dict[str, float]) -> str:
    """LLM ブロックと LLM 関連の run スコープ指標を整形する．

    移行前の `run_metadata.json` は 2 つに分かれた — モデル・provider・温度は
    `run.json` の `llm` ブロック，呼び出し数と cache-hit は指標である．cache-hit 率は
    呼び出しが 1 本も無いと «定義できない» ので，行そのものが無い．
    """
    llm = (meta or {}).get("llm")
    lines: list[str] = []
    lines.append("")
    lines.append("LLM (run.json の llm ブロック / run スコープ指標)")
    lines.append("-" * 70)
    if llm:
        lines.append(f"provider         : {llm.get('provider', '-')}")
        lines.append(f"モデル           : {llm.get('model_snapshot', '-')}")
        lines.append(f"温度             : {llm.get('temperature', '-')}")
    else:
        lines.append("llm ブロックなし (掃引/再現の親，または legacy の run)")
    for name, label in SCOPE_LABELS.items():
        if name in scoped:
            lines.append(f"{label}: {scoped[name]:g}")
    if "llm_cache_hit_rate" in scoped:
        lines.append(f"cache-hit 率     : {scoped['llm_cache_hit_rate'] * 100:.1f}%")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_scope(scoped: dict[str, float]) -> str:
    """reproduce 親の «シナリオをまたいだ観測値» を整形する．"""
    lines: list[str] = []
    lines.append("")
    lines.append("シナリオをまたいだ観測値 (scope=sweep)")
    lines.append("-" * 70)
    for name in sorted(scoped):
        lines.append(f"{name:<32}: {scoped[name]:+.4f}")
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
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    if args.results_dir is None:
        results_dir = Path(
            runvault_path(
                EXPERIMENT,
                results_root=args.results_root,
                subcommand="run",
                standalone=True,
            )
        )
    else:
        results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg, cfg_path, kind = load_config(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    meta = load_run_meta(results_dir, required=False)
    scoped = scope_metrics(results_dir, "run")
    swept = scope_metrics(results_dir, "sweep")

    if args.json:
        payload = {
            "run_dir": str(results_dir),
            "source": str(cfg_path),
            "subcommand": kind,
            "parameters": cfg,
            "llm": (meta or {}).get("llm"),
            "run_scope_metrics": scoped,
            "sweep_scope_metrics": swept,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(render_config(cfg, cfg_path, kind))
    if (meta or {}).get("llm") or scoped:
        print(render_llm(meta, scoped))
    if swept:
        print(render_sweep_scope(swept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
