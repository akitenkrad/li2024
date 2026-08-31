#!/usr/bin/env python3
"""reproduce_paper.py — Li et al. (2024) EconAgent 論文 headline 一括再現レポート．

Rust の `econagent reproduce` が書く sweep 親 run を読む．政策レジームをまたいだ
観測値 (3 年目以降の平均インフレ/失業，Phillips 曲線・Okun の法則の Pearson 相関) は
親の `metrics.csv` の scope=sweep 行に，論文 Fig. 3 の報告値は `reference.csv` に
ある．レジームごとの月次指標は `lineage.parent_run_uid` で親を指す子 run
(`subcommand=reproduce-scenario`) にあり，図はそこから描く:

  - マクロ時系列 (progressive シナリオ): インフレ率・失業率・名目 GDP・Gini 係数．
  - Phillips 曲線 + Okun の法則 散布図: 各点に最小二乗回帰直線を重ね，scipy で
    Pearson r/p を併記する (負相関 = headline)．

観測値と論文値の差は出すが，PASS/OFF の «帯» はここでは判定しない — 帯は論文の主張
ではなくこの再現実装が置いたものなので `reference.csv` に載らず，同じ閾値を Python と
Rust の 2 箇所に置くと食い違う余地ができる．帯つきの判定は `econagent reproduce` の
コンソール出力にある．

`--run` を付けると先に Rust バイナリ (`econagent reproduce`) を実行して最新結果を
作る．`--mock` / `--quick` はそのまま Rust バイナリへ渡す (オフライン・短縮再現)．
runvault 以前の `results/<timestamp>_reproduce/` を `--results-dir` に渡した場合は
`reproduce_summary.json` を読む従来の経路で処理する．

Usage:
    uv run econagent-tools reproduce --run --mock --quick
    uv run econagent-tools reproduce
    uv run econagent-tools reproduce --results-dir "$(runvault path --experiment econagent --latest --subcommand reproduce)"
    uv run econagent-tools reproduce --json

Outputs:
    <results-root>/econagent/figures/<run_slug>/
    ├── macro_timeseries.png    ← インフレ・失業・GDP・Gini の時系列 (progressive)
    └── phillips_okun.png       ← Phillips 曲線 + Okun の法則 散布図 (r/p 併記)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from runvault.read import (
    config_parameters,
    figures_dir,
    load_run_meta,
    runvault_path,
    sweep_children,
)

# 散布図とマクロ時系列の描画ロジックは visualize.py と共有する (重複回避)．
from econagent_tools.visualize import (
    load_metrics,
    save_macro_timeseries,
    save_phillips_okun,
)

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "econagent"

# 論文標準 (headline) のシナリオ．
HEADLINE = "progressive"

# 表示するシナリオの順序 (Rust 側の scenarios_spec と揃える)．
REGIMES = ("progressive", "proportional", "none")


def _run_binary(seed: int, mock: bool, quick: bool, output_dir: str) -> None:
    """cargo run --release -- reproduce を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed),
           "--output-dir", output_dir]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# 親 run の読み取り
# --------------------------------------------------------------------------- #


def sweep_scope_metrics(parent_dir: str) -> dict[str, float]:
    """親の `metrics.csv` の scope=sweep 行 (シナリオをまたいだ観測値)．"""
    path = Path(parent_dir) / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"metrics.csv が見つかりません: {path}\n"
            f"  先に `econagent-tools reproduce --run --mock` を実行してください．"
        )
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["scope"] == "sweep" and row["step"] == "":
                out[row["name"]] = float(row["value"])
    return out


def paper_values(parent_dir: str) -> list[dict]:
    """`reference.csv` の行 (論文が印字した数だけ)．"""
    path = Path(parent_dir) / "reference.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return [
            {"name": r["name"], "value": float(r["value"]),
             "target_id": r["target_id"], "source": r["source"]}
            for r in csv.DictReader(f)
        ]


def scenario_children(parent_dir: str) -> dict[str, str]:
    """政策レジーム → 子 run のディレクトリ．"""
    out: dict[str, str] = {}
    for child in sweep_children(parent_dir):
        params = config_parameters(child, required=False) or {}
        regime = params.get("regime")
        if regime is not None:
            out[regime] = child
    return out


def scenarios(parent_dir: str, scoped: dict[str, float],
              children: dict[str, str]) -> list[dict]:
    """シナリオ別の行を «移行前の scenarios» と同じ形に組み直す．

    3 年目以降の平均と相関は親の scope=sweep 指標にある．最終 GDP・最終 Gini・実行
    月数は子 run の月次指標が正本なので (同じ数を 2 箇所に置かない)，そちらから採る．
    """
    rows: list[dict] = []
    for regime in REGIMES:
        if f"mean_inflation_3y_{regime}" not in scoped:
            continue
        row = {
            "regime": regime,
            "mean_inflation_3y": scoped[f"mean_inflation_3y_{regime}"],
            "mean_unemployment_3y": scoped[f"mean_unemployment_3y_{regime}"],
            # 相関は標本不足・分散 0 のとき «定義できない» ので行そのものが無い．
            "phillips_r": scoped.get(f"phillips_r_{regime}"),
            "okun_r": scoped.get(f"okun_r_{regime}"),
            "final_gdp": None,
            "final_gini": None,
            "final_month": None,
            "run_dir": children.get(regime),
        }
        child = children.get(regime)
        if child is not None:
            df = load_metrics(os.path.join(child, "metrics.csv"))
            last = df.iloc[-1]
            row["final_gdp"] = float(last["nominal_gdp"])
            row["final_gini"] = float(last["gini_savings"])
            row["final_month"] = int(df["month"].max()) + 1
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# legacy (runvault 以前の reproduce_summary.json)
# --------------------------------------------------------------------------- #


def _legacy_summary(results_dir: Path) -> dict | None:
    path = results_dir / "reproduce_summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _legacy_scenarios(summary: dict, results_dir: Path) -> list[dict]:
    rows = []
    for s in summary.get("scenarios", []):
        rows.append({
            "regime": s["regime"],
            "mean_inflation_3y": s["mean_inflation_3y"],
            "mean_unemployment_3y": s["mean_unemployment_3y"],
            "phillips_r": s.get("phillips_r"),
            "okun_r": s.get("okun_r"),
            "final_gdp": s["final_gdp"],
            "final_gini": s["final_gini"],
            "final_month": s["final_month"],
            "run_dir": str(results_dir / s.get("results_subdir", s["regime"])),
        })
    return rows


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #


def _fmt(v: float | None, spec: str) -> str:
    return format(v, spec) if v is not None else "n/a"


def _print_report(rows: list[dict], papers: list[dict],
                  scoped: dict[str, float], results_dir: str) -> None:
    print("=" * 78)
    print("Li et al. (2024) EconAgent — headline (Phillips / Okun / マクロ動態) 再現レポート")
    print(f"  source: {results_dir}")
    print("=" * 78)
    for s in rows:
        print(
            f"  [{s['regime']:<12}] π̄(3y+)={s['mean_inflation_3y']:.3f} "
            f"ū(3y+)={s['mean_unemployment_3y']:.3f} "
            f"GDP={_fmt(s['final_gdp'], '.1f')} Gini={_fmt(s['final_gini'], '.3f')} | "
            f"Phillips r={_fmt(s['phillips_r'], '.3f')} Okun r={_fmt(s['okun_r'], '.3f')}"
        )
    if not papers:
        return
    print("-" * 78)
    print("[観測 vs 論文の報告値 (reference.csv)]")
    for pv in papers:
        obs = scoped.get(pv["name"])
        if obs is None:
            continue
        print(f"  {pv['name']:<26} obs={obs:+.4f} paper={pv['value']:+.4f} "
              f"diff={obs - pv['value']:+.4f}")
        print(f"  {'':<26} 出典: {pv['source']}")
    print("-" * 78)
    print("帯つきの PASS/OFF 判定は `econagent reproduce` のコンソール出力にある — "
          "帯は論文の主張ではないので記録しない．")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="econagent-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help="reproduce の親 run (省略時は runvault path が返す直近の reproduce)")
    parser.add_argument("--results-root", "--results_root", default="results",
                        help="runvault の results ルート (default: results)")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: <results-root>/econagent/figures/<run_slug>)")
    parser.add_argument("--run", action="store_true",
                        help="先に Rust バイナリ (econagent reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true",
                        help="--run 時に scripted mock を使う (オフライン)．")
    parser.add_argument("--quick", action="store_true", help="--run 時に短縮再現 (months=60)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--json", action="store_true",
                        help="サマリを JSON で出力する (図は生成しない)．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(args.seed, args.mock, args.quick, args.results_root)

    results_dir = args.results_dir or runvault_path(
        EXPERIMENT, results_root=args.results_root, subcommand="reproduce"
    )
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"エラー: ディレクトリが存在しません: {results_path}", file=sys.stderr)
        return 1

    legacy = _legacy_summary(results_path)
    if legacy is not None:
        rows = _legacy_scenarios(legacy, results_path)
        scoped, papers = {}, []
        meta = None
    else:
        try:
            scoped = sweep_scope_metrics(results_dir)
        except FileNotFoundError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        papers = paper_values(results_dir)
        rows = scenarios(results_dir, scoped, scenario_children(results_dir))
        meta = load_run_meta(results_dir, required=False)

    if args.json:
        payload = {
            "run_dir": str(results_path),
            "parameters": config_parameters(results_path, required=False)
            or (legacy if legacy is not None else None),
            "scenarios": rows,
            "sweep_scope_metrics": scoped,
            "paper_values": papers,
            "subcommand": (meta or {}).get("subcommand") if meta else None,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    _print_report(rows, papers, scoped, str(results_path))

    out_dir = Path(args.output_dir) if args.output_dir else Path(figures_dir(results_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("-" * 78)
    print(f"図の出力先: {out_dir}")

    # headline シナリオ (progressive) の月次指標からマクロ時系列 + Phillips/Okun 散布．
    headline = next((r for r in rows if r["regime"] == HEADLINE), rows[0] if rows else None)
    if headline is None or headline["run_dir"] is None:
        print("エラー: headline シナリオの run が見つかりません", file=sys.stderr)
        return 1
    metrics_path = Path(headline["run_dir"]) / "metrics.csv"
    if not metrics_path.exists():
        print(f"エラー: metrics.csv が見つかりません: {metrics_path}", file=sys.stderr)
        return 1
    df: pd.DataFrame = load_metrics(str(metrics_path))
    print(f"      headline シナリオ '{headline['regime']}': {len(df)} か月分のマクロ指標")
    save_macro_timeseries(df, str(out_dir / "macro_timeseries.png"))
    save_phillips_okun(df, str(out_dir / "phillips_okun.png"))

    print("-" * 78)
    print("完了．出力ファイル一覧:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s} ({size_kb:6.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
