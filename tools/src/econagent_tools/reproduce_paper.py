#!/usr/bin/env python3
"""reproduce_paper.py — Li et al. (2024) EconAgent 論文 headline 一括再現レポート．

Rust の `econagent reproduce` が書き出す `reproduce_summary.json` (政策レジーム別の
マクロ動態と，Phillips 曲線 (失業率 vs インフレ率; 負相関)・Okun の法則 (失業率変化
vs GDP 成長率; 負相関) の観測相関を期待符号アンカーと突合した PASS/off) を読み，
論文の headline を再現する:

  - マクロ時系列 (progressive シナリオ): インフレ率・失業率・名目 GDP・Gini 係数．
  - Phillips 曲線 + Okun の法則 散布図: 各点に最小二乗回帰直線を重ね，scipy で
    Pearson r/p を併記する (負相関 = headline)．

`--run` を付けると先に Rust バイナリ (`econagent reproduce`) を実行して最新結果を
作る．`--mock` / `--quick` はそのまま Rust バイナリへ渡す (オフライン・短縮再現)．

Usage:
    econagent-tools reproduce
    econagent-tools reproduce --run --mock --quick
    econagent-tools reproduce --results-dir results/20260530_000000_reproduce
    econagent-tools reproduce --json

Outputs:
    <results_dir>/figures/
    ├── macro_timeseries.png    ← インフレ・失業・GDP・Gini の時系列 (progressive)
    └── phillips_okun.png       ← Phillips 曲線 + Okun の法則 散布図 (r/p 併記)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from socsim_tools.io import resolve_results_dir

# 散布図とマクロ時系列の描画ロジックは visualize.py と共有する (重複回避)．
from econagent_tools.visualize import save_macro_timeseries, save_phillips_okun


def _run_binary(seed: int, mock: bool, quick: bool) -> None:
    """cargo run --release -- reproduce を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed)]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_summary(results_dir: Path) -> dict:
    path = results_dir / "reproduce_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"reproduce_summary.json が見つかりません: {path}\n"
            f"  先に `econagent-tools reproduce --run` を実行してください．"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _print_table(summary: dict) -> None:
    print("=" * 78)
    print("Li et al. (2024) EconAgent — headline (Phillips / Okun / マクロ動態) 再現レポート")
    print(f"  paper : {summary.get('paper', '')}")
    print(
        f"  config: N={summary.get('n_agents')} months={summary.get('months')} "
        f"| mock={summary.get('mock')} quick={summary.get('quick')}"
    )
    print("=" * 78)
    for s in summary.get("scenarios", []):
        ph = s.get("phillips_r")
        ok = s.get("okun_r")
        ph_s = f"{ph:.3f}" if ph is not None else "n/a"
        ok_s = f"{ok:.3f}" if ok is not None else "n/a"
        print(
            f"  [{s['regime']:<12}] π̄(3y+)={s['mean_inflation_3y']:.3f} "
            f"ū(3y+)={s['mean_unemployment_3y']:.3f} GDP={s['final_gdp']:.1f} "
            f"Gini={s['final_gini']:.3f} | Phillips r={ph_s} Okun r={ok_s}"
        )
    print("-" * 78)
    n_pass = 0
    for a in summary.get("anchors", []):
        hi = a["target_hi"]
        hi_str = "∞" if hi is None or hi == float("inf") or hi > 1e30 else f"{hi:.3f}"
        status = "PASS" if a["pass"] else "OFF "
        if a["pass"]:
            n_pass += 1
        print(
            f"[{status}] {a['name']:<52} "
            f"obs={a['observed']:.4f} target=[{a['target_lo']:.3f},{hi_str}] "
            f"paper={a['paper_value']}"
        )
    print("-" * 78)
    print(f"{n_pass}/{len(summary.get('anchors', []))} アンカーが in-band")


def _headline_subdir(summary: dict) -> str:
    """図に使う headline シナリオ (progressive) のサブディレクトリ名を返す．"""
    for s in summary.get("scenarios", []):
        if s.get("regime") == "progressive":
            return s.get("results_subdir", "progressive")
    # progressive が無ければ最初のシナリオ．
    scen = summary.get("scenarios", [])
    return scen[0]["results_subdir"] if scen else "progressive"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="econagent-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument(
        "--output-dir", "--output_dir", default=None, help="図の保存先 (既定: <results>/figures)"
    )
    parser.add_argument(
        "--run", action="store_true", help="先に Rust バイナリ (econagent reproduce) を実行する．"
    )
    parser.add_argument("--mock", action="store_true", help="--run 時に scripted mock を使う (オフライン)．")
    parser.add_argument("--quick", action="store_true", help="--run 時に短縮再現 (months=60)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--json", action="store_true", help="サマリを JSON で出力する (図は生成しない)．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(args.seed, args.mock, args.quick)

    results_dir = resolve_results_dir(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_table(summary)

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("-" * 78)
    print(f"図の出力先: {out_dir}")

    # headline シナリオ (progressive) の metrics.csv からマクロ時系列 + Phillips/Okun 散布を描く．
    subdir = _headline_subdir(summary)
    metrics_path = results_dir / subdir / "metrics.csv"
    if not metrics_path.exists():
        print(f"エラー: metrics.csv が見つかりません: {metrics_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(metrics_path)
    print(f"      headline シナリオ '{subdir}': {len(df)} か月分のマクロ指標")
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
