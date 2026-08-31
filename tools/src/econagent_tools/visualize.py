#!/usr/bin/env python3
"""
visualize.py — Li et al. (2024) EconAgent マクロ経済シミュレーション 可視化スクリプト

runvault の run ディレクトリの `metrics.csv` を読み，
(1) マクロ指標時系列図 (インフレ率・失業率・名目 GDP・Gini 係数) と，
(2) Phillips 曲線 (インフレ率 vs 失業率) ・Okun の法則 (GDP 成長率 vs 失業率変化)
    の散布図 (Pearson 相関 r / p 値を scipy で算出して併記) を生成する．

どの run を見るかは `--results-dir` を省略すれば runvault が答える
(`runvault path --experiment econagent --latest --subcommand run --standalone`)．
`results/` を自分で走査して新しそうなディレクトリを当てにいくことはしない．

図は run ディレクトリの *隣* (`results/econagent/figures/<run_slug>/`) に置く．
`manifest.csv` は `finish()` が確定させたもので，run が終わった後に足したものは
そこに載らないためである．

Usage:
    uv run econagent-tools visualize
    uv run econagent-tools visualize --results-dir "$(runvault path --experiment econagent --latest --subcommand run --standalone)"
    uv run econagent-tools visualize --output-dir out

Outputs:
    output_dir/
    ├── macro_timeseries.png    ← インフレ・失業・GDP・Gini の時系列
    └── phillips_okun.png       ← Phillips 曲線 + Okun の法則 散布図 (r/p 併記)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from runvault.read import figures_dir, metrics_wide, runvault_path
from scipy import stats

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "econagent"

# --------------------------------------------------------------------------- #
# 日本語フォント設定 (CJK フォントが利用不能でも落ちないように try)
# --------------------------------------------------------------------------- #
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:  # pragma: no cover - フォント未インストール環境用フォールバック
    pass

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_INFL = "#F44336"
COLOR_UNEMP = "#2196F3"
COLOR_GDP = "#4CAF50"
COLOR_GINI = "#9C27B0"
COLOR_SCATTER = "#FF9800"


def load_metrics(path: str) -> pd.DataFrame:
    """月次のマクロ指標を 1 か月 1 行の表として読む．

    runvault の `metrics.csv` は long 形式なので `metrics_wide` で横に倒す．時間軸の
    列名は runvault では `step` だが，本モデルの表記は論文に合わせた `month` なので
    こちら側の呼び名に揃えてから返す (legacy の wide な metrics.csv はもともと
    `month` 列を持つので何もしない)．legacy が持つ `year` 列は使わない — `month`
    から一意に決まる導出値である．
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    df = metrics_wide(path)
    if "step" in df.columns and "month" not in df.columns:
        df = df.rename(columns={"step": "month"})
    return df


def save_macro_timeseries(df: pd.DataFrame, out_path: str) -> None:
    """インフレ率・失業率・名目 GDP・Gini 係数の時系列図を保存する (2×2)．"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=COLOR_BG)
    fig.suptitle("Li et al. (2024) EconAgent — マクロ指標時系列", fontsize=14)
    t = df["month"]

    ax = axes[0, 0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, df["inflation_rate"] * 100.0, color=COLOR_INFL, lw=2)
    ax.axhspan(-5.0, 5.0, color="#F44336", alpha=0.08)
    ax.axhline(0.0, color="#888888", lw=0.8, linestyle="--")
    ax.set_xlabel("月 t")
    ax.set_ylabel("インフレ率 (%)")
    ax.set_title("インフレ率 π (3 年目以降 −5%〜5% が論文目標)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, df["unemployment_rate"] * 100.0, color=COLOR_UNEMP, lw=2)
    ax.axhspan(2.0, 12.0, color="#2196F3", alpha=0.08)
    ax.set_xlabel("月 t")
    ax.set_ylabel("失業率 (%)")
    ax.set_title("失業率 u (論文目標 2%〜12%)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, df["nominal_gdp"], color=COLOR_GDP, lw=2)
    ax.set_xlabel("月 t")
    ax.set_ylabel("名目 GDP")
    ax.set_title("名目 GDP (月次)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, df["gini_savings"], color=COLOR_GINI, lw=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("月 t")
    ax.set_ylabel("Gini 係数")
    ax.set_title("貯蓄の Gini 係数 (不平等)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson 相関係数 r と p 値を返す (有効標本が 2 未満なら nan)．"""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def compute_phillips(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Phillips 曲線: 失業率 (x) vs インフレ率 (y)．3 年目以降のみ使う．"""
    sub = df[df["month"] >= 36]
    if sub.empty:
        sub = df
    u = sub["unemployment_rate"].to_numpy() * 100.0
    pi = sub["inflation_rate"].to_numpy() * 100.0
    r, p = _pearson(u, pi)
    return u, pi, r, p


def compute_okun(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Okun の法則: 失業率変化 Δu (x) vs GDP 成長率 (y)．"""
    gdp = df["nominal_gdp"].to_numpy()
    u = df["unemployment_rate"].to_numpy() * 100.0
    # 月次成長率と失業率変化 (差分)．
    with np.errstate(divide="ignore", invalid="ignore"):
        gdp_growth = np.diff(gdp) / np.where(gdp[:-1] == 0.0, np.nan, gdp[:-1]) * 100.0
    du = np.diff(u)
    r, p = _pearson(du, gdp_growth)
    return du, gdp_growth, r, p


def save_phillips_okun(df: pd.DataFrame, out_path: str) -> None:
    """Phillips 曲線と Okun の法則の散布図 (Pearson r/p 併記) を保存する．"""
    u, pi, r_ph, p_ph = compute_phillips(df)
    du, growth, r_ok, p_ok = compute_okun(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=COLOR_BG)
    fig.suptitle("Li et al. (2024) EconAgent — Phillips 曲線 / Okun の法則", fontsize=14)

    # --- Phillips 曲線 ---
    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.scatter(u, pi, color=COLOR_SCATTER, alpha=0.6, edgecolors="none")
    _fit_line(ax, u, pi)
    ax.set_xlabel("失業率 (%)")
    ax.set_ylabel("インフレ率 (%)")
    ax.set_title(f"Phillips 曲線 (3 年目以降)\nPearson r={r_ph:.3f}, p={p_ph:.3g}")
    ax.grid(True, alpha=0.3)

    # --- Okun の法則 ---
    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.scatter(du, growth, color=COLOR_UNEMP, alpha=0.6, edgecolors="none")
    _fit_line(ax, du, growth)
    ax.set_xlabel("失業率変化 Δu (ポイント)")
    ax.set_ylabel("名目 GDP 成長率 (%)")
    ax.set_title(f"Okun の法則\nPearson r={r_ok:.3f}, p={p_ok:.3g}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")
    print(f"      Phillips r={r_ph:.3f} (p={p_ph:.3g}) | Okun r={r_ok:.3f} (p={p_ok:.3g})")


def _fit_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    """有効点に最小二乗の回帰直線を引く (退化時はスキップ)．"""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) < 1e-12:
        return
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), color="#555555", lw=1.5, linestyle="--")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="econagent-tools visualize",
        description="Li et al. (2024) EconAgent マクロ経済 可視化スクリプト",
    )
    p.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    p.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        default=None,
        help="図の保存先 (既定: <results-root>/econagent/figures/<run_slug>)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    results_dir = args.results_dir or runvault_path(
        EXPERIMENT,
        results_root=args.results_root,
        subcommand="run",
        standalone=True,
    )
    metrics_path = os.path.join(results_dir, "metrics.csv")
    out_dir = args.output_dir if args.output_dir else figures_dir(results_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=== Li et al. (2024) EconAgent マクロ経済 可視化 ===")
    print(f"メトリクス: {metrics_path}")
    print(f"出力先:     {out_dir}")
    print("-----------------------------------------")

    print("[1/2] マクロ指標時系列を保存中 ...")
    df = load_metrics(metrics_path)
    print(f"      {len(df)} か月分のマクロ指標")
    save_macro_timeseries(df, os.path.join(out_dir, "macro_timeseries.png"))

    print("[2/2] Phillips 曲線 / Okun の法則 散布図を保存中 ...")
    save_phillips_okun(df, os.path.join(out_dir, "phillips_okun.png"))

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
