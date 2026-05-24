#!/usr/bin/env python3
"""
visualize_sweep.py — Li et al. (2024) EconAgent スイープ結果 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv を読み，
税率スケール × エージェント数 × LLM モデルの格子について，最終 Gini 係数・
3 年目以降の平均失業率/インフレ率・平均消費傾向を集計し，棒グラフ/ヒートマップで
可視化する (政策レジーム・税率スケール・N 比較)．

Usage:
    uv run econagent-tools visualize-sweep
    uv run econagent-tools visualize-sweep --sweep_dir results/20260524_160000_sweep

Outputs:
    output_dir/
    ├── sweep_gini_by_taxscale.png   ← 税率スケール別の Gini / 失業率 (再分配効果)
    ├── sweep_gini_heatmap.png       ← Gini (tax_scale × N) ヒートマップ
    └── sweep_macro_by_n.png         ← N 別のインフレ・失業 (頑健性確認)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """sweep_summary.csv を読み込む．"""
    path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    return pd.read_csv(path)


def save_gini_by_taxscale(df: pd.DataFrame, out_path: str) -> None:
    """税率スケール別の平均 Gini 係数・平均失業率を棒グラフで比較する (再分配効果)．"""
    scales = sorted(df["tax_scale"].unique())
    gini_means = [df[df["tax_scale"] == s]["final_gini"].mean() for s in scales]
    unemp_means = [df[df["tax_scale"] == s]["mean_unemployment_3y"].mean() * 100.0 for s in scales]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=COLOR_BG)
    labels = [f"{s:g}" for s in scales]

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(labels, gini_means, color="#9C27B0", alpha=0.85)
    for i, v in enumerate(gini_means):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("税率スケール")
    ax.set_ylabel("最終 Gini 係数 (平均)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("税率スケール↑ → Gini↓ (再分配の効果)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(labels, unemp_means, color="#2196F3", alpha=0.85)
    for i, v in enumerate(unemp_means):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("税率スケール")
    ax.set_ylabel("平均失業率 3y+ (%)")
    ax.set_title("税率スケール別の平均失業率")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_gini_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Gini 係数を (tax_scale × N) ヒートマップで可視化する．"""
    agg = df.groupby(["tax_scale", "n_agents"])["final_gini"].mean().reset_index()
    table = agg.pivot(index="tax_scale", columns="n_agents", values="final_gini")
    table = table.sort_index()

    fig, ax = plt.subplots(
        figsize=(1.6 + 1.4 * table.shape[1], 1.4 + 0.9 * table.shape[0]),
        facecolor=COLOR_BG,
    )
    ax.set_facecolor(COLOR_BG)
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([f"{s:g}" for s in table.index])
    ax.set_xlabel("エージェント数 N")
    ax.set_ylabel("税率スケール")
    ax.set_title("最終 Gini 係数 (税率スケール × N)", fontsize=12)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_macro_by_n(df: pd.DataFrame, out_path: str) -> None:
    """エージェント数 N 別の平均インフレ率・失業率 (頑健性確認)．"""
    ns = sorted(df["n_agents"].unique())
    infl = [df[df["n_agents"] == n]["mean_inflation_3y"].mean() * 100.0 for n in ns]
    unemp = [df[df["n_agents"] == n]["mean_unemployment_3y"].mean() * 100.0 for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    x = np.arange(len(ns))
    width = 0.38
    ax.bar(x - width / 2, infl, width, color="#F44336", alpha=0.85, label="インフレ率 3y+ (%)")
    ax.bar(x + width / 2, unemp, width, color="#2196F3", alpha=0.85, label="失業率 3y+ (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("エージェント数 N")
    ax.set_ylabel("%")
    ax.set_title("N 別の平均インフレ・失業 (符号/水準の頑健性)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="econagent-tools visualize-sweep",
        description="Li et al. (2024) EconAgent スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        default="results/latest",
        help="スイープ出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Li et al. (2024) EconAgent スイープ可視化 ===")
    print(f"スイープ: {args.sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/3] sweep_summary.csv を読み込み中 ...")
    df = load_summary(args.sweep_dir)
    print(
        f"      tax_scale {df['tax_scale'].nunique()} 種 × N {df['n_agents'].nunique()} 種 "
        f"× model {df['llm_model'].nunique()} 種"
    )

    print("[2/3] 税率スケール別の Gini/失業率 棒グラフを保存中 ...")
    save_gini_by_taxscale(df, os.path.join(out_dir, "sweep_gini_by_taxscale.png"))

    if df["n_agents"].nunique() > 1:
        print("[3/3] Gini ヒートマップ + N 別マクロ図を保存中 ...")
        save_gini_heatmap(df, os.path.join(out_dir, "sweep_gini_heatmap.png"))
        save_macro_by_n(df, os.path.join(out_dir, "sweep_macro_by_n.png"))
    else:
        print("[3/3] N が単一のためヒートマップ/N 別図はスキップ")

    print("-------------------------------------------------")
    print("税率スケール別の平均 Gini (再分配が不平等に与える影響):")
    for s in sorted(df["tax_scale"].unique()):
        g = df[df["tax_scale"] == s]["final_gini"].mean()
        print(f"  tax_scale={s:<4g} → Ginī = {g:.3f}")

    print("-------------------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
