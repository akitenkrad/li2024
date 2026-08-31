#!/usr/bin/env python3
"""
visualize_sweep.py — Li et al. (2024) EconAgent スイープ結果 可視化スクリプト

スイープ親 run の子 (`subcommand=sweep-point`) から 1 行 1 試行の表を組み直し，
税率スケール × エージェント数 × LLM モデルの格子について，最終 Gini 係数・
3 年目以降の平均失業率/インフレ率・平均消費傾向を集計して棒グラフ/ヒートマップで
可視化する (政策レジーム・税率スケール・N 比較)．

runvault はこの表をディスクに持たない．各子 run が «そのセルの試行 1 本» の月次
指標を丸ごと持っているので，そこから旧 `sweep_summary.csv` と同じ列を作る．
runvault 以前のスイープには `sweep_summary.csv` が残っているので，あればそちらを
読む — そちらが正本だった時期の結果を読めなくする理由はない．

Usage:
    uv run econagent-tools visualize-sweep
    uv run econagent-tools visualize-sweep --sweep-dir "$(runvault path --experiment econagent --latest --subcommand sweep)"

Outputs:
    <results-root>/econagent/figures/<run_slug>/
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
from runvault.read import (
    config_parameters,
    figures_dir,
    load_run_meta,
    metrics_wide,
    run_scope_metrics,
    runvault_path,
    sweep_children,
)

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "econagent"

try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:  # pragma: no cover - フォント未インストール環境用フォールバック
    pass

COLOR_BG = "#FAFAF8"


def _child_row(child_dir: str) -> dict:
    """子 run 1 本を旧 `sweep_summary.csv` の 1 行に畳む．

    条件は `config.json` の `parameters`，シードと試行番号は `run.json` の `rng`，
    モデル名は `run.json` の `llm` ブロック，残りは月次指標から集約する．集約を子 run
    側の指標として書かないのは，どれも月次指標から一意に決まるからである (同じ数を
    2 箇所に置かない)．
    """
    params = config_parameters(child_dir) or {}
    meta = load_run_meta(child_dir) or {}
    scoped = run_scope_metrics(child_dir)
    df = metrics_wide(os.path.join(child_dir, "metrics.csv"))
    after_3y = df[df["step"] >= 36]
    last = df.iloc[-1]
    return {
        "n_agents": params.get("n_agents"),
        "tax_scale": params.get("tax_scale"),
        "llm_model": (meta.get("llm") or {}).get("model_snapshot"),
        "regime": params.get("regime"),
        "run": (meta.get("rng") or {}).get("replicate_index"),
        "seed": (meta.get("rng") or {}).get("master_seed"),
        "final_month": scoped.get("final_month"),
        # 3 年目以降の行が 1 つも無いときの平均は Rust 側と同じく 0.0 とする
        # (`metrics::mean` が空スライスに 0 を返す)．
        "mean_inflation_3y": float(after_3y["inflation_rate"].mean()) if len(after_3y) else 0.0,
        "mean_unemployment_3y": (
            float(after_3y["unemployment_rate"].mean()) if len(after_3y) else 0.0
        ),
        "final_gdp": float(last["nominal_gdp"]),
        "final_gini": float(last["gini_savings"]),
        "mean_work_propensity": float(df["avg_work_propensity"].mean()),
        "mean_consume_propensity": float(df["avg_consume_propensity"].mean()),
        "cache_hit_rate": scoped.get("llm_cache_hit_rate"),
    }


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """1 行 1 試行の表 (条件 + 最終/平均マクロ指標) を返す．"""
    legacy = os.path.join(sweep_dir, "sweep_summary.csv")
    if os.path.exists(legacy):
        return pd.read_csv(legacy)

    children = sweep_children(sweep_dir)
    if not children:
        raise FileNotFoundError(
            f"スイープの子 run が見つかりません: {sweep_dir}\n"
            f"  先に `cargo run --release -- sweep` を実行してください．"
        )
    return pd.DataFrame([_child_row(c) for c in children])


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
        "--sweep-dir",
        "--sweep_dir",
        default=None,
        help="スイープ親 run (省略時は runvault path が返す直近の sweep)",
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

    sweep_dir = args.sweep_dir or runvault_path(
        EXPERIMENT, results_root=args.results_root, subcommand="sweep"
    )
    out_dir = args.output_dir if args.output_dir else figures_dir(sweep_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=== Li et al. (2024) EconAgent スイープ可視化 ===")
    print(f"スイープ: {sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/3] スイープの子 run から 1 行 1 試行の表を組み直し中 ...")
    df = load_summary(sweep_dir)
    print(
        f"      tax_scale {df['tax_scale'].nunique()} 種 × N {df['n_agents'].nunique()} 種 "
        f"× model {df['llm_model'].nunique()} 種 ({len(df)} 試行)"
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
