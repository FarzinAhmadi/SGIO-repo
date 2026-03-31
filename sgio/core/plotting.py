"""Validation visualisation: 6-panel figure for S-MGIL day-2 holdout."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLORS = {
    "natural": "#888888",
    "r1": "#2166ac",
    "r2": "#4dac26",
    "r3": "#d6604d",
    "r4": "#762a83",
}


def plot_validation_figures(
    df: pd.DataFrame,
    food_cap: float = 5000.0,
    output_prefix: str | Path = "smgil_validation_figures",
):
    """Generate the 6-panel validation figure and save to PDF + PNG.

    Parameters
    ----------
    df          : DataFrame with columns d_nut_natural, d_nut_r1..r4,
                  d_food_natural, d_food_r1..r4
    food_cap    : winsorise food-distance values above this
    output_prefix : filename prefix for saved figures
    """
    # Winsorise food distances
    for col in [c for c in df.columns if c.startswith("d_food_r")]:
        df[col] = df[col].where(df[col] < food_cap)

    nat_nut_med = df["d_nut_natural"].median()
    nat_food_med = df["d_food_natural"].median()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"S-MGIL Day 2 Holdout Validation (n={len(df)})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # -- Fig 1: Nutrient distance scatter (natural vs r=1) --
    ax = axes[0, 0]
    merged = df[["d_nut_natural", "d_nut_r1"]].dropna()
    ax.scatter(
        merged["d_nut_natural"],
        merged["d_nut_r1"],
        alpha=0.65,
        color=COLORS["r1"],
        edgecolors="white",
        linewidths=0.4,
        s=45,
    )
    lim = max(merged["d_nut_natural"].max(), merged["d_nut_r1"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.9, alpha=0.5, label="y=x")
    ax.axhline(
        nat_nut_med, color=COLORS["natural"], lw=1.2, ls=":", alpha=0.7,
        label=f"Natural median ({nat_nut_med:.2f})",
    )
    ax.set_xlabel("Nutrient distance: Day1 -> Day2 (natural)", fontsize=9)
    ax.set_ylabel("Nutrient distance: Day1 -> r=1 rec.", fontsize=9)
    ax.set_title("Fig 1 -- Nutrient distance scatter (r=1)", fontsize=10)
    ax.legend(fontsize=8)
    pct = 100 * (merged["d_nut_r1"] < merged["d_nut_natural"]).mean()
    ax.text(
        0.97, 0.05, f"{pct:.0f}% below diagonal",
        ha="right", va="bottom", transform=ax.transAxes, fontsize=8, color=COLORS["r1"],
    )

    # -- Fig 2: Nutrient distance boxplot across r --
    ax = axes[0, 1]
    nut_data = [df["d_nut_natural"].dropna().values]
    nut_labels = ["Natural"]
    nut_colors = [COLORS["natural"]]
    for r in [1, 2, 3, 4]:
        col = f"d_nut_r{r}"
        if col in df.columns:
            nut_data.append(df[col].dropna().values)
            nut_labels.append(f"r={r}\n(n={len(df[col].dropna())})")
            nut_colors.append(COLORS[f"r{r}"])
    _styled_boxplot(ax, nut_data, nut_labels, nut_colors)
    ax.axhline(nat_nut_med, color=COLORS["natural"], lw=1.2, ls=":", alpha=0.7)
    ax.set_ylabel("Normalised nutrient L2 distance", fontsize=9)
    ax.set_title("Fig 2 -- Nutrient distance by iteration", fontsize=10)

    # -- Fig 3: Food distance boxplot across r --
    ax = axes[0, 2]
    food_data = [df["d_food_natural"].dropna().values]
    food_labels = ["Natural"]
    food_colors = [COLORS["natural"]]
    for r in [1, 2, 3, 4]:
        col = f"d_food_r{r}"
        if col in df.columns:
            s = df[col].dropna()
            food_data.append(s.values)
            food_labels.append(f"r={r}\n(n={len(s)})")
            food_colors.append(COLORS[f"r{r}"])
    _styled_boxplot(ax, food_data, food_labels, food_colors)
    ax.axhline(nat_food_med, color=COLORS["natural"], lw=1.2, ls=":", alpha=0.7)
    ax.set_ylabel("Similarity-weighted food L2 distance (g)", fontsize=9)
    ax.set_title("Fig 3 -- Food distance by iteration (winsorised)", fontsize=10)

    # -- Fig 4: % below natural median bar chart --
    ax = axes[1, 0]
    rs = [1, 2, 3, 4]
    pct_nut, pct_food = [], []
    for r in rs:
        for metric, pct_list, med in [
            ("d_nut", pct_nut, nat_nut_med),
            ("d_food", pct_food, nat_food_med),
        ]:
            col = f"{metric}_r{r}"
            if col in df.columns:
                s = df[col].dropna()
                pct_list.append(100 * (s < med).mean())
            else:
                pct_list.append(np.nan)

    x = np.arange(len(rs))
    w = 0.35
    bars1 = ax.bar(x - w / 2, pct_nut, w, label="Nutrient space", color="#2166ac", alpha=0.8)
    bars2 = ax.bar(x + w / 2, pct_food, w, label="Food space", color="#d6604d", alpha=0.8)
    ax.axhline(50, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={r}" for r in rs])
    ax.set_ylabel("% participants below natural median", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Fig 4 -- Recommendations vs natural baseline", fontsize=10)
    ax.legend(fontsize=8)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8,
            )

    # -- Fig 5a: Nutrient ratio histograms --
    ax = axes[1, 1]
    for r, color in [(1, COLORS["r1"]), (2, COLORS["r2"]), (3, COLORS["r3"])]:
        nc = f"d_nut_r{r}"
        merged = df[[nc, "d_nut_natural"]].dropna()
        ratio = merged[nc] / merged["d_nut_natural"]
        ax.hist(ratio, bins=20, alpha=0.45, color=color, label=f"r={r}", density=True)
    ax.axvline(1.0, color="black", lw=1.2, ls="--", label="ratio=1 (= natural)")
    ax.set_xlabel("Rec. distance / Natural distance (nutrient space)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Fig 5a -- Nutrient distance ratio distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 4)

    # -- Fig 5b: Food ratio histograms --
    ax = axes[1, 2]
    for r, color in [(1, COLORS["r1"]), (2, COLORS["r2"]), (3, COLORS["r3"])]:
        fc_rec = f"d_food_r{r}"
        merged = df[[fc_rec, "d_food_natural"]].dropna()
        ratio = merged[fc_rec] / merged["d_food_natural"]
        ratio = ratio[ratio < 5]
        ax.hist(ratio, bins=20, alpha=0.45, color=color, label=f"r={r}", density=True)
    ax.axvline(1.0, color="black", lw=1.2, ls="--", label="ratio=1 (= natural)")
    ax.set_xlabel("Rec. distance / Natural distance (food space, clipped at 5)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Fig 5b -- Food distance ratio distribution", fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    output_prefix = Path(output_prefix)
    plt.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", dpi=200)
    plt.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.show()
    print(f"Saved {output_prefix.with_suffix('.pdf')} and {output_prefix.with_suffix('.png')}")


def _styled_boxplot(ax, data, labels, colors):
    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.5),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=8)


def print_validation_stats(df: pd.DataFrame, food_cap: float = 5000.0):
    """Print summary statistics for the day-2 validation."""
    df_clean = df.copy()
    for col in [c for c in df.columns if c.startswith("d_food_r")]:
        df_clean[col] = df_clean[col].where(df_clean[col] < food_cap)

    print("=== NUTRIENT DISTANCE (all n) ===")
    print(
        f"Natural:  median={df['d_nut_natural'].median():.3f}  "
        f"mean={df['d_nut_natural'].mean():.3f}"
    )
    for r in [1, 2, 3, 4]:
        col = f"d_nut_r{r}"
        if col in df.columns:
            s = df[col].dropna()
            print(
                f"r={r} (n={len(s)}): median={s.median():.3f}  mean={s.mean():.3f}  "
                f"pct<natural_median="
                f"{100 * (s < df['d_nut_natural'].median()).mean():.0f}%"
            )

    print("\n=== FOOD DISTANCE (winsorised) ===")
    print(
        f"Natural:  median={df['d_food_natural'].median():.1f}  "
        f"mean={df['d_food_natural'].mean():.1f}"
    )
    for r in [1, 2, 3, 4]:
        col = f"d_food_r{r}"
        if col in df_clean.columns:
            s = df_clean[col].dropna()
            print(
                f"r={r} (n={len(s)}): median={s.median():.1f}  mean={s.mean():.1f}  "
                f"pct<natural_median="
                f"{100 * (s < df['d_food_natural'].median()).mean():.0f}%"
            )

    print("\n=== PARTICIPATION BREAKDOWN ===")
    print(f"r=0 (cost too high): {(df['n_iter'] == 0).sum()}")
    print(f"r>=1: {(df['n_iter'] >= 1).sum()}")
    print(f"r>=4: {(df['n_iter'] >= 4).sum()}")
    print(f"Coverage=0 (no crosswalk): {(df['coverage_day2'] == 0).sum()}")
