"""Generate validation figures and run statistical tests.

Reads the CSV produced by cohort_validation.py and generates:
- 6-panel validation figure (PDF + PNG)
- Wilcoxon signed-rank tests
"""

from pathlib import Path

import pandas as pd
from scipy import stats

from smgil.plotting import plot_validation_figures, print_validation_stats

# ── Configuration ──────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
VALIDATION_CSV = OUTPUT_DIR / "smgil_day2_validation.csv"
FOOD_CAP = 5000

# ── Load results ──────────────────────────────────────────────────
df = pd.read_csv(VALIDATION_CSV)

# ── Summary statistics ────────────────────────────────────────────
print_validation_stats(df, food_cap=FOOD_CAP)

# ── Figures ───────────────────────────────────────────────────────
plot_validation_figures(df, food_cap=FOOD_CAP, output_prefix=OUTPUT_DIR / "smgil_validation_figures")

# ── Wilcoxon signed-rank tests ────────────────────────────────────
print("\n=== Wilcoxon Signed-Rank Test ===")
print("H0: rec distance >= natural distance")
print("H1: rec distance < natural distance (one-sided)\n")

for metric, label in [
    ("d_nut", "Nutrient space"),
    ("d_food", "Food space (winsorised)"),
]:
    print(f"-- {label} --")
    nat_col = f"{metric}_natural"
    for r in [1, 2, 3, 4]:
        rec_col = f"{metric}_r{r}"
        if rec_col not in df.columns:
            continue
        merged = df[[nat_col, rec_col]].dropna()
        if len(merged) < 10:
            continue
        stat, p = stats.wilcoxon(
            merged[rec_col], merged[nat_col], alternative="less"
        )
        n = len(merged)
        med_rec = merged[rec_col].median()
        med_nat = merged[nat_col].median()
        print(
            f"  r={r} (n={n}): median rec={med_rec:.3f}, "
            f"median nat={med_nat:.3f}, W={stat:.1f}, p={p:.2e}"
        )
    print()
