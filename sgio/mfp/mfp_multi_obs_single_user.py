"""Single MFP user: multi-observation S-MGIL vs single-observation baseline.

Uses per-day observation vectors instead of collapsing to a single mean.
The solver finds per-day recommendations sharing a common tight constraint set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from smgil import build_crosswalk, build_A_b_mfp, MFP_DASH_CONSTRAINTS
from smgil import run_smgil_tradeoff
from smgil.mfp_preprocessing import build_mfp_crosswalk, build_mfp_daily_matrix
from smgil.tradeoff import tune_weights, run_smgil_tradeoff_multi_obs
from smgil.reporting import print_full_report

# -- Configuration ----------------------------------------------------------
DATA_DIR = Path("data")
MFP_CSV = DATA_DIR / "myfitnesspal/processed/myfitnesspal_foods.csv"
MFP_SIMILARITY_JSON = DATA_DIR / "similarity/myfitnesspal_to_usda.json"
FNDDS_PATH = DATA_DIR / "nhanes/2017/rawdata/DRXFCD_J.xpt"
USDA_PARQUET = DATA_DIR / "similarity/usda_index.parquet"
NUTRIENT_CSV = DATA_DIR / "usda/2017-2018/processed/nutrient_values.csv"

K_NEIGHBORS = 5
MAX_ITERATIONS = 4
COST_THRESHOLD = 100

# -- Select a user ----------------------------------------------------------
print("Loading MFP data ...")
mfp_df = pd.read_csv(MFP_CSV, low_memory=False)

user_stats = mfp_df.groupby("user_id").agg(
    n_days=("date", "nunique"),
    n_foods=("food_name", "nunique"),
    n_rows=("food_name", "count"),
).reset_index()
user_stats["foods_per_day"] = user_stats["n_foods"] / user_stats["n_days"]

candidates = user_stats[
    (user_stats["n_days"] >= 30) &
    (user_stats["foods_per_day"] >= 8)
].sort_values("n_days", ascending=False)

print(f"Users with >= 30 days and >= 8 foods/day: {len(candidates)}")
if candidates.empty:
    candidates = user_stats[user_stats["n_days"] >= 14].sort_values(
        "n_days", ascending=False
    )
    print(f"Relaxed: users with >= 14 days: {len(candidates)}")

USER_ID = int(candidates.iloc[0]["user_id"])
print(f"Selected user: {USER_ID} "
      f"({int(candidates.iloc[0]['n_days'])} days, "
      f"{candidates.iloc[0]['foods_per_day']:.1f} foods/day)")

# -- Build crosswalks -------------------------------------------------------
print("\nBuilding MFP crosswalk ...")
name_to_id, similarity_index = build_mfp_crosswalk(MFP_SIMILARITY_JSON)

print("Building NHANES crosswalk ...")
nhanes_crosswalk = build_crosswalk(
    fndds_path=FNDDS_PATH,
    usda_parquet_path=USDA_PARQUET,
)

# -- Build daily observation matrix ------------------------------------------
user_df = mfp_df[mfp_df["user_id"] == USER_ID]

X_daily, x_vector, W_S, item_index, meta = build_mfp_daily_matrix(
    user_df=user_df,
    similarity_index=similarity_index,
    name_to_id=name_to_id,
    user_id=USER_ID,
    K=K_NEIGHBORS,
    verbose=True,
)

K_days = X_daily.shape[0]
n = X_daily.shape[1]

# -- Build DASH constraints --------------------------------------------------
A, b, constraint_names, directions = build_A_b_mfp(
    item_index=item_index,
    mfp_nutrients=meta["mfp_nutrients"],
    fndds_nutrient_csv=NUTRIENT_CSV,
    nhanes_crosswalk=nhanes_crosswalk,
    x_vector=x_vector,
    similarity_index=similarity_index,
    verbose=True,
)

# -- Tune weights -------------------------------------------------------------
W_S_tuned = tune_weights(W_S, meta)

# -- Run single-obs baseline --------------------------------------------------
print("\n" + "=" * 70)
print("  SINGLE-OBSERVATION BASELINE  (mean across all days)")
print("=" * 70)

X_obs = x_vector.reshape(1, -1)
tradeoff_single = run_smgil_tradeoff(
    A, b, X_obs, W_S_tuned, item_index, meta,
    constraint_names=constraint_names,
    max_iterations=MAX_ITERATIONS,
    cost_threshold=COST_THRESHOLD,
)

# -- Run multi-obs S-MGIL ----------------------------------------------------
print("\n" + "=" * 70)
print(f"  MULTI-OBSERVATION S-MGIL  (K={K_days} daily observations)")
print("=" * 70)

tradeoff_multi = run_smgil_tradeoff_multi_obs(
    A, b, X_daily, W_S_tuned, item_index, meta,
    constraint_names=constraint_names,
    max_iterations=MAX_ITERATIONS,
    cost_threshold=COST_THRESHOLD,
)

# -- Compare results ----------------------------------------------------------
print("\n" + "=" * 70)
print("  COMPARISON: Single-Obs vs Multi-Obs")
print("=" * 70)

signs = np.array([1 if d == "upper" else -1 for d in directions])
scale = np.abs(b)
scale[scale < 1e-6] = 1.0

n_compare = min(len(tradeoff_single), len(tradeoff_multi))

print(f"\n  {'Iter':<6s}  {'Method':<12s}  {'Tight Constraints':<40s}  "
      f"{'d_food':>8s}  {'d_foodW':>8s}")
print(f"  {'-'*6}  {'-'*12}  {'-'*40}  {'-'*8}  {'-'*8}")

for i in range(n_compare):
    rs = tradeoff_single[i]
    rm = tradeoff_multi[i]

    # Single-obs distances
    diff_s = x_vector - rs["z"]
    d_food_s = float(np.linalg.norm(diff_s))
    d_foodW_s = float(np.sqrt(np.dot(W_S * diff_s, diff_s)))

    # Multi-obs distances (mean recommendation vs mean observation)
    diff_m = x_vector - rm["z"]
    d_food_m = float(np.linalg.norm(diff_m))
    d_foodW_m = float(np.sqrt(np.dot(W_S * diff_m, diff_m)))

    tight_s = ", ".join(rs["tight_constraints"])
    tight_m = ", ".join(rm["tight_constraints"])

    print(f"  r={rs['iteration']:<4d}  {'single':<12s}  {tight_s:<40s}  "
          f"{d_food_s:8.3f}  {d_foodW_s:8.3f}")
    print(f"  r={rm['iteration']:<4d}  {'multi-obs':<12s}  {tight_m:<40s}  "
          f"{d_food_m:8.3f}  {d_foodW_m:8.3f}")
    print()

# Per-day variability in multi-obs recommendations
if tradeoff_multi:
    print("\n  Per-day recommendation variability (last iteration):")
    last = tradeoff_multi[-1]
    Z_all = last["z_all"]
    Z_mean = last["z"]
    per_day_spread = np.std(Z_all, axis=0)
    active_items = np.where(Z_mean > 0.01)[0]
    if len(active_items) > 0:
        idx_to_code = {v: k for k, v in item_index.items()}
        name_map = {}
        for item in meta["observed_items"] + meta["neighbor_items"]:
            name_map[item["food_code"]] = item["food_name"]

        items_by_spread = sorted(active_items, key=lambda i: per_day_spread[i],
                                 reverse=True)
        print(f"  {'Food':<45s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
        print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for idx in items_by_spread[:10]:
            code = idx_to_code.get(idx)
            fname = name_map.get(code, str(code))[:45]
            vals = Z_all[:, idx]
            print(f"  {fname:<45s}  {Z_mean[idx]:8.2f}  {vals.std():8.2f}  "
                  f"{vals.min():8.2f}  {vals.max():8.2f}")

# -- Export -------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

output_path = OUTPUT_DIR / f"smgil_mfp_multi_obs_user_{USER_ID}.csv"
rows = []
for r in tradeoff_multi:
    for s in r["swaps"]:
        rows.append({
            "iteration": r["iteration"],
            "tight_constraints": ", ".join(r["tight_constraints"]),
            "food_code": s["food_code"],
            "food_name": s["food_name"],
            "observed_qty": s["observed_qty"],
            "recommended_qty": s["recommended_qty"],
            "delta": s["delta"],
            "action": s["action"],
        })
if rows:
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved multi-obs results to {output_path}")
