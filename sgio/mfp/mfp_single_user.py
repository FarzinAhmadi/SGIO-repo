"""Single MFP user S-MGIL pipeline demo.

Mirrors single_respondent.py but uses MyFitnessPal data in servings space.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from smgil import build_crosswalk, build_A_b_mfp, MFP_DASH_CONSTRAINTS
from smgil import run_smgil_tradeoff
from smgil.mfp_preprocessing import build_mfp_crosswalk, build_mfp_observation_vector
from smgil.tradeoff import tune_weights
from smgil.reporting import print_full_report, summary_table

# -- Configuration ----------------------------------------------------------
DATA_DIR = Path("data")
MFP_CSV = DATA_DIR / "myfitnesspal/processed/myfitnesspal_foods.csv"
MFP_SIMILARITY_JSON = DATA_DIR / "similarity/myfitnesspal_to_usda.json"
FNDDS_PATH = DATA_DIR / "nhanes/2017/rawdata/DRXFCD_J.xpt"
USDA_PARQUET = DATA_DIR / "similarity/usda_index.parquet"
NUTRIENT_CSV = DATA_DIR / "usda/2017-2018/processed/nutrient_values.csv"

K_NEIGHBORS = 5
MAX_ITERATIONS = 4
COST_THRESHOLD = 100  # servings^2 — much smaller scale than grams^2

# -- Select a user ----------------------------------------------------------
# Find a user with >= 30 days and >= 8 distinct foods/day average
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
    # Relax filters
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

print("Building NHANES crosswalk (for USDA neighbor nutrient lookup) ...")
nhanes_crosswalk = build_crosswalk(
    fndds_path=FNDDS_PATH,
    usda_parquet_path=USDA_PARQUET,
)

# -- Build observation vector ------------------------------------------------
user_df = mfp_df[mfp_df["user_id"] == USER_ID]

x_vector, W_S, item_index, meta = build_mfp_observation_vector(
    user_df=user_df,
    similarity_index=similarity_index,
    name_to_id=name_to_id,
    user_id=USER_ID,
    K=K_NEIGHBORS,
    verbose=True,
)

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

# -- Sanity check: A @ x_vector vs actual daily nutrients --------------------
print("\nSanity check: A @ x_vector vs DASH bounds")
from smgil.constraints import check_observed_intake
check_df = check_observed_intake(A, b, x_vector, constraint_names, MFP_DASH_CONSTRAINTS)
print(check_df.to_string(index=False))

# -- Reshape & tune ----------------------------------------------------------
X_obs = x_vector.reshape(1, -1)
print(f"\nX_obs shape: {X_obs.shape}   (K=1 observation, n={X_obs.shape[1]} items)")
print(f"A shape    : {A.shape}")
assert X_obs.shape[1] == A.shape[1], "Dimension mismatch between X and A!"

W_S_tuned = tune_weights(W_S, meta)

# -- Run S-MGIL tradeoff path ------------------------------------------------
tradeoff_path = run_smgil_tradeoff(
    A, b, X_obs, W_S_tuned, item_index, meta,
    constraint_names=constraint_names,
    max_iterations=MAX_ITERATIONS,
    cost_threshold=COST_THRESHOLD,
)

# -- Report -------------------------------------------------------------------
print_full_report(
    x_vector, A, b, item_index, meta,
    constraint_names, directions, tradeoff_path, USER_ID,
)

# -- Export -------------------------------------------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df_summary = summary_table(tradeoff_path)
output_path = OUTPUT_DIR / f"smgil_mfp_tradeoff_user_{USER_ID}.csv"
df_summary.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")
