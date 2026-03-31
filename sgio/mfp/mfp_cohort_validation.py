"""Temporal-split holdout validation over a cohort of MFP users.

For each user: split dates into train/test halves, build pipeline from
training data, run S-MGIL, project test-period intake, compute distances.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from smgil import build_crosswalk, build_A_b_mfp, MFP_DASH_CONSTRAINTS
from smgil import run_smgil_tradeoff
from smgil.mfp_preprocessing import build_mfp_crosswalk, build_mfp_observation_vector
from smgil.tradeoff import tune_weights
from smgil.validation import project_mfp_holdout, compute_distances

# -- Configuration ----------------------------------------------------------
DATA_DIR = Path("data")
MFP_CSV = DATA_DIR / "myfitnesspal/processed/myfitnesspal_foods.csv"
MFP_USERS_CSV = DATA_DIR / "myfitnesspal/processed/myfitnesspal_users.csv"
MFP_SIMILARITY_JSON = DATA_DIR / "similarity/myfitnesspal_to_usda.json"
FNDDS_PATH = DATA_DIR / "nhanes/2017/rawdata/DRXFCD_J.xpt"
USDA_PARQUET = DATA_DIR / "similarity/usda_index.parquet"
NUTRIENT_CSV = DATA_DIR / "usda/2017-2018/processed/nutrient_values.csv"

K_NEIGHBORS = 5
N_COHORT = 200
MAX_ITER = 4
COST_THRESHOLD = 100
RANDOM_SEED = 42
MIN_DAYS = 14
MIN_FOODS_PER_DAY = 5
MIN_CALORIES = 800
MAX_CALORIES = 5000

# MFP nutrient columns for computing test-period actual nutrients
MFP_NUTRIENT_COLS = ["sodium", "sat fat", "sugar", "chol", "fat",
                     "fiber", "potass.", "calcium", "protein"]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "smgil_mfp_temporal_validation.csv"

# -- Load shared data -------------------------------------------------------
print("Loading MFP foods CSV ...")
mfp_df = pd.read_csv(MFP_CSV, low_memory=False)

print("Loading MFP users CSV (for calorie filtering) ...")
users_df = pd.read_csv(MFP_USERS_CSV, low_memory=False)

print("Building MFP crosswalk ...")
name_to_id, similarity_index = build_mfp_crosswalk(MFP_SIMILARITY_JSON)

print("Building NHANES crosswalk ...")
nhanes_crosswalk = build_crosswalk(fndds_path=FNDDS_PATH, usda_parquet_path=USDA_PARQUET)

# -- Select cohort -----------------------------------------------------------
# Compute per-user stats
user_day_stats = users_df.groupby("user_id").agg(
    n_days=("date", "nunique"),
    avg_calories=("calories", "mean"),
).reset_index()

food_stats = mfp_df.groupby("user_id").agg(
    n_foods=("food_name", "nunique"),
    n_food_days=("date", "nunique"),
).reset_index()
food_stats["foods_per_day"] = food_stats["n_foods"] / food_stats["n_food_days"]

stats = user_day_stats.merge(food_stats[["user_id", "foods_per_day"]], on="user_id", how="inner")

eligible = stats[
    (stats["n_days"] >= MIN_DAYS) &
    (stats["foods_per_day"] >= MIN_FOODS_PER_DAY) &
    (stats["avg_calories"] >= MIN_CALORIES) &
    (stats["avg_calories"] <= MAX_CALORIES)
]

print(f"Eligible users (>={MIN_DAYS} days, >={MIN_FOODS_PER_DAY} foods/day, "
      f"{MIN_CALORIES}-{MAX_CALORIES} kcal): {len(eligible)}")

rng = np.random.default_rng(RANDOM_SEED)
eligible_ids = [int(x) for x in eligible["user_id"].tolist()]
rng.shuffle(eligible_ids)
cohort = eligible_ids[:N_COHORT]
print(f"Selected cohort: {len(cohort)}")

# -- Main loop ---------------------------------------------------------------
records = []
failed = 0

for i, uid in enumerate(cohort):
    print(f"[{i + 1:3d}/{len(cohort)}] user={uid}", end=" ... ")

    try:
        user_food_df = mfp_df[mfp_df["user_id"] == uid]
        all_dates = sorted(user_food_df["date"].unique())
        n_dates = len(all_dates)

        if n_dates < MIN_DAYS:
            print(f"only {n_dates} days, skipping")
            failed += 1
            continue

        # Temporal split: first half = train, second half = test
        mid = n_dates // 2
        train_dates = list(all_dates[:mid])
        test_dates = list(all_dates[mid:])
        n_train_days = len(train_dates)
        n_test_days = len(test_dates)

        # Build observation vector from training dates
        x_vector, W_S, item_index, meta = build_mfp_observation_vector(
            user_df=user_food_df,
            similarity_index=similarity_index,
            name_to_id=name_to_id,
            user_id=uid,
            train_dates=train_dates,
            K=K_NEIGHBORS,
            verbose=False,
        )

        # Build constraints
        A, b, constraint_names, directions = build_A_b_mfp(
            item_index=item_index,
            mfp_nutrients=meta["mfp_nutrients"],
            fndds_nutrient_csv=NUTRIENT_CSV,
            nhanes_crosswalk=nhanes_crosswalk,
            x_vector=x_vector,
            similarity_index=similarity_index,
            verbose=False,
        )

        W_S_tuned = tune_weights(W_S, meta)
        X_obs = x_vector.reshape(1, -1)

        signs = np.array([1 if d == "upper" else -1 for d in directions])
        scale = np.abs(b)
        scale[scale < 1e-6] = 1.0  # avoid division by zero
        n1 = signs * (A @ x_vector)  # training-period nutrients

        # Test-period projection
        test_food_df = user_food_df[user_food_df["date"].isin(test_dates)]
        x_test, coverage = project_mfp_holdout(
            test_food_df, item_index, name_to_id, similarity_index,
        )

        # Test-period actual nutrients from MFP user-day totals
        test_user_days = users_df[
            (users_df["user_id"] == uid) &
            (users_df["date"].isin(test_dates))
        ]
        n2_values = []
        for mfp_col in MFP_NUTRIENT_COLS:
            if mfp_col in test_user_days.columns:
                n2_values.append(test_user_days[mfp_col].sum() / n_test_days)
            else:
                n2_values.append(0.0)
        n2 = np.array(n2_values, dtype=float)

        # Natural distances (train vs test, no optimization)
        d_nut_nat = float(np.linalg.norm((n1 - n2) / scale))
        d_food_nat = float(np.linalg.norm(x_vector - x_test))
        d_food_w_nat = float(np.sqrt(np.dot(W_S * (x_vector - x_test), (x_vector - x_test))))

        # S-MGIL tradeoff
        tradeoff = run_smgil_tradeoff(
            A, b, X_obs, W_S_tuned, item_index, meta,
            constraint_names=constraint_names,
            max_iterations=MAX_ITER,
            cost_threshold=COST_THRESHOLD,
            verbose=False,
        )

        rec = {
            "uid": uid,
            "n_items": len(item_index),
            "n_train_days": n_train_days,
            "n_test_days": n_test_days,
            "coverage_test": round(coverage, 4),
            "n_iter": len(tradeoff),
            "d_nut_natural": d_nut_nat,
            "d_food_natural": d_food_nat,
            "d_foodW_natural": d_food_w_nat,
        }
        for r in tradeoff:
            ell = r["iteration"]
            dists = compute_distances(x_vector, r["z"], W_S, A, b, n1, signs, scale)
            rec[f"d_nut_r{ell}"] = dists["d_nut"]
            rec[f"d_food_r{ell}"] = dists["d_food"]
            rec[f"d_foodW_r{ell}"] = dists["d_foodW"]
            rec[f"tight_r{ell}"] = ", ".join(r["tight_constraints"])

        records.append(rec)
        print(f"OK  (r={len(tradeoff)}, cov={coverage:.2f}, n={len(item_index)})")

    except Exception as e:
        print(f"FAILED: {e}")
        failed += 1

# -- Output ------------------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. {len(df)} succeeded, {failed} failed.")
if not df.empty:
    print(df.describe().round(4))
