"""Two-day holdout validation over a cohort of respondents.

For each participant: build Day 1 pipeline, run S-MGIL, project Day 2
intake, and compute nutrient/food-space distances.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from smgil import build_crosswalk, build_observation_vector, build_A_b
from smgil import run_smgil_tradeoff
from smgil.tradeoff import tune_weights
from smgil.validation import project_day2, compute_distances

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = Path("data")
NHANES_CSV = DATA_DIR / "nhanes/2017/cleaned/day1_interview.csv"
DAY2_CSV = DATA_DIR / "nhanes/2017/cleaned/day2_interview.csv"
SIMILARITY_JSON = DATA_DIR / "similarity/usda_refined.json"
FNDDS_PATH = DATA_DIR / "nhanes/2017/rawdata/DRXFCD_J.xpt"
USDA_PARQUET = DATA_DIR / "similarity/usda_index.parquet"
NUTRIENT_CSV = DATA_DIR / "usda/2017-2018/processed/nutrient_values.csv"
SIMILARITY_NPZ = DATA_DIR / "similarity/usda.npz"

K_NEIGHBORS = 10
QUANTITY_COL = "grams"
N_COHORT = 100
MAX_ITER = 4
COST_THRESHOLD = 5_000_000
RANDOM_SEED = 42

NUTRIENT_COLS_CSV = [
    "sodium_mg", "total_saturated_fatty_acids_gm", "total_sugars_gm",
    "cholesterol_mg", "total_fat_gm", "dietary_fiber_gm",
    "potassium_mg", "calcium_mg", "magnesium_mg", "protein_gm",
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "smgil_day2_validation.csv"

# ── Load shared data ──────────────────────────────────────────────
crosswalk = build_crosswalk(fndds_path=FNDDS_PATH, usda_parquet_path=USDA_PARQUET)
d1_full = pd.read_csv(NHANES_CSV)
d2_full = pd.read_csv(DAY2_CSV)
npz = np.load(SIMILARITY_NPZ)
sim_matrix = npz["similarity"]

# ── Select cohort ─────────────────────────────────────────────────
d1_ids = set(d1_full["respondent_sequence_number"].unique())
d2_ids = set(d2_full["respondent_sequence_number"].unique())
common = list(d1_ids & d2_ids)
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(common)
cohort = common[:N_COHORT]
print(f"Overlap: {len(d1_ids & d2_ids)} | Selected: {len(cohort)}")

# ── Main loop ─────────────────────────────────────────────────────
records = []
failed = 0

for i, pid in enumerate(cohort):
    print(f"[{i + 1:3d}/{len(cohort)}] {pid}", end=" ... ")

    try:
        x_vector, W_S, item_index, meta = build_observation_vector(
            nhanes_csv_path=NHANES_CSV,
            similarity_json=SIMILARITY_JSON,
            respondent_id=pid,
            K=K_NEIGHBORS,
            quantity_col=QUANTITY_COL,
            score_field="reranker_score",
            crosswalk=crosswalk,
            verbose=False,
        )
        A, b, constraint_names, directions = build_A_b(
            fndds_nutrient_csv=NUTRIENT_CSV,
            item_index=item_index,
            crosswalk=crosswalk,
            x_vector=x_vector,
            verbose=False,
        )

        W_S_tuned = tune_weights(W_S, meta)
        X_obs = x_vector.reshape(1, -1)

        signs = np.array([1 if d == "upper" else -1 for d in directions])
        scale = np.abs(b)
        n1 = signs * (A @ x_vector)

        # Day 2 projection
        d2_pid_df = d2_full[d2_full["respondent_sequence_number"] == pid]
        if d2_pid_df.empty:
            print("no Day 2 data")
            failed += 1
            continue

        x_day2, coverage = project_day2(d2_pid_df, item_index, crosswalk, sim_matrix)

        # Day 2 nutrients from CSV
        n2 = d2_pid_df[NUTRIENT_COLS_CSV].sum().values.astype(float)

        # Natural distances
        d_nut_nat = float(np.linalg.norm((n1 - n2) / scale))
        d_food_nat = float(np.linalg.norm(x_vector - x_day2))
        d_food_w_nat = float(np.sqrt(np.dot(W_S * (x_vector - x_day2), (x_vector - x_day2))))

        # S-MGIL tradeoff
        tradeoff = run_smgil_tradeoff(
            A, b, X_obs, W_S_tuned, item_index, meta,
            constraint_names=constraint_names,
            max_iterations=MAX_ITER,
            cost_threshold=COST_THRESHOLD,
            verbose=False,
        )

        rec = {
            "pid": pid,
            "n_items": len(item_index),
            "coverage_day2": round(coverage, 4),
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
        print(f"OK  (r={len(tradeoff)}, cov={coverage:.2f})")

    except Exception as e:
        print(f"FAILED: {e}")
        failed += 1

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. {len(df)} succeeded, {failed} failed.")
print(df.describe().round(4))
