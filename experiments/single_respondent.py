"""Single-respondent S-MGIL pipeline demo.

Reproduces the main notebook workflow: build observation vector,
construct DASH constraints, run S-MGIL tradeoff, and print report.
"""

from pathlib import Path

import numpy as np

from smgil import build_crosswalk, build_observation_vector, build_A_b
from smgil import run_smgil_tradeoff
from smgil.tradeoff import tune_weights
from smgil.reporting import print_full_report, summary_table

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = Path("data")
NHANES_CSV = DATA_DIR / "nhanes/2017/cleaned/day1_interview.csv"
SIMILARITY_JSON = DATA_DIR / "similarity/usda_refined.json"
FNDDS_PATH = DATA_DIR / "nhanes/2017/rawdata/DRXFCD_J.xpt"
USDA_PARQUET = DATA_DIR / "similarity/usda_index.parquet"
NUTRIENT_CSV = DATA_DIR / "usda/2017-2018/processed/nutrient_values.csv"

RESPONDENT_ID = 93704
K_NEIGHBORS = 10
QUANTITY_COL = "grams"
MAX_ITERATIONS = 4
COST_THRESHOLD = 5_000_000  # grams^2

# ── Build crosswalk & observation vector ───────────────────────────
crosswalk = build_crosswalk(
    fndds_path=FNDDS_PATH,
    usda_parquet_path=USDA_PARQUET,
)

x_vector, W_S, item_index, meta = build_observation_vector(
    nhanes_csv_path=NHANES_CSV,
    similarity_json=SIMILARITY_JSON,
    respondent_id=RESPONDENT_ID,
    K=K_NEIGHBORS,
    quantity_col=QUANTITY_COL,
    score_field="reranker_score",
    crosswalk=crosswalk,
    verbose=True,
)

# ── Build DASH constraints ─────────────────────────────────────────
A, b, constraint_names, directions = build_A_b(
    fndds_nutrient_csv=NUTRIENT_CSV,
    item_index=item_index,
    crosswalk=crosswalk,
    x_vector=x_vector,
    verbose=True,
)

# ── Reshape for single respondent ──────────────────────────────────
X_obs = x_vector.reshape(1, -1)
print(f"X_obs shape: {X_obs.shape}   (K=1 observation, n={X_obs.shape[1]} items)")
print(f"A shape    : {A.shape}")
assert X_obs.shape[1] == A.shape[1], "Dimension mismatch between X and A!"

# ── Tune similarity weights ────────────────────────────────────────
W_S_tuned = tune_weights(W_S, meta)

# ── Run S-MGIL tradeoff path ──────────────────────────────────────
tradeoff_path = run_smgil_tradeoff(
    A, b, X_obs, W_S_tuned, item_index, meta,
    constraint_names=constraint_names,
    max_iterations=MAX_ITERATIONS,
    cost_threshold=COST_THRESHOLD,
)

# ── Report ─────────────────────────────────────────────────────────
print_full_report(
    x_vector, A, b, item_index, meta,
    constraint_names, directions, tradeoff_path, RESPONDENT_ID,
)

# ── Export ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df_summary = summary_table(tradeoff_path)
output_path = OUTPUT_DIR / f"smgil_tradeoff_respondent_{RESPONDENT_ID}.csv"
df_summary.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")
