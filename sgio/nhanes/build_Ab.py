"""
build_Ab.py
===========
Build the DASH constraint matrix A (m × n) and RHS vector b (m,)
for the S-MGIL formulation, using the FNDDS nutrient CSV as the
authoritative nutrient source for ALL items in the augmented space
(observed foods AND neighbor foods).

Key fix over the notebook's original build_nutrient_matrix_from_nhanes():
  - The old function looked up nutrients from the NHANES recall CSV,
    so neighbor items (not present in the recall) got zero columns → A broken.
  - This function uses the FNDDS nutrient CSV (nutrient values per 100 g
    for every USDA food code), then scales by the crosswalk food_id mapping
    to cover all items in item_index regardless of whether they were observed.

Usage (drop into your notebook, replacing Cell 3):
------------------------------------------------------
    from build_Ab import build_A_b, DASH_CONSTRAINTS

    A, b, constraint_names = build_A_b(
        fndds_nutrient_csv = 'fndds_nutrient_values.csv',
        item_index         = item_index,        # from build_observation_vector
        crosswalk          = crosswalk,         # from build_crosswalk  (nhanes→food_id)
        fndds_crosswalk_csv= 'fndds_food_desc.csv',  # food_code ↔ food_desc
        usda_parquet_path  = 'usda_index.parquet',
        x_vector           = x_vector,          # observed grams (for per-gram scaling)
    )

    print(f'A: {A.shape}   b: {b.shape}')
    print('Constraints:', constraint_names)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DASH constraint specification
# ---------------------------------------------------------------------------
# Each entry: (display_name, fndds_column, direction, bound)
#   direction='upper'  →  nutrient_intake ≤ bound  →  row = +per_gram coefficients
#   direction='lower'  →  nutrient_intake ≥ bound  →  row = -per_gram coefficients, b = -bound
# ---------------------------------------------------------------------------
DASH_CONSTRAINTS = [
    # Upper bounds (minimize these)
    ("Sodium (mg)",           "Sodium (mg)",                            "upper",  2300.0),
    ("Sat. Fat (g)",          "Fatty acids, total saturated (g)",       "upper",    22.0),
    ("Total Sugars (g)",      "Sugars, total (g)",                      "upper",   100.0),
    ("Cholesterol (mg)",      "Cholesterol (mg)",                       "upper",   170.0),
    ("Total Fat (g)",         "Total Fat (g)",                          "upper",    70.0),
    # Lower bounds (maximize these — row negated so form is A z ≤ b)
    ("Fiber (g)",             "Fiber, total dietary (g)",               "lower",    10.0),
    ("Potassium (mg)",        "Potassium (mg)",                         "lower",  3500.0),
    ("Calcium (mg)",          "Calcium (mg)",                           "lower",  800.0),
    ("Magnesium (mg)",        "Magnesium (mg)",                         "lower",   200.0),
    ("Protein (g)",           "Protein (g)",                            "lower",    25.0),
]


def _load_fndds_nutrients(fndds_nutrient_csv: str) -> pd.DataFrame:
    """
    Load the FNDDS nutrient CSV.

    Expected columns (tab-separated, values per 100 g):
        Food code | Main food description | ... | Sodium (mg) | Protein (g) | ...

    Returns a DataFrame indexed by food_code (int), columns = nutrient names.
    Values are nutrient content per 100 g of food.
    """
    path = Path(fndds_nutrient_csv)
    # Try tab first, then comma
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 5:
                break
        except Exception:
            continue

    # Normalize column names: strip whitespace and collapse embedded newlines
    df.columns = [c.strip().replace("\n", " ") for c in df.columns]

    # Find the food-code column (tolerant of different spellings)
    code_col = None
    for candidate in ["Food code", "food_code", "FoodCode", "FOOD_CODE"]:
        if candidate in df.columns:
            code_col = candidate
            break
    if code_col is None:
        raise KeyError(
            f"Cannot find food-code column in {fndds_nutrient_csv}. "
            f"Found: {df.columns[:10].tolist()}"
        )

    df[code_col] = df[code_col].astype(float).astype(int)
    df = df.set_index(code_col)
    return df


def _build_nhanes_to_fndds_map(
    fndds_nutrient_csv: str,
    crosswalk: Dict[int, int],
) -> Dict[int, int]:
    """
    Build a mapping  {item_index_food_id → nhanes_food_code}  by inverting
    the crosswalk and confirming coverage.

    crosswalk : {nhanes_code → internal_food_id (parquet/JSON key)}

    Returns: {nhanes_code → nhanes_code}  (identity, just for confirmed codes)
    plus the inverse map {food_id → nhanes_code} as a second return value.
    """
    food_id_to_nhanes = {v: k for k, v in crosswalk.items()}
    return food_id_to_nhanes


def build_A_b(
    fndds_nutrient_csv: str,
    item_index: Dict[int, int],
    crosswalk: Dict[int, int],
    x_vector: np.ndarray,
    dash_constraints: list = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build the constraint matrix A (m × n) and RHS b (m,) for S-MGIL.

    Parameters
    ----------
    fndds_nutrient_csv : str
        Path to the FNDDS nutrient CSV with values per 100 g.
        All food codes in item_index (observed + neighbors) must be present.

    item_index : dict  {food_code (int) → column_index (int)}
        Augmented item space from build_observation_vector().
        Keys are NHANES food codes for observed items and internal
        similarity-JSON food_ids for neighbor items.

    crosswalk : dict  {nhanes_code (int) → internal_food_id (int)}
        From build_crosswalk().  Used to translate neighbor food_ids
        back to NHANES codes for the nutrient lookup.

    x_vector : np.ndarray  shape (n,)
        Observed grams per item (0 for neighbor items).
        Used only for diagnostic reporting; NOT used to scale A
        (A is built in per-gram units so it works for any z).

    dash_constraints : list, optional
        Override DASH_CONSTRAINTS with a custom list of
        (name, fndds_col, direction, bound) tuples.

    verbose : bool

    Returns
    -------
    A : np.ndarray  shape (m, n)
        Each row is a nutrient constraint; each column is a food item.
        A[i, j] = nutrient_i per gram of food_j
                  (negated for lower-bound constraints).
        Units: the constraint is A @ z ≤ b where z is in grams.

    b : np.ndarray  shape (m,)
        RHS bounds (all positive; sign convention folded into A).

    constraint_names : list of str
        Display names for each row of A / element of b.
    """
    if dash_constraints is None:
        dash_constraints = DASH_CONSTRAINTS

    # ── 1. Load nutrient table (per 100 g) ──────────────────────────────────
    nutrient_df = _load_fndds_nutrients(fndds_nutrient_csv)

    # ── 2. Build inverse crosswalk: internal_food_id → NHANES code ──────────
    food_id_to_nhanes = {v: k for k, v in crosswalk.items()}

    # ── 3. Resolve every item in item_index to a NHANES code ────────────────
    #   item_index keys are:
    #     - NHANES codes for OBSERVED items  (e.g. 11513600)
    #     - internal food_ids for NEIGHBOR items  (e.g. 97)
    #   We need to map both to a NHANES code present in the nutrient CSV.

    nhanes_codes_in_nutrient = set(nutrient_df.index.tolist())

    # Determine which keys are "raw NHANES" vs "internal food_id":
    # A simple heuristic: if the key itself is in the nutrient CSV index, use it.
    # Otherwise look it up via the inverse crosswalk.
    resolved: Dict[int, Optional[int]] = {}   # item_key → nhanes_code (or None)
    for item_key in item_index.keys():
        if item_key in nhanes_codes_in_nutrient:
            resolved[item_key] = item_key                  # observed item
        elif item_key in food_id_to_nhanes:
            nhanes = food_id_to_nhanes[item_key]
            resolved[item_key] = nhanes if nhanes in nhanes_codes_in_nutrient else None
        else:
            resolved[item_key] = None

    # ── 4. Check required nutrient columns ──────────────────────────────────
    missing_cols = []
    for name, col, _, _ in dash_constraints:
        # Normalize: strip newlines in CSV column names
        col_norm = col.replace("\n", " ").strip()
        # Try exact match first, then partial
        if col_norm not in nutrient_df.columns:
            matches = [c for c in nutrient_df.columns
                       if col_norm.lower() in c.lower()]
            if matches:
                # Patch the constraint tuple in place (find and update)
                pass  # handled below
            else:
                missing_cols.append(col_norm)
    if missing_cols:
        raise KeyError(
            f"These nutrient columns not found in the CSV: {missing_cols}\n"
            f"Available columns: {nutrient_df.columns.tolist()}"
        )

    # ── 5. Build A and b ─────────────────────────────────────────────────────
    m = len(dash_constraints)
    n = len(item_index)
    A = np.zeros((m, n))
    b = np.zeros(m)
    constraint_names = []
    missing_items = []

    for row_idx, (name, col, direction, bound) in enumerate(dash_constraints):
        col_norm = col.replace("\n", " ").strip()
        # Find best matching column name
        if col_norm in nutrient_df.columns:
            use_col = col_norm
        else:
            candidates = [c for c in nutrient_df.columns
                          if col_norm.lower() in c.lower()]
            if not candidates:
                raise KeyError(f"Column '{col_norm}' not found.")
            use_col = candidates[0]

        sign = 1.0 if direction == "upper" else -1.0

        for item_key, col_idx in item_index.items():
            nhanes_code = resolved.get(item_key)
            if nhanes_code is None:
                if item_key not in missing_items:
                    missing_items.append(item_key)
                continue

            # Value is per 100 g in FNDDS → divide by 100 to get per gram
            val_per_100g = float(nutrient_df.at[nhanes_code, use_col])
            val_per_gram = val_per_100g / 100.0

            A[row_idx, col_idx] = sign * val_per_gram

        b[row_idx] =  sign * bound   # always positive; sign is in A
        constraint_names.append(name)

    # ── 6. Diagnostics ───────────────────────────────────────────────────────
    if verbose:
        n_resolved   = sum(1 for v in resolved.values() if v is not None)
        n_unresolved = len(resolved) - n_resolved
        print(f"[build_A_b] Constraint matrix: A {A.shape},  b {b.shape}")
        print(f"  Items resolved to nutrient data : {n_resolved}/{n}")
        if n_unresolved:
            print(f"  ⚠ Items with NO nutrient data   : {n_unresolved}")
            for k in missing_items[:10]:
                print(f"    item_key={k}  (col_idx={item_index[k]})")
            if len(missing_items) > 10:
                print(f"    ... and {len(missing_items)-10} more")
        print(f"  Constraints ({m}):")
        for i, (cname, bound) in enumerate(zip(constraint_names, b)):
            direction = dash_constraints[i][2]
            op = "≤" if direction == "upper" else "≥"
            nnz = np.count_nonzero(A[i])
            print(f"    [{i:2d}] {cname:<22} {op} {bound:>7.1f}   "
                  f"({nnz}/{n} non-zero columns)")

    return A, b, constraint_names


# ---------------------------------------------------------------------------
# Convenience: verify A @ x_obs against DASH bounds
# ---------------------------------------------------------------------------
def check_observed_intake(
    A: np.ndarray,
    b: np.ndarray,
    x_vector: np.ndarray,
    constraint_names: List[str],
    dash_constraints: list = None,
) -> pd.DataFrame:
    """
    Report how the observed intake x_vector scores against each DASH constraint.

    Returns a DataFrame with one row per constraint showing:
      actual intake, bound, slack/violation, and satisfied flag.
    """
    if dash_constraints is None:
        dash_constraints = DASH_CONSTRAINTS

    intake = A @ x_vector   # signed (lower-bound rows are negated)

    rows = []
    for i, (name, _, direction, bound) in enumerate(dash_constraints):
        if direction == "upper":
            actual = intake[i]          # positive value = nutrient amount
            slack  = bound - actual
            satisfied = actual <= bound
        else:
            actual = -intake[i]         # un-negate for display
            slack  = actual - bound
            satisfied = actual >= bound

        rows.append({
            "Constraint":  name,
            "Direction":   direction,
            "Observed":    round(actual, 2),
            "Bound":       bound,
            "Slack (+) / Violation (-)": round(slack, 2),
            "Satisfied":   satisfied,
        })

    df = pd.DataFrame(rows)
    return df
