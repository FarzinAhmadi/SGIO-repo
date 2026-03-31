"""
build_nutrient_matrix.py
========================
Constructs the nutrient constraint matrix A and bounds vector b for S-MGIL,
resolving nutrient coefficients for BOTH observed NHANES food codes AND
neighbor items (internal similarity-index food_ids) via the FNDDS nutrient file.

The FNDDS nutrient file reports nutrients per 100g.
NHANES quantities are in grams → coefficient = nutrient_per_100g / 100.

Usage (notebook):
-----------------
    from build_nutrient_matrix import build_nutrient_matrix, DASH_CONSTRAINTS

    A, b, nutrient_names = build_nutrient_matrix(
        item_index          = item_index,       # from build_observation_vector
        fndds_nutrient_path = "fndds_nutrients.csv",
        crosswalk           = crosswalk,        # from build_crosswalk
        usda_parquet_path   = "usda_index.parquet",
    )
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DASH constraint specification
# ---------------------------------------------------------------------------
# Each entry: (fndds_column_name, daily_bound, is_upper_bound, display_name)
# Upper bound:  nutrient ≤ bound  →  stored as  +coeff·z ≤ b
# Lower bound:  nutrient ≥ bound  →  stored as  -coeff·z ≤ -b  (negated)
# Values for adult women 51+, 2000-2300 kcal/day DASH target.

DASH_CONSTRAINTS: List[Tuple[str, float, bool, str]] = [
    ("Sodium (mg)",                      2300.0, True,  "Sodium (mg)"),
    ("Total Fat (g)",                      65.0, True,  "Total Fat (g)"),
    ("Fatty acids, total saturated (g)",   22.0, True,  "Saturated Fat (g)"),
    ("Cholesterol (mg)",                  150.0, True,  "Cholesterol (mg)"),
    ("Sugars, total\n(g)",                 50.0, True,  "Total Sugars (g)"),
    ("Fiber, total dietary (g)",           25.0, False, "Fiber (g)"),
    ("Potassium (mg)",                   4700.0, False, "Potassium (mg)"),
    ("Protein (g)",                        46.0, False, "Protein (g)"),
    ("Calcium (mg)",                     1200.0, False, "Calcium (mg)"),
    ("Magnesium (mg)",                    320.0, False, "Magnesium (mg)"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_fndds_nutrients(path: str) -> pd.DataFrame:
    """Load FNDDS nutrient file (tab- or comma-separated). Returns df indexed by food_code (int)."""
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
            if df.shape[1] > 5:
                break
        except Exception:
            continue

    code_col = next(
        (c for c in df.columns if c.strip().lower() in ("food code", "food_code")), None
    )
    if code_col is None:
        raise KeyError(f"No food-code column found. Columns: {df.columns[:6].tolist()}")

    df[code_col] = pd.to_numeric(df[code_col], errors="coerce")
    df = df.dropna(subset=[code_col])
    df[code_col] = df[code_col].astype(int)
    return df.set_index(code_col)


def _load_parquet_names(path: str) -> pd.DataFrame:
    """Load parquet; returns df indexed by food_id with column 'name'."""
    try:
        import pyarrow  # noqa
    except ImportError:
        raise ImportError("pip install pyarrow")
    return pd.read_parquet(path, columns=["food_id", "name"]).set_index("food_id")


def _reverse_crosswalk(crosswalk: Dict[int, int]) -> Dict[int, int]:
    """Invert: internal_food_id → nhanes_code (first mapping wins)."""
    rev: Dict[int, int] = {}
    for nhanes, fid in crosswalk.items():
        if fid not in rev:
            rev[fid] = nhanes
    return rev


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_nutrient_matrix(
    item_index: Dict[int, int],
    fndds_nutrient_path: str,
    crosswalk: Dict[int, int],
    usda_parquet_path: str,
    dash_constraints: List[Tuple[str, float, bool, str]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build the constraint matrix A (m × n) and bounds vector b (m,) for S-MGIL.

    Resolves nutrient coefficients for every item in item_index using a
    three-strategy lookup:
      1. Direct: item_code is itself a NHANES/FNDDS code present in FNDDS file.
      2. Reverse crosswalk: item_code is an internal similarity-index food_id
         → map to NHANES code via reverse crosswalk → look up in FNDDS.
      3. Name fallback: look up item_code in parquet to get food name
         → map name to NHANES code → look up in FNDDS.

    Parameters
    ----------
    item_index : dict {food_code → column_index}
        From build_observation_vector. Keys are NHANES codes (observed) or
        internal IDs (neighbors).
    fndds_nutrient_path : str
        FNDDS nutrient CSV/TSV with nutrients per 100g, one row per food code.
    crosswalk : dict {nhanes_code → internal_food_id}
        From build_crosswalk.
    usda_parquet_path : str
        usda_index.parquet — for name-based fallback lookup of neighbors.
    dash_constraints : list of (fndds_col, bound, is_upper, name), optional
        Defaults to module-level DASH_CONSTRAINTS.
    verbose : bool

    Returns
    -------
    A : np.ndarray (m, n)   — constraint matrix in ≤ form: A @ z ≤ b
    b : np.ndarray (m,)     — right-hand side
    nutrient_names : list[str] — display name for each row
    """
    if dash_constraints is None:
        dash_constraints = DASH_CONSTRAINTS

    m = len(dash_constraints)
    n = len(item_index)

    # ── load data sources ───────────────────────────────────────────────────
    fndds   = _load_fndds_nutrients(fndds_nutrient_path)
    parquet = _load_parquet_names(usda_parquet_path)
    rev_cw  = _reverse_crosswalk(crosswalk)

    # name → nhanes_code map  (for strategy 3)
    name_to_nhanes: Dict[str, int] = {}
    for nhanes_code, fid in crosswalk.items():
        if fid in parquet.index:
            name = str(parquet.at[fid, "name"])
            name_to_nhanes.setdefault(name, nhanes_code)

    # ── initialise A and b ──────────────────────────────────────────────────
    A = np.zeros((m, n), dtype=float)
    b = np.array(
        [bound if is_upper else -bound
         for _, bound, is_upper, _ in dash_constraints],
        dtype=float,
    )
    nutrient_names = [display for _, _, _, display in dash_constraints]

    # Warn about missing nutrient columns once
    for ncol, _, _, display in dash_constraints:
        if ncol not in fndds.columns:
            warnings.warn(
                f"Nutrient column '{ncol}' not found in FNDDS — "
                f"'{display}' will be all-zero."
            )

    # ── fill A column by column ─────────────────────────────────────────────
    missing: List[int] = []
    resolved_via: Dict[int, str] = {}

    for item_code, col_idx in item_index.items():
        nhanes_lookup = None

        # Strategy 1: direct NHANES code
        if item_code in fndds.index:
            nhanes_lookup = item_code
            resolved_via[item_code] = "direct"

        # Strategy 2: internal food_id → reverse crosswalk
        elif item_code in rev_cw and rev_cw[item_code] in fndds.index:
            nhanes_lookup = rev_cw[item_code]
            resolved_via[item_code] = f"rev_cw→{nhanes_lookup}"

        # Strategy 3: internal food_id → parquet name → nhanes code
        elif item_code in parquet.index:
            name = str(parquet.at[item_code, "name"])
            mapped = name_to_nhanes.get(name)
            if mapped is not None and mapped in fndds.index:
                nhanes_lookup = mapped
                resolved_via[item_code] = f"name→{nhanes_lookup}"

        if nhanes_lookup is None:
            missing.append(item_code)
            continue

        for row_idx, (ncol, _, is_upper, _) in enumerate(dash_constraints):
            if ncol not in fndds.columns:
                continue
            raw = fndds.at[nhanes_lookup, ncol]
            coeff = float(raw) / 100.0 if pd.notna(raw) else 0.0
            # Upper bound row: +coeff; lower bound row (negated): -coeff
            A[row_idx, col_idx] = coeff if is_upper else -coeff

    # ── report ──────────────────────────────────────────────────────────────
    if verbose:
        resolved = n - len(missing)
        print(f"[build_nutrient_matrix]  {resolved}/{n} items resolved "
              f"({resolved/n*100:.1f}%)")
        if missing:
            print(f"  ⚠  {len(missing)} items unresolved (A column = 0): "
                  f"{missing[:10]}" + (" ..." if len(missing) > 10 else ""))
        print(f"  A shape : ({m} constraints × {n} items)")
        print(f"  Constraints:")
        for i, (_, bound, is_upper, name) in enumerate(dash_constraints):
            print(f"    [{i:2d}] {name:<28s}  {'≤' if is_upper else '≥'} {bound}")

    return A, b, nutrient_names


# ---------------------------------------------------------------------------
# Convenience: DASH compliance report for any solution vector z
# ---------------------------------------------------------------------------

def check_dash_compliance(
    z: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    nutrient_names: List[str],
    dash_constraints: List[Tuple[str, float, bool, str]] = None,
    label: str = "Solution",
) -> pd.DataFrame:
    """
    Compute actual nutrient totals from solution z and report DASH satisfaction.

    Returns a DataFrame with columns:
        Constraint | Direction | Bound | Actual | Gap | Satisfied
    """
    if dash_constraints is None:
        dash_constraints = DASH_CONSTRAINTS

    rows = []
    for i, (_, bound, is_upper, name) in enumerate(dash_constraints):
        # Recover actual nutrient total from stored (possibly negated) row
        actual = float(A[i] @ z) if is_upper else float(-A[i] @ z)
        satisfied = (actual <= bound) if is_upper else (actual >= bound)
        gap = bound - actual if is_upper else actual - bound
        rows.append({
            "Constraint": name,
            "Direction": "≤" if is_upper else "≥",
            "Bound": bound,
            "Actual": round(actual, 2),
            "Gap": round(gap, 2),
            "Satisfied": satisfied,
        })

    df = pd.DataFrame(rows)
    if label:
        n_sat = df["Satisfied"].sum()
        print(f"\n── DASH Compliance: {label}  ({n_sat}/{len(df)} satisfied) ──")
        print(df.to_string(index=False))
    return df
