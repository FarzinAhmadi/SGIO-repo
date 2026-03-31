"""DASH constraint matrix (A, b) construction for S-MGIL.

Builds the constraint matrix A (m x n) and RHS vector b (m,) using the
FNDDS nutrient CSV as the authoritative nutrient source for ALL items in the
augmented space (observed foods AND neighbor foods).
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DASH constraint specification
# ---------------------------------------------------------------------------
# Each entry: (display_name, fndds_column, direction, bound)
#   direction='upper'  →  nutrient_intake <= bound
#   direction='lower'  →  nutrient_intake >= bound  (row negated so Az <= b)
# ---------------------------------------------------------------------------
DASH_CONSTRAINTS = [
    # Upper bounds
    ("Sodium (mg)", "sodium_mg", "upper", 2300.0),
    ("Sat. Fat (g)", "total_saturated_fatty_acids_gm", "upper", 22.0),
    ("Total Sugars (g)", "total_sugars_gm", "upper", 100.0),
    ("Cholesterol (mg)", "cholesterol_mg", "upper", 150.0),
    ("Total Fat (g)", "total_fat_gm", "upper", 65.0),
    # Lower bounds (row negated so form is Az <= b)
    ("Fiber (g)", "dietary_fiber_gm", "lower", 25.0),
    ("Potassium (mg)", "potassium_mg", "lower", 4700.0),
    ("Calcium (mg)", "calcium_mg", "lower", 1200.0),
    ("Magnesium (mg)", "magnesium_mg", "lower", 320.0),
    ("Protein (g)", "protein_gm", "lower", 46.0),
]

# MFP DASH constraints: column names match MFP CSV columns.
# Magnesium dropped (0% coverage in MFP data).
# Each entry: (display_name, mfp_column, direction, bound)
MFP_DASH_CONSTRAINTS = [
    ("Sodium (mg)",       "sodium",   "upper", 2300.0),
    ("Sat. Fat (g)",      "sat fat",  "upper",   22.0),
    ("Total Sugars (g)",  "sugar",    "upper",  100.0),
    ("Cholesterol (mg)",  "chol",     "upper",  150.0),
    ("Total Fat (g)",     "fat",      "upper",   65.0),
    ("Fiber (g)",         "fiber",    "lower",   25.0),
    ("Potassium (mg)",    "potass.",  "lower", 4700.0),
    ("Calcium (mg)",      "calcium",  "lower", 1200.0),
    # Magnesium dropped: 0% coverage in MFP
    ("Protein (g)",       "protein",  "lower",   46.0),
]

# Maps MFP nutrient column names to FNDDS column names for imputation
_MFP_TO_FNDDS_COL = {
    "sodium":   "sodium_mg",
    "sat fat":  "total_saturated_fatty_acids_gm",
    "sugar":    "total_sugars_gm",
    "chol":     "cholesterol_mg",
    "fat":      "total_fat_gm",
    "fiber":    "dietary_fiber_gm",
    "potass.":  "potassium_mg",
    "calcium":  "calcium_mg",
    "protein":  "protein_gm",
}


def _load_fndds_nutrients(fndds_nutrient_csv: str | Path) -> pd.DataFrame:
    """Load the FNDDS nutrient CSV (values per 100 g), indexed by food_code."""
    path = Path(fndds_nutrient_csv)
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 5:
                break
        except Exception:
            continue

    df.columns = [c.strip().replace("\n", " ") for c in df.columns]

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
    df = df.drop_duplicates(subset=code_col).set_index(code_col)
    return df


def build_A_b(
    fndds_nutrient_csv: str | Path,
    item_index: dict[int, int],
    crosswalk: dict[int, int],
    x_vector: np.ndarray,
    dash_constraints: list | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build the constraint matrix A (m x n) and RHS b (m,) for S-MGIL.

    Parameters
    ----------
    fndds_nutrient_csv : Path to FNDDS nutrient CSV with values per 100 g.
    item_index         : {food_code -> column_index} from build_observation_vector().
    crosswalk          : {nhanes_code -> internal_food_id} from build_crosswalk().
    x_vector           : (n,) observed grams per item (used only for diagnostics).
    dash_constraints   : Override DASH_CONSTRAINTS with custom list of
                         (name, fndds_col, direction, bound) tuples.

    Returns
    -------
    A                   : (m, n) constraint matrix — A[i,j] = nutrient_i per gram of food_j
    b                   : (m,) RHS bounds
    constraint_names    : display names for each row of A
    constraint_directions : "upper" or "lower" for each row of A
    """
    if dash_constraints is None:
        dash_constraints = DASH_CONSTRAINTS

    # 1. Load nutrient table (per 100 g)
    nutrient_df = _load_fndds_nutrients(fndds_nutrient_csv)

    # 2. Build inverse crosswalk: internal_food_id → NHANES code
    food_id_to_nhanes = {v: k for k, v in crosswalk.items()}

    # 3. Resolve every item in item_index to a NHANES code
    nhanes_codes_in_nutrient = set(nutrient_df.index.tolist())

    resolved: dict[int, int | None] = {}
    for item_key in item_index:
        if item_key in nhanes_codes_in_nutrient:
            resolved[item_key] = item_key
        elif item_key in food_id_to_nhanes:
            nhanes = food_id_to_nhanes[item_key]
            resolved[item_key] = nhanes if nhanes in nhanes_codes_in_nutrient else None
        else:
            resolved[item_key] = None

    # 4. Check required nutrient columns
    missing_cols = []
    for _name, col, _, _ in dash_constraints:
        col_norm = col.replace("\n", " ").strip()
        if col_norm not in nutrient_df.columns:
            matches = [c for c in nutrient_df.columns if col_norm.lower() in c.lower()]
            if not matches:
                missing_cols.append(col_norm)
    if missing_cols:
        raise KeyError(
            f"These nutrient columns not found in the CSV: {missing_cols}\n"
            f"Available columns: {nutrient_df.columns.tolist()}"
        )

    # 5. Build A and b
    m = len(dash_constraints)
    n = len(item_index)
    A = np.zeros((m, n))
    b = np.zeros(m)
    constraint_names = []
    constraint_directions = []
    missing_items: list[int] = []

    for row_idx, (name, col, direction, bound) in enumerate(dash_constraints):
        col_norm = col.replace("\n", " ").strip()
        if col_norm in nutrient_df.columns:
            use_col = col_norm
        else:
            candidates = [c for c in nutrient_df.columns if col_norm.lower() in c.lower()]
            use_col = candidates[0]

        sign = 1.0 if direction == "upper" else -1.0

        for item_key, col_idx in item_index.items():
            nhanes_code = resolved.get(item_key)
            if nhanes_code is None:
                if item_key not in missing_items:
                    missing_items.append(item_key)
                continue

            val_per_100g = float(nutrient_df.at[nhanes_code, use_col])
            A[row_idx, col_idx] = sign * val_per_100g / 100.0

        b[row_idx] = -bound if direction == "lower" else bound
        constraint_names.append(name)
        constraint_directions.append(direction)

    # 6. Diagnostics
    if verbose:
        n_resolved = sum(1 for v in resolved.values() if v is not None)
        n_unresolved = len(resolved) - n_resolved
        print(f"[build_A_b] Constraint matrix: A {A.shape},  b {b.shape}")
        print(f"  Items resolved to nutrient data : {n_resolved}/{n}")
        if n_unresolved:
            print(f"  ! Items with NO nutrient data   : {n_unresolved}")
            for k in missing_items[:10]:
                print(f"    item_key={k}  (col_idx={item_index[k]})")
            if len(missing_items) > 10:
                print(f"    ... and {len(missing_items) - 10} more")
        print(f"  Constraints ({m}):")
        for i, cname in enumerate(constraint_names):
            direction = dash_constraints[i][2]
            op = "<=" if direction == "upper" else ">="
            display_bound = dash_constraints[i][3]
            nnz = np.count_nonzero(A[i])
            print(
                f"    [{i:2d}] {cname:<22} {op} {display_bound:>7.1f}   "
                f"({nnz}/{n} non-zero columns)"
            )

    return A, b, constraint_names, constraint_directions


def check_observed_intake(
    A: np.ndarray,
    b: np.ndarray,
    x_vector: np.ndarray,
    constraint_names: list[str],
    dash_constraints: list | None = None,
) -> pd.DataFrame:
    """Report how observed intake x_vector scores against each DASH constraint."""
    if dash_constraints is None:
        dash_constraints = DASH_CONSTRAINTS

    intake = A @ x_vector

    rows = []
    for i, (name, _, direction, bound) in enumerate(dash_constraints):
        if direction == "upper":
            actual = intake[i]
            slack = bound - actual
            satisfied = actual <= bound
        else:
            actual = -intake[i]
            slack = actual - bound
            satisfied = actual >= bound

        rows.append(
            {
                "Constraint": name,
                "Direction": direction,
                "Observed": round(actual, 2),
                "Bound": bound,
                "Slack (+) / Violation (-)": round(slack, 2),
                "Satisfied": satisfied,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MFP constraint builder
# ---------------------------------------------------------------------------

def build_A_b_mfp(
    item_index: dict[int, int],
    mfp_nutrients: dict[int, dict],
    fndds_nutrient_csv: str | Path,
    nhanes_crosswalk: dict[int, int],
    x_vector: np.ndarray,
    similarity_index: dict[int, dict] | None = None,
    mfp_id_offset: int = 1_000_000,
    dash_constraints: list | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build constraint matrix A (m x n) and RHS b for MFP data.

    For observed MFP items (key >= mfp_id_offset):
        A[row, col] = sign * nutrient_per_serving (from mfp_nutrients).
        If nutrient is missing, impute from best USDA neighbor's FNDDS value.

    For USDA neighbor items (key < mfp_id_offset):
        A[row, col] = sign * fndds_nutrient_per_100g (1 serving = 100g).

    Parameters
    ----------
    item_index       : {item_key -> col_idx} from build_mfp_observation_vector().
    mfp_nutrients    : {item_key -> {mfp_col: value}} from metadata["mfp_nutrients"].
    fndds_nutrient_csv : Path to FNDDS nutrient CSV (per 100g).
    nhanes_crosswalk : {nhanes_code -> usda_food_id} from build_crosswalk().
    x_vector         : (n,) observation vector (for diagnostics only).
    similarity_index : {mfp_json_id -> {"name", "neighbors"}} for imputation.
    mfp_id_offset    : Offset used for MFP item keys.
    dash_constraints : Override with custom constraint list.

    Returns
    -------
    A, b, constraint_names, constraint_directions
    """
    if dash_constraints is None:
        dash_constraints = MFP_DASH_CONSTRAINTS

    # Load FNDDS nutrients for USDA neighbor items
    nutrient_df = _load_fndds_nutrients(fndds_nutrient_csv)

    # Build reverse crosswalk: usda_food_id → nhanes_code (for FNDDS lookup)
    food_id_to_nhanes = {v: k for k, v in nhanes_crosswalk.items()}
    nhanes_codes_in_nutrient = set(nutrient_df.index.tolist())

    def _fndds_lookup(usda_food_id: int, fndds_col: str) -> float | None:
        """Look up a FNDDS nutrient value for a USDA food_id."""
        nhanes_code = food_id_to_nhanes.get(usda_food_id)
        if nhanes_code is not None and nhanes_code in nhanes_codes_in_nutrient:
            return float(nutrient_df.at[nhanes_code, fndds_col])
        return None

    def _impute_from_neighbors(mfp_json_id: int, fndds_col: str) -> float | None:
        """Impute a nutrient by looking up the best USDA neighbor in FNDDS."""
        if similarity_index is None:
            return None
        entry = similarity_index.get(mfp_json_id)
        if entry is None:
            return None
        for nbr in entry.get("neighbors", []):
            usda_id = int(nbr.get("food_id", -1))
            if usda_id == -1:
                continue
            val = _fndds_lookup(usda_id, fndds_col)
            if val is not None:
                return val
        return None

    m = len(dash_constraints)
    n = len(item_index)
    A = np.zeros((m, n))
    b = np.zeros(m)
    constraint_names = []
    constraint_directions = []

    # Track imputation stats per constraint
    imputation_counts = []

    for row_idx, (name, mfp_col, direction, bound) in enumerate(dash_constraints):
        sign = 1.0 if direction == "upper" else -1.0
        fndds_col = _MFP_TO_FNDDS_COL.get(mfp_col)
        n_imputed = 0
        n_filled = 0

        for item_key, col_idx in item_index.items():
            if item_key >= mfp_id_offset:
                # Observed MFP item: use per-serving nutrient from MFP data
                nutrients = mfp_nutrients.get(item_key, {})
                val = nutrients.get(mfp_col)

                if val is not None and not np.isnan(val):
                    A[row_idx, col_idx] = sign * val
                    n_filled += 1
                elif fndds_col is not None:
                    # Impute: find best USDA neighbor via similarity_index,
                    # then look up its FNDDS value
                    mfp_json_id = item_key - mfp_id_offset
                    fndds_val = _impute_from_neighbors(mfp_json_id, fndds_col)
                    if fndds_val is not None:
                        A[row_idx, col_idx] = sign * fndds_val
                        n_imputed += 1
                        n_filled += 1
            else:
                # USDA neighbor item: use FNDDS nutrient per 100g directly
                if fndds_col is None:
                    continue
                val_per_100g = _fndds_lookup(item_key, fndds_col)
                if val_per_100g is not None:
                    A[row_idx, col_idx] = sign * val_per_100g
                    n_filled += 1

        b[row_idx] = -bound if direction == "lower" else bound
        constraint_names.append(name)
        constraint_directions.append(direction)
        imputation_counts.append((name, n_filled, n_imputed))

    if verbose:
        n_obs = sum(1 for k in item_index if k >= mfp_id_offset)
        n_nbr = n - n_obs
        print(f"[build_A_b_mfp] Constraint matrix: A {A.shape},  b {b.shape}")
        print(f"  Items: {n_obs} observed (MFP) + {n_nbr} neighbors (USDA)")
        print(f"  Constraints ({m}):")
        for i, (cname, n_filled, n_imputed) in enumerate(imputation_counts):
            direction = dash_constraints[i][2]
            op = "<=" if direction == "upper" else ">="
            display_bound = dash_constraints[i][3]
            nnz = np.count_nonzero(A[i])
            imp_str = f", {n_imputed} imputed from USDA" if n_imputed > 0 else ""
            print(
                f"    [{i:2d}] {cname:<22} {op} {display_bound:>7.1f}   "
                f"({nnz}/{n} non-zero{imp_str})"
            )

    return A, b, constraint_names, constraint_directions


# ---------------------------------------------------------------------------
# AI4FoodDB constraint builder
# ---------------------------------------------------------------------------

# A4F uses full DASH constraints since all nutrients come from FNDDS
A4F_DASH_CONSTRAINTS = DASH_CONSTRAINTS


def build_A_b_a4f(
    item_index: dict[int, int],
    similarity_index: dict[int, dict],
    fndds_nutrient_csv: str | Path,
    nhanes_crosswalk: dict[int, int],
    x_vector: np.ndarray,
    a4f_id_offset: int = 2_000_000,
    dash_constraints: list | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build constraint matrix A (m x n) and RHS b for AI4FoodDB data.

    All nutrient values come from FNDDS via the similarity mapping.
    For observed A4F items: best USDA neighbor's FNDDS value per 100g.
    For USDA neighbor items: FNDDS value per 100g directly.
    1 serving ≈ 100g for all items (same convention as MFP neighbors).

    Parameters
    ----------
    item_index       : {item_key -> col_idx} from build_a4f_observation_vector().
    similarity_index : {a4f_json_id -> {"name", "neighbors"}} from crosswalk.
    fndds_nutrient_csv : Path to FNDDS nutrient CSV (per 100g).
    nhanes_crosswalk : {nhanes_code -> usda_food_id} from build_crosswalk().
    x_vector         : (n,) observation vector (for diagnostics only).
    a4f_id_offset    : Offset used for A4F item keys.
    dash_constraints : Override with custom constraint list.

    Returns
    -------
    A, b, constraint_names, constraint_directions
    """
    if dash_constraints is None:
        dash_constraints = DASH_CONSTRAINTS

    nutrient_df = _load_fndds_nutrients(fndds_nutrient_csv)

    # Reverse crosswalk: usda_food_id → nhanes_code (for FNDDS lookup)
    food_id_to_nhanes = {v: k for k, v in nhanes_crosswalk.items()}
    nhanes_codes_in_nutrient = set(nutrient_df.index.tolist())

    def _fndds_lookup(usda_food_id: int, fndds_col: str) -> float | None:
        nhanes_code = food_id_to_nhanes.get(usda_food_id)
        if nhanes_code is not None and nhanes_code in nhanes_codes_in_nutrient:
            return float(nutrient_df.at[nhanes_code, fndds_col])
        return None

    def _resolve_a4f_item(a4f_json_id: int, fndds_col: str) -> float | None:
        """Look up FNDDS nutrient for an A4F item via its best USDA neighbor."""
        entry = similarity_index.get(a4f_json_id)
        if entry is None:
            return None
        for nbr in entry.get("neighbors", []):
            usda_id = int(nbr.get("food_id", -1))
            if usda_id == -1:
                continue
            val = _fndds_lookup(usda_id, fndds_col)
            if val is not None:
                return val
        return None

    # Validate required nutrient columns upfront
    missing_cols = []
    resolved_cols = {}
    for _name, col, _, _ in dash_constraints:
        col_norm = col.replace("\n", " ").strip()
        if col_norm in nutrient_df.columns:
            resolved_cols[col_norm] = col_norm
        else:
            matches = [c for c in nutrient_df.columns if col_norm.lower() in c.lower()]
            if matches:
                resolved_cols[col_norm] = matches[0]
            else:
                missing_cols.append(col_norm)
    if missing_cols:
        raise KeyError(
            f"These nutrient columns not found in the CSV: {missing_cols}\n"
            f"Available columns: {nutrient_df.columns.tolist()}"
        )

    m = len(dash_constraints)
    n = len(item_index)
    A = np.zeros((m, n))
    b = np.zeros(m)
    constraint_names = []
    constraint_directions = []
    fill_counts = []

    for row_idx, (name, col, direction, bound) in enumerate(dash_constraints):
        col_norm = col.replace("\n", " ").strip()
        use_col = resolved_cols[col_norm]

        sign = 1.0 if direction == "upper" else -1.0
        n_filled = 0

        for item_key, col_idx in item_index.items():
            if item_key >= a4f_id_offset:
                # Observed A4F item: resolve via similarity index
                a4f_json_id = item_key - a4f_id_offset
                val = _resolve_a4f_item(a4f_json_id, use_col)
                if val is not None:
                    # 1 serving ≈ 100g, so use nutrient per 100g directly
                    A[row_idx, col_idx] = sign * val
                    n_filled += 1
            else:
                # USDA neighbor item: lookup FNDDS directly
                val = _fndds_lookup(item_key, use_col)
                if val is not None:
                    A[row_idx, col_idx] = sign * val
                    n_filled += 1

        b[row_idx] = -bound if direction == "lower" else bound
        constraint_names.append(name)
        constraint_directions.append(direction)
        fill_counts.append(n_filled)

    if verbose:
        n_obs = sum(1 for k in item_index if k >= a4f_id_offset)
        n_nbr = n - n_obs
        print(f"[build_A_b_a4f] Constraint matrix: A {A.shape},  b {b.shape}")
        print(f"  Items: {n_obs} observed (A4F) + {n_nbr} neighbors (USDA)")
        print(f"  Constraints ({m}):")
        for i, cname in enumerate(constraint_names):
            direction = dash_constraints[i][2]
            op = "<=" if direction == "upper" else ">="
            display_bound = dash_constraints[i][3]
            nnz = np.count_nonzero(A[i])
            print(
                f"    [{i:2d}] {cname:<22} {op} {display_bound:>7.1f}   "
                f"({nnz}/{n} non-zero, {fill_counts[i]}/{n} resolved)"
            )

    return A, b, constraint_names, constraint_directions
