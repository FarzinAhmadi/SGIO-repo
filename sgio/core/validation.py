"""Holdout validation for S-MGIL cohort experiments.

Supports both NHANES Day-2 holdout (project_day2) and MFP temporal-split
holdout (project_mfp_holdout).
"""

import numpy as np
import pandas as pd


def project_day2(
    d2_pid_df: pd.DataFrame,
    item_index: dict,
    crosswalk: dict,
    sim_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project Day 2 intake into the item_index space.

    Uses the similarity matrix to assign unmatched foods to their
    closest neighbour in the observation vector.

    Returns (x_day2, coverage) where coverage is the fraction of
    Day 2 grams that were matched.
    """
    # Build mapping from similarity-JSON internal IDs to item_index columns.
    # Items in item_index are either NHANES food codes (present in crosswalk)
    # or similarity-JSON internal IDs (neighbors, not in crosswalk).
    nhanes_codes = set(crosswalk.keys())
    json_idx_to_col = {}
    for k, col in item_index.items():
        if k in nhanes_codes:
            jidx = crosswalk.get(k)
            if jidx is not None:
                json_idx_to_col[jidx] = col
        else:
            # Already a similarity-JSON internal ID (neighbor item)
            json_idx_to_col[k] = col

    item_json_arr = np.array(list(json_idx_to_col.keys()))
    item_col_arr = np.array(list(json_idx_to_col.values()))

    n = len(item_index)
    x_day2 = np.zeros(n)
    total_g, matched_g = 0.0, 0.0

    for _, row in d2_pid_df.iterrows():
        nhanes_code = int(row["usda_food_code"])
        grams = float(row["grams"])
        total_g += grams
        json_idx = crosswalk.get(nhanes_code)

        if json_idx is None:
            continue

        if json_idx in json_idx_to_col:
            x_day2[json_idx_to_col[json_idx]] += grams
            matched_g += grams
        else:
            sim_row = sim_matrix[json_idx, item_json_arr].astype(float)
            best_pos = int(np.argmax(sim_row))
            best_col = item_col_arr[best_pos]
            best_sim = sim_row[best_pos]
            x_day2[best_col] += grams * best_sim
            matched_g += grams * best_sim

    coverage = matched_g / total_g if total_g > 0 else 0.0
    return x_day2, coverage


def compute_distances(
    x_vector: np.ndarray,
    z: np.ndarray,
    W_S: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    obs_nutrients: np.ndarray,
    signs: np.ndarray,
    scale: np.ndarray,
) -> dict:
    """Compute nutrient-space and food-space distances for a recommendation."""
    n_rec = signs * (A @ z)
    diff = x_vector - z
    return {
        "d_nut": float(np.linalg.norm((obs_nutrients - n_rec) / scale)),
        "d_food": float(np.linalg.norm(diff)),
        "d_foodW": float(np.sqrt(np.dot(W_S * diff, diff))),
    }


def project_mfp_holdout(
    test_df: pd.DataFrame,
    item_index: dict[int, int],
    name_to_id: dict[str, int],
    similarity_index: dict[int, dict],
    mfp_id_offset: int = 1_000_000,
) -> tuple[np.ndarray, float]:
    """Project MFP test-period intake into the training item_index space.

    For each food in test_df:
      - Look up mfp_json_id → item_key = mfp_id_offset + mfp_json_id
      - If item_key is in item_index: add 1/n_test_days servings
      - If not: find best USDA neighbor that IS in item_index, add weighted

    Parameters
    ----------
    test_df          : MFP food rows for the test period (one user).
    item_index       : {item_key -> col_idx} from training.
    name_to_id       : {food_name -> mfp_json_id} from crosswalk.
    similarity_index : {mfp_json_id -> {"name", "neighbors"}} from crosswalk.
    mfp_id_offset    : Offset for MFP item keys.

    Returns
    -------
    x_test   : (n,) projected test vector (avg daily servings)
    coverage : fraction of test servings that were matched
    """
    n = len(item_index)
    x_test = np.zeros(n)
    n_test_days = test_df["date"].nunique()
    if n_test_days == 0:
        return x_test, 0.0

    total_servings = 0
    matched_servings = 0.0

    for food_name, group in test_df.groupby("food_name"):
        servings = len(group)
        total_servings += servings
        daily_servings = servings / n_test_days

        food_name_stripped = str(food_name).strip()
        mfp_id = name_to_id.get(food_name_stripped)
        if mfp_id is None:
            continue

        item_key = mfp_id_offset + mfp_id

        if item_key in item_index:
            x_test[item_index[item_key]] += daily_servings
            matched_servings += servings
        else:
            # Try to find a USDA neighbor that IS in item_index
            entry = similarity_index.get(mfp_id)
            if entry is None:
                continue
            for nbr in entry.get("neighbors", []):
                nbr_id = int(nbr.get("food_id", -1))
                if nbr_id in item_index:
                    sim = float(nbr.get("final_score", 0.0))
                    x_test[item_index[nbr_id]] += daily_servings * sim
                    matched_servings += servings * sim
                    break

    coverage = matched_servings / total_servings if total_servings > 0 else 0.0
    return x_test, coverage


def project_a4f_holdout(
    test_df: pd.DataFrame,
    item_index: dict[int, int],
    name_to_id: dict[str, int],
    similarity_index: dict[int, dict],
    a4f_id_offset: int = 2_000_000,
) -> tuple[np.ndarray, float]:
    """Project AI4FoodDB test-period intake into the training item_index space.

    For each food in test_df:
      - Look up a4f_json_id → item_key = a4f_id_offset + a4f_json_id
      - If item_key is in item_index: add 1/n_test_days servings
      - If not: find best USDA neighbor that IS in item_index, add weighted

    Parameters
    ----------
    test_df          : A4F food rows for the test period (one user).
    item_index       : {item_key -> col_idx} from training.
    name_to_id       : {food_name -> a4f_json_id} from crosswalk.
    similarity_index : {a4f_json_id -> {"name", "neighbors"}} from crosswalk.
    a4f_id_offset    : Offset for A4F item keys.

    Returns
    -------
    x_test   : (n,) projected test vector (avg daily servings)
    coverage : fraction of test servings that were matched
    """
    n = len(item_index)
    x_test = np.zeros(n)
    n_test_days = test_df["date"].nunique()
    if n_test_days == 0:
        return x_test, 0.0

    total_servings = 0
    matched_servings = 0.0

    for food_name, group in test_df.groupby("food_name"):
        servings = len(group)
        total_servings += servings
        daily_servings = servings / n_test_days

        food_name_stripped = str(food_name).strip()
        a4f_id = name_to_id.get(food_name_stripped)
        if a4f_id is None:
            continue

        item_key = a4f_id_offset + a4f_id

        if item_key in item_index:
            x_test[item_index[item_key]] += daily_servings
            matched_servings += servings
        else:
            # Try to find a USDA neighbor that IS in item_index
            entry = similarity_index.get(a4f_id)
            if entry is None:
                continue
            for nbr in entry.get("neighbors", []):
                nbr_id = int(nbr.get("food_id", -1))
                if nbr_id in item_index:
                    sim = float(nbr.get("final_score", 0.0))
                    x_test[item_index[nbr_id]] += daily_servings * sim
                    matched_servings += servings * sim
                    break

    coverage = matched_servings / total_servings if total_servings > 0 else 0.0
    return x_test, coverage
