"""MyFitnessPal dietary logs → S-MGIL observation vectors.

Operates in servings space (not grams): x_vector[i] = average daily servings
of food i across training dates. USDA neighbor items use 1 serving = 100g.

ID scheme: MFP observed items use key `MFP_ID_OFFSET + mfp_json_id` in
item_index. USDA neighbor items use their raw `food_id` as key. This avoids
collisions between MFP JSON IDs (0–631K) and USDA food_ids (0–7K).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


MFP_ID_OFFSET = 1_000_000


# ---------------------------------------------------------------------------
# Crosswalk builder  (MFP food_name → similarity-JSON mfp_json_id)
# ---------------------------------------------------------------------------

def build_mfp_crosswalk(
    mfp_similarity_json: str | Path,
    verbose: bool = True,
) -> tuple[dict[str, int], dict[int, dict]]:
    """Build MFP food name → JSON ID mapping and similarity index.

    Parameters
    ----------
    mfp_similarity_json : Path to myfitnesspal_to_usda.json (~500MB).

    Returns
    -------
    name_to_id       : {food_name.strip() → mfp_json_id}
    similarity_index : {mfp_json_id → {"name": str, "neighbors": [...]}}
    """
    path = Path(mfp_similarity_json)
    if not path.exists():
        raise FileNotFoundError(f"MFP similarity JSON not found: {path}")

    if verbose:
        print(f"[build_mfp_crosswalk] Loading {path.name} ...")

    with open(path) as f:
        raw = json.load(f)

    name_to_id: dict[str, int] = {}
    similarity_index: dict[int, dict] = {}

    for key, value in raw.items():
        mfp_id = int(key)
        name = value["name"].strip()
        neighbors = value.get("neighbors", [])
        # Sort neighbors by final_score descending
        neighbors = sorted(
            neighbors,
            key=lambda n: n.get("final_score", 0.0),
            reverse=True,
        )
        name_to_id[name] = mfp_id
        similarity_index[mfp_id] = {"name": name, "neighbors": neighbors}

    if verbose:
        print(
            f"[build_mfp_crosswalk] {len(name_to_id)} food names mapped, "
            f"{len(similarity_index)} entries with neighbors"
        )

    return name_to_id, similarity_index


# ---------------------------------------------------------------------------
# Observation vector builder
# ---------------------------------------------------------------------------

def build_mfp_observation_vector(
    user_df: pd.DataFrame,
    similarity_index: dict[int, dict],
    name_to_id: dict[str, int],
    user_id: int | str,
    train_dates: list[str] | None = None,
    K: int = 5,
    score_field: str = "final_score",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict]:
    """Build the S-MGIL observation vector for one MFP user.

    Parameters
    ----------
    user_df         : Pre-filtered DataFrame from myfitnesspal_foods.csv
                      (already filtered to this user, or will be filtered here).
    similarity_index: From build_mfp_crosswalk().
    name_to_id      : From build_mfp_crosswalk().
    user_id         : MFP user ID for filtering and metadata.
    train_dates     : If given, restrict to these dates only.
    K               : Number of USDA neighbors per observed food (max 5).
    score_field     : Similarity score field to use for ranking.

    Returns
    -------
    x_vector   : (n,) average daily servings (observed foods), 0 for neighbors
    W_S        : (n,) weight vector — 1 for observed, (1-sim) for neighbors
    item_index : {item_key -> column index}  (observed: MFP_ID_OFFSET + mfp_id,
                 neighbors: usda_food_id)
    metadata   : structured info about the augmented item space
    """
    # 1. Filter to user + train dates
    df = user_df[user_df["user_id"] == user_id].copy() if "user_id" in user_df.columns else user_df.copy()

    if train_dates is not None:
        df = df[df["date"].isin(train_dates)]

    if df.empty:
        raise ValueError(f"No data for user {user_id} (after date filter).")

    n_train_days = df["date"].nunique()

    # 2. Aggregate: count servings and compute mean nutrients per food_name
    # Each row in MFP is one serving of a food item
    nutrient_cols = ["calories", "carbs", "fat", "protein", "sodium", "sugar",
                     "fiber", "potass.", "iron", "calcium", "sat fat", "chol"]
    available_nutrient_cols = [c for c in nutrient_cols if c in df.columns]

    agg_dict = {"date": "count"}  # count = number of servings
    for col in available_nutrient_cols:
        agg_dict[col] = "mean"  # mean nutrient per serving

    food_agg = df.groupby("food_name").agg(agg_dict).reset_index()
    food_agg = food_agg.rename(columns={"date": "total_servings"})

    # 3. Look up MFP JSON IDs and build item space
    observed_items: list[dict] = []
    mfp_nutrients: dict[int, dict] = {}  # item_key → {nutrient: value_per_serving}
    missing_from_similarity: list[str] = []

    for _, row in food_agg.iterrows():
        food_name = str(row["food_name"]).strip()
        mfp_id = name_to_id.get(food_name)

        if mfp_id is None:
            missing_from_similarity.append(food_name)
            continue

        item_key = MFP_ID_OFFSET + mfp_id
        avg_daily_servings = row["total_servings"] / n_train_days

        # Store per-serving nutrient values
        nutrients_per_serving = {}
        for col in available_nutrient_cols:
            val = row[col]
            if pd.notna(val):
                nutrients_per_serving[col] = float(val)
        mfp_nutrients[item_key] = nutrients_per_serving

        observed_items.append({
            "food_code": item_key,
            "food_name": food_name,
            "mfp_json_id": mfp_id,
            "quantity": avg_daily_servings,
            "total_servings": int(row["total_servings"]),
        })

    if not observed_items:
        raise ValueError(f"No observed items could be matched for user {user_id}.")

    # 4. Retrieve USDA neighbors for each observed food
    observed_keys = {item["food_code"] for item in observed_items}
    claimed_neighbor_codes: set[int] = set()
    all_neighbor_dicts: list[dict] = []

    for item in observed_items:
        mfp_id = item["mfp_json_id"]
        entry = similarity_index.get(mfp_id)
        if entry is None or not entry.get("neighbors"):
            continue

        for nbr in entry["neighbors"]:
            nbr_food_id = int(nbr.get("food_id", -1))
            if nbr_food_id == -1 or nbr_food_id in claimed_neighbor_codes:
                continue

            score = float(nbr.get(score_field, nbr.get("final_score", 0.0)))
            all_neighbor_dicts.append({
                "food_code": nbr_food_id,  # raw USDA food_id
                "food_name": nbr.get("name", str(nbr_food_id)),
                "score": score,
                "parent_code": item["food_code"],
                "parent_name": item["food_name"],
            })
            claimed_neighbor_codes.add(nbr_food_id)

            if sum(1 for n in all_neighbor_dicts if n["parent_code"] == item["food_code"]) >= K:
                break

    # 5. Build ordered item list and index map
    ordered_keys: list[int] = [item["food_code"] for item in observed_items]
    for nbr in all_neighbor_dicts:
        if nbr["food_code"] not in observed_keys:
            ordered_keys.append(nbr["food_code"])

    seen: set[int] = set()
    unique_keys: list[int] = []
    for key in ordered_keys:
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)

    item_index: dict[int, int] = {key: idx for idx, key in enumerate(unique_keys)}
    n = len(unique_keys)

    # 6. Build x_vector and W_S
    x_vector = np.zeros(n, dtype=float)
    W_S = np.ones(n, dtype=float)

    for item in observed_items:
        if item["food_code"] in item_index:
            x_vector[item_index[item["food_code"]]] = item["quantity"]

    for nbr in all_neighbor_dicts:
        fc = nbr["food_code"]
        if fc in item_index and fc not in observed_keys:
            switching_cost = max(0.0, 1.0 - float(nbr["score"]))
            W_S[item_index[fc]] = switching_cost

    # 7. Build metadata
    observed_meta = []
    for item in observed_items:
        fc = item["food_code"]
        if fc in item_index:
            observed_meta.append({
                "food_code": fc,
                "food_name": item["food_name"],
                "quantity": item["quantity"],
                "unit": "servings",
                "index": item_index[fc],
            })

    neighbor_meta = []
    seen_nbr: set[int] = set()
    for nbr in all_neighbor_dicts:
        fc = nbr["food_code"]
        if fc in item_index and fc not in observed_keys and fc not in seen_nbr:
            seen_nbr.add(fc)
            neighbor_meta.append({
                "food_code": fc,
                "food_name": nbr["food_name"],
                "parent_code": nbr["parent_code"],
                "similarity_score": float(nbr["score"]),
                "switching_cost_W_S": float(W_S[item_index[fc]]),
                "index": item_index[fc],
            })

    metadata = {
        "respondent_id": user_id,
        "observed_items": observed_meta,
        "neighbor_items": neighbor_meta,
        "n_obs": len(observed_meta),
        "n_neighbors": len(neighbor_meta),
        "n_aug": n,
        "missing_from_similarity": missing_from_similarity,
        "mfp_nutrients": mfp_nutrients,
        "n_train_days": n_train_days,
        "unit": "servings",
        "quantity_col": "servings",
        "score_field": score_field,
        "K": K,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MFP User {user_id}:  {len(observed_meta)} distinct foods, "
              f"{n_train_days} training days")
        print(f"  Neighbors K     : {K}")
        print(f"{'=' * 60}")
        print(f"  Observed items  : {metadata['n_obs']}")
        print(f"  Neighbor items  : {metadata['n_neighbors']}")
        print(f"  Augmented size n: {n}")
        if missing_from_similarity:
            print(
                f"  ! Not in similarity JSON ({len(missing_from_similarity)}): "
                f"{missing_from_similarity[:5]}"
            )
        print(f"\n  Top observed foods (avg daily servings):")
        for item in sorted(observed_meta, key=lambda r: r["quantity"], reverse=True)[:5]:
            print(
                f"    [{item['index']:3d}] {item['food_name'][:45]:<45}  "
                f"{item['quantity']:6.2f} servings/day"
            )
        print(f"\n  Sample neighbors (highest similarity):")
        for nbr in sorted(neighbor_meta, key=lambda r: r["similarity_score"], reverse=True)[:5]:
            print(
                f"    [{nbr['index']:3d}] {nbr['food_name'][:35]:<35}  "
                f"sim={nbr['similarity_score']:.3f}  "
                f"cost={nbr['switching_cost_W_S']:.3f}"
            )
        print()

    return x_vector, W_S, item_index, metadata


# ---------------------------------------------------------------------------
# Daily observation matrix builder (multi-obs IO)
# ---------------------------------------------------------------------------

def build_mfp_daily_matrix(
    user_df: pd.DataFrame,
    similarity_index: dict[int, dict],
    name_to_id: dict[str, int],
    user_id: int | str,
    train_dates: list[str] | None = None,
    K: int = 5,
    score_field: str = "final_score",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict]:
    """Build per-day observation matrix for multi-observation S-MGIL.

    Uses the standard observation vector builder to define the item space
    (item_index, W_S), then constructs a (n_days, n) matrix where each row
    is one training day's servings vector.

    Parameters
    ----------
    Same as build_mfp_observation_vector().

    Returns
    -------
    X_daily    : (n_days, n) matrix — one row per training day (servings)
    x_vector   : (n,) mean across days (same as single-obs builder)
    W_S        : (n,) weight vector
    item_index : {item_key -> column index}
    metadata   : dict (includes 'daily_dates' listing the date per row)
    """
    # Build item space using the standard builder
    x_vector, W_S, item_index, meta = build_mfp_observation_vector(
        user_df=user_df,
        similarity_index=similarity_index,
        name_to_id=name_to_id,
        user_id=user_id,
        train_dates=train_dates,
        K=K,
        score_field=score_field,
        verbose=False,
    )

    # Filter to this user + training dates
    if "user_id" in user_df.columns:
        df = user_df[user_df["user_id"] == user_id].copy()
    else:
        df = user_df.copy()
    if train_dates is not None:
        df = df[df["date"].isin(train_dates)]

    dates = sorted(df["date"].unique())
    n = len(item_index)
    X_daily = np.zeros((len(dates), n))

    for day_idx, date in enumerate(dates):
        day_df = df[df["date"] == date]
        for food_name, group in day_df.groupby("food_name"):
            food_name_stripped = str(food_name).strip()
            mfp_id = name_to_id.get(food_name_stripped)
            if mfp_id is None:
                continue
            item_key = MFP_ID_OFFSET + mfp_id
            if item_key in item_index:
                X_daily[day_idx, item_index[item_key]] = len(group)

    meta["daily_dates"] = dates

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MFP User {user_id} — Multi-Observation Matrix")
        print(f"{'=' * 60}")
        print(f"  Training days   : {len(dates)}")
        print(f"  Item space n    : {n}")
        print(f"  X_daily shape   : {X_daily.shape}")
        items_per_day = (X_daily > 0).sum(axis=1)
        servings_per_day = X_daily.sum(axis=1)
        print(f"  Items/day       : {items_per_day.mean():.1f} "
              f"(min={items_per_day.min()}, max={items_per_day.max()})")
        print(f"  Servings/day    : {servings_per_day.mean():.1f} "
              f"(min={servings_per_day.min():.0f}, max={servings_per_day.max():.0f})")
        print()

    return X_daily, x_vector, W_S, item_index, meta
