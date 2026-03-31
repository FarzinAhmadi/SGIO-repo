"""NHANES dietary recall → S-MGIL observation vectors.

Given a respondent_sequence_number this module:
  1. Reads all food items eaten (Day 1 recall) from the NHANES CSV.
  2. Looks up each USDA food code in the food-similarity JSON and retrieves
     the top-K most similar items (neighbors) not already in the observed set.
  3. Builds the augmented item space  F_aug = F_obs ∪ ⋃ N_K(f_i).
  4. Returns:
       - x_vector  : observed items → quantity in grams, neighbor items → 0.0
       - W_S       : observed items → 1.0, neighbor items → 1 - similarity_score
       - item_index: food_code → column index in x_vector / W_S
       - metadata  : food names, neighbor relationships, similarity scores
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Crosswalk builder  (NHANES FNDDS code → similarity-JSON food_id)
# ---------------------------------------------------------------------------

def build_crosswalk(
    fndds_path: str | Path,
    usda_parquet_path: str | Path,
    verbose: bool = True,
) -> dict[int, int]:
    """Build a mapping  {nhanes_food_code → similarity_json_food_id}.

    The similarity JSON uses internal sequential IDs (0-N) as keys, not NHANES
    food codes.  This function bridges the gap via food name:

        NHANES code  --[long description]-->  parquet name  -->  parquet food_id

    Parameters
    ----------
    fndds_path : Path to the NHANES/FNDDS food description file.
        Accepts XPT/SAS (e.g. DRXFCD_J.XPT) or CSV with columns
        food_code+food_desc or DRXFDCD+DRXFCLD.

    usda_parquet_path : Parquet file with columns food_id (int) and name (str).
        food_id values correspond to the JSON keys in the similarity matrix.
    """
    fndds_path = Path(fndds_path)
    usda_parquet_path = Path(usda_parquet_path)

    # 1. Load the food description file
    if fndds_path.suffix.lower() in (".xpt", ".sas7bdat"):
        raw = pd.read_sas(fndds_path, encoding="latin1")
        raw.columns = raw.columns.str.strip()
        raw = raw.rename(columns={"DRXFDCD": "food_code", "DRXFCLD": "food_desc"})
        raw["food_code"] = raw["food_code"].astype(float).astype(int)
        raw["food_desc"] = raw["food_desc"].astype(str).str.strip()
    else:
        raw = pd.read_csv(fndds_path)
        raw.columns = raw.columns.str.strip()
        if "DRXFDCD" in raw.columns:
            raw = raw.rename(columns={"DRXFDCD": "food_code", "DRXFCLD": "food_desc"})
        elif "food_code" in raw.columns and "food_desc" in raw.columns:
            pass
        else:
            raise KeyError(
                f"Cannot find food-code/description columns in {fndds_path}. "
                f"Expected 'DRXFDCD'+'DRXFCLD' or 'food_code'+'food_desc'. "
                f"Found: {raw.columns.tolist()}"
            )
        raw["food_code"] = raw["food_code"].astype(float).astype(int)
        raw["food_desc"] = raw["food_desc"].astype(str).str.strip()

    # 2. Load parquet index
    parquet = pd.read_parquet(usda_parquet_path, columns=["food_id", "name"])
    parquet["name"] = parquet["name"].astype(str).str.strip()

    # 3. Join on exact name
    merged = raw[["food_code", "food_desc"]].merge(
        parquet, left_on="food_desc", right_on="name", how="left"
    )

    crosswalk: dict[int, int] = {}
    matched = 0
    unmatched_examples: list[str] = []

    for _, row in merged.iterrows():
        nhanes_code = int(row["food_code"])
        if pd.notna(row.get("food_id")):
            crosswalk[nhanes_code] = int(row["food_id"])
            matched += 1
        elif len(unmatched_examples) < 5:
            unmatched_examples.append(f'{nhanes_code} -> "{row["food_desc"]}"')

    total = len(merged)
    if verbose:
        print(
            f"[build_crosswalk] {matched}/{total} NHANES codes matched to "
            f"similarity-JSON food_id ({matched / total * 100:.1f}% match rate)"
        )
        if unmatched_examples:
            print(f"  Unmatched examples (first {len(unmatched_examples)}):")
            for ex in unmatched_examples:
                print(f"    {ex}")

    return crosswalk


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_similarity_index(json_path: str | Path) -> dict[int, list[dict]]:
    """Load the food-similarity JSON keyed by food code.

    Supports two layouts:
      A) flat list: [{"food_code": ..., "neighbors": [...]}, ...]
      B) dict keyed by food_code string: {"0": [...], ...} or {"0": {"neighbors": [...]}, ...}
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Similarity JSON not found: {json_path}")

    with open(json_path) as f:
        raw = json.load(f)

    index: dict[int, list[dict]] = {}

    if isinstance(raw, list):
        for entry in raw:
            code = int(entry["food_code"])
            neighbors = sorted(
                entry.get("neighbors", []),
                key=lambda n: n.get("reranker_score", n.get("final_score", 0.0)),
                reverse=True,
            )
            index[code] = neighbors
    elif isinstance(raw, dict):
        for key, value in raw.items():
            code = int(key)
            if isinstance(value, dict) and "neighbors" in value:
                neighbors = value["neighbors"]
            elif isinstance(value, list):
                neighbors = value
            else:
                continue
            neighbors = [n for n in neighbors if isinstance(n, dict)]
            index[code] = sorted(
                neighbors,
                key=lambda n: n.get("reranker_score", n.get("final_score", 0.0)),
                reverse=True,
            )
    else:
        raise ValueError("Unrecognized similarity JSON structure.")

    return index


def _get_top_k_neighbors(
    food_code: int,
    similarity_index: dict[int, list[dict]],
    exclude_codes: set,
    K: int,
    score_field: str,
) -> list[dict]:
    """Return up to K neighbors of `food_code` not in `exclude_codes`."""
    if food_code not in similarity_index:
        return []

    neighbors = []
    for nbr in similarity_index[food_code]:
        nbr_code = int(nbr.get("food_id", nbr.get("food_code", -1)))
        if nbr_code == -1 or nbr_code in exclude_codes:
            continue
        score = nbr.get(score_field, nbr.get("final_score", 0.0))
        neighbors.append(
            {
                "food_code": nbr_code,
                "food_name": nbr.get("name", nbr.get("food_name", str(nbr_code))),
                "score": score,
                "parent_code": food_code,
            }
        )
        if len(neighbors) >= K:
            break

    return neighbors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_observation_vector(
    nhanes_csv_path: str | Path,
    similarity_json: str | Path,
    respondent_id: int,
    K: int = 10,
    quantity_col: str = "grams",
    score_field: str = "reranker_score",
    crosswalk: dict[int, int] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict]:
    """Build the S-MGIL observation vector for one respondent.

    Returns
    -------
    x_vector   : (n,) observation vector — quantity for observed, 0 for neighbors
    W_S        : (n,) diagonal weight vector — 1 for observed, (1-sim) for neighbors
    item_index : {food_code -> column index}
    metadata   : structured info about the augmented item space
    """
    # 1. Load NHANES CSV and filter respondent
    df = pd.read_csv(nhanes_csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    id_col = "respondent_sequence_number"
    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found. Available: {list(df.columns[:10])} ...")

    respondent_df = df[df[id_col] == respondent_id].copy()
    if respondent_df.empty:
        raise ValueError(f"Respondent {respondent_id} not found in {nhanes_csv_path}.")

    usda_col = "usda_food_code"
    if usda_col not in respondent_df.columns:
        raise KeyError(f"Column '{usda_col}' not found in CSV.")

    qty_col_lower = quantity_col.lower().replace(" ", "_")
    if qty_col_lower not in respondent_df.columns:
        available = [c for c in respondent_df.columns if "gram" in c or "kcal" in c]
        raise KeyError(
            f"Quantity column '{quantity_col}' not found. "
            f"Available numeric columns include: {available}"
        )

    # 2. Aggregate quantities per food code
    agg = respondent_df.groupby(usda_col)[qty_col_lower].sum().reset_index()
    agg.columns = ["food_code", "quantity"]
    agg["food_code"] = agg["food_code"].astype(int)
    agg = agg[agg["quantity"] > 0].reset_index(drop=True)

    observed_codes = set(agg["food_code"].tolist())

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Respondent {respondent_id}:  {len(observed_codes)} distinct food items")
        print(f"  Quantity column : '{quantity_col}'")
        print(f"  Neighbors K     : {K}")
        print(f"{'=' * 60}")

    # 3. Load similarity index and retrieve neighbors
    similarity_index = _load_similarity_index(similarity_json)
    missing_from_similarity: list[int] = []
    all_neighbor_dicts: list[dict] = []
    claimed_neighbor_codes: set = set(observed_codes)

    for _, row in agg.iterrows():
        fc = int(row["food_code"])

        lookup_fc = fc
        if crosswalk is not None:
            mapped = crosswalk.get(fc)
            if mapped is not None:
                lookup_fc = mapped
            else:
                missing_from_similarity.append(fc)
                if verbose:
                    warnings.warn(
                        f"  NHANES code {fc} has no crosswalk entry -- "
                        f"no neighbors added for this item."
                    )
                continue

        if lookup_fc not in similarity_index:
            missing_from_similarity.append(fc)
            if verbose:
                warnings.warn(
                    f"  USDA code {fc} (-> similarity id {lookup_fc}) not found "
                    f"in similarity JSON -- no neighbors added for this item."
                )
            continue

        nbrs = _get_top_k_neighbors(
            food_code=lookup_fc,
            similarity_index=similarity_index,
            exclude_codes=claimed_neighbor_codes,
            K=K,
            score_field=score_field,
        )
        for nbr in nbrs:
            claimed_neighbor_codes.add(nbr["food_code"])
        all_neighbor_dicts.extend(nbrs)

    # 4. Build ordered item list and index map
    ordered_items: list[int] = agg["food_code"].tolist()
    for nbr in all_neighbor_dicts:
        if nbr["food_code"] not in observed_codes:
            ordered_items.append(nbr["food_code"])

    seen: set = set()
    unique_items: list[int] = []
    for code in ordered_items:
        if code not in seen:
            unique_items.append(code)
            seen.add(code)

    item_index: dict[int, int] = {code: idx for idx, code in enumerate(unique_items)}
    n = len(unique_items)

    # 5. Build x_vector and W_S
    x_vector = np.zeros(n, dtype=float)
    W_S = np.ones(n, dtype=float)

    for _, row in agg.iterrows():
        fc = int(row["food_code"])
        if fc in item_index:
            x_vector[item_index[fc]] = row["quantity"]

    for nbr in all_neighbor_dicts:
        fc = nbr["food_code"]
        if fc in item_index and fc not in observed_codes:
            switching_cost = max(0.0, 1.0 - float(nbr["score"]))
            W_S[item_index[fc]] = switching_cost

    # 6. Build metadata
    name_map: dict[int, str] = {}
    if "food_name" in respondent_df.columns or "description" in respondent_df.columns:
        name_col = "food_name" if "food_name" in respondent_df.columns else "description"
        for _, row in respondent_df.drop_duplicates(usda_col).iterrows():
            name_map[int(row[usda_col])] = str(row[name_col])

    observed_meta = []
    for _, row in agg.iterrows():
        fc = int(row["food_code"])
        observed_meta.append(
            {
                "food_code": fc,
                "food_name": name_map.get(fc, str(fc)),
                "quantity": float(row["quantity"]),
                "unit": quantity_col,
                "index": item_index[fc],
            }
        )

    neighbor_meta = []
    seen_nbr: set = set()
    for nbr in all_neighbor_dicts:
        fc = nbr["food_code"]
        if fc in item_index and fc not in observed_codes and fc not in seen_nbr:
            seen_nbr.add(fc)
            neighbor_meta.append(
                {
                    "food_code": fc,
                    "food_name": nbr["food_name"],
                    "parent_code": nbr["parent_code"],
                    "similarity_score": float(nbr["score"]),
                    "switching_cost_W_S": float(W_S[item_index[fc]]),
                    "index": item_index[fc],
                }
            )

    metadata = {
        "respondent_id": respondent_id,
        "observed_items": observed_meta,
        "neighbor_items": neighbor_meta,
        "n_obs": len(observed_codes),
        "n_neighbors": len(neighbor_meta),
        "n_aug": n,
        "missing_from_similarity": missing_from_similarity,
        "quantity_col": quantity_col,
        "score_field": score_field,
        "K": K,
    }

    if verbose:
        print(f"  Observed items  : {metadata['n_obs']}")
        print(f"  Neighbor items  : {metadata['n_neighbors']}")
        print(f"  Augmented size n: {n}")
        if missing_from_similarity:
            print(
                f"  ! Not in similarity JSON ({len(missing_from_similarity)}): "
                f"{missing_from_similarity}"
            )
        print(f"\n  Top observed foods:")
        for item in sorted(observed_meta, key=lambda r: r["quantity"], reverse=True)[:5]:
            print(
                f"    [{item['index']:3d}] {item['food_code']}  "
                f"{item['food_name'][:40]:<40}  "
                f"{item['quantity']:8.1f} {quantity_col}"
            )
        print(f"\n  Sample neighbors (highest similarity):")
        for nbr in sorted(neighbor_meta, key=lambda r: r["similarity_score"], reverse=True)[:5]:
            parent_name = name_map.get(nbr["parent_code"], str(nbr["parent_code"]))
            print(
                f"    [{nbr['index']:3d}] {nbr['food_code']}  "
                f"{nbr['food_name'][:35]:<35}  "
                f"sim={nbr['similarity_score']:.3f}  "
                f"cost={nbr['switching_cost_W_S']:.3f}  "
                f"<- {parent_name[:20]}"
            )
        print()

    return x_vector, W_S, item_index, metadata
