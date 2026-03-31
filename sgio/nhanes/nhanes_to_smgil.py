"""
nhanes_to_smgil.py
==================
Preprocessing pipeline: NHANES dietary recall CSV  →  S-MGIL observation vectors.

Given a respondent_sequence_number this module:
  1. Reads all food items eaten (Day 1 recall) from the NHANES CSV.
  2. Looks up each USDA food code in the food-similarity JSON and retrieves
     the top-K most similar items (neighbors) not already in the observed set.
  3. Builds the augmented item space  F_aug = F_obs ∪ ⋃ N_K(f_i).
  4. Returns:
       - x_vector  : np.ndarray of shape (|F_aug|,)
                     observed items  → quantity in grams (from NHANES)
                     neighbor items  → 0.0
       - W_S       : np.ndarray of shape (|F_aug|,)  – diagonal of weight matrix W_S
                     observed items  → 1.0
                     neighbor items  → 1 - reranker_score  (switching cost)
       - item_index: dict  usda_code → column index in x_vector / W_S
       - metadata  : dict with food names, neighbor relationships, similarity scores

Usage
-----
    from nhanes_to_smgil import build_crosswalk, build_observation_vector

    # Step 1: build crosswalk once (NHANES FNDDS code → similarity JSON food_id)
    crosswalk = build_crosswalk(
        fndds_path        = "DRXFCD_J.XPT",        # XPT or CSV with food codes + descriptions
        usda_parquet_path = "usda_index.parquet",   # food_id, name columns
    )

    # Step 2: build the augmented observation vector for one respondent
    x, W_S, item_index, meta = build_observation_vector(
        nhanes_csv_path  = "nhanes_dietary.csv",
        similarity_json  = "food_similarity.json",
        respondent_id    = 93704,
        K                = 10,
        quantity_col     = "grams",
        score_field      = "reranker_score",
        crosswalk        = crosswalk,
    )

    # x and W_S are ready to feed into IO / MGIL / S-MGIL
    X_matrix = x.reshape(1, -1)          # shape (K=1, n)
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Crosswalk builder  (NHANES FNDDS code → similarity-JSON food_id)
# ---------------------------------------------------------------------------

def build_crosswalk(
    fndds_path: str,
    usda_parquet_path: str,
    verbose: bool = True,
) -> Dict[int, int]:
    """
    Build a mapping  {nhanes_food_code (int) → similarity_json_food_id (int)}

    The similarity JSON uses internal sequential IDs (0–N) as keys — NOT NHANES
    food codes.  This function bridges the gap via food name:

        NHANES code  ──[long description]──▶  parquet name  ──▶  parquet food_id
        e.g. 11513600 ── "Chocolate milk, NFS" ──────────────▶  food_id 97

    Parameters
    ----------
    fndds_path : str
        Path to the NHANES/FNDDS food description file.  Two formats accepted:

        1. **XPT / SAS** (e.g. DRXFCD_J.XPT from CDC NHANES 2017-18):
           Columns:  DRXFDCD (food code),  DRXFCLD (long description)

        2. **CSV** with either:
           - Columns  food_code + food_desc   (your original FNDDS CSV), or
           - Columns  DRXFDCD  + DRXFCLD      (exported from XPT)

    usda_parquet_path : str
        Parquet file with columns  food_id (int, 0-N)  and  name (str).
        food_id values correspond directly to the JSON keys in the
        similarity matrix.

    verbose : bool
        Print match statistics.

    Returns
    -------
    crosswalk : dict  {nhanes_code (int) → food_id (int)}
        Only entries with a successful name match are included.
        Unmatched NHANES codes are omitted (they will get no neighbors).
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError("pyarrow is required to read the parquet: pip install pyarrow")

    # ── 1. Load the food description file ──────────────────────────────────
    path = Path(fndds_path)
    ext = path.suffix.lower()

    if ext in (".xpt", ".sas7bdat"):
        raw = pd.read_sas(fndds_path, encoding="latin1")
        raw.columns = raw.columns.str.strip()
        # Normalise to standard column names
        raw = raw.rename(columns={"DRXFDCD": "food_code", "DRXFCLD": "food_desc"})
        raw["food_code"] = raw["food_code"].astype(float).astype(int)
        raw["food_desc"] = raw["food_desc"].astype(str).str.strip()
    else:
        # CSV — tolerate both naming conventions
        raw = pd.read_csv(fndds_path)
        raw.columns = raw.columns.str.strip()
        if "DRXFDCD" in raw.columns:
            raw = raw.rename(columns={"DRXFDCD": "food_code", "DRXFCLD": "food_desc"})
        elif "food_code" in raw.columns and "food_desc" in raw.columns:
            pass  # already correct
        else:
            raise KeyError(
                f"Cannot find food-code/description columns in {fndds_path}. "
                f"Expected 'DRXFDCD'+'DRXFCLD' or 'food_code'+'food_desc'. "
                f"Found: {raw.columns.tolist()}"
            )
        raw["food_code"] = raw["food_code"].astype(float).astype(int)
        raw["food_desc"] = raw["food_desc"].astype(str).str.strip()

    # ── 2. Load parquet index ───────────────────────────────────────────────
    parquet = pd.read_parquet(usda_parquet_path, columns=["food_id", "name"])
    parquet["name"] = parquet["name"].astype(str).str.strip()

    # ── 3. Join on exact name ───────────────────────────────────────────────
    merged = raw[["food_code", "food_desc"]].merge(
        parquet, left_on="food_desc", right_on="name", how="left"
    )

    crosswalk: Dict[int, int] = {}
    matched = 0
    unmatched_examples: List[str] = []

    for _, row in merged.iterrows():
        nhanes_code = int(row["food_code"])
        if pd.notna(row.get("food_id")):
            crosswalk[nhanes_code] = int(row["food_id"])
            matched += 1
        elif len(unmatched_examples) < 5:
            unmatched_examples.append(f"{nhanes_code} → \"{row['food_desc']}\"")

    total = len(merged)
    if verbose:
        print(f"[build_crosswalk] {matched}/{total} NHANES codes matched to "
              f"similarity-JSON food_id ({matched/total*100:.1f}% match rate)")
        if unmatched_examples:
            print(f"  Unmatched examples (first {len(unmatched_examples)}):")
            for ex in unmatched_examples:
                print(f"    {ex}")

    return crosswalk


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_similarity_index(json_path: str) -> Dict[int, List[dict]]:
    """
    Load the food-similarity JSON and return a dict keyed by USDA food code
    (int) mapping to a list of neighbor dicts sorted by descending score.

    Expected JSON structure (two supported layouts):

    Layout A – flat list of entries:
    [
      {
        "food_code": 55100050,
        "food_name": "Pancakes, plain",
        "neighbors": [
          {"food_code": 55100060, "food_name": "...",
           "original_score": 0.91, "reranker_score": 0.87, "final_score": 0.89},
          ...
        ]
      },
      ...
    ]

    Layout B – dict keyed by food_code string:
    {
      "55100050": [
        {"food_code": 55100060, "original_score": 0.91,
         "reranker_score": 0.87, "final_score": 0.89},
        ...
      ],
      ...
    }
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Similarity JSON not found: {json_path}")

    with open(path, "r") as f:
        raw = json.load(f)

    index: Dict[int, List[dict]] = {}

    if isinstance(raw, list):
        # Layout A
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
            # Layout B-nested: { "0": { "name": "...", "neighbors": [...] } }
            if isinstance(value, dict) and "neighbors" in value:
                neighbors = value["neighbors"]
            # Layout B-flat: { "0": [ {neighbor_dict}, ... ] }
            elif isinstance(value, list):
                neighbors = value
            else:
                continue
            # Filter to only dict entries (guard against malformed data)
            neighbors = [n for n in neighbors if isinstance(n, dict)]
            sorted_nbrs = sorted(
                neighbors,
                key=lambda n: n.get("reranker_score", n.get("final_score", 0.0)),
                reverse=True,
            )
            index[code] = sorted_nbrs
    else:
        raise ValueError("Unrecognized similarity JSON structure.")

    return index


def _get_top_k_neighbors(
    food_code: int,
    similarity_index: Dict[int, List[dict]],
    exclude_codes: set,
    K: int,
    score_field: str,
) -> List[dict]:
    """
    Return up to K neighbors of `food_code` not in `exclude_codes`,
    sorted by descending `score_field`.
    """
    if food_code not in similarity_index:
        return []

    neighbors = []
    for nbr in similarity_index[food_code]:
        # JSON uses "food_id" and "name" — support both naming conventions
        nbr_code = int(nbr.get("food_id", nbr.get("food_code", -1)))
        if nbr_code == -1:
            continue
        if nbr_code in exclude_codes:
            continue
        score = nbr.get(score_field, nbr.get("final_score", 0.0))
        neighbors.append({
            "food_code": nbr_code,
            "food_name": nbr.get("name", nbr.get("food_name", str(nbr_code))),
            "score": score,
            "parent_code": food_code,
        })
        if len(neighbors) >= K:
            break

    return neighbors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_observation_vector(
    nhanes_csv_path: str,
    similarity_json: str,
    respondent_id: int,
    K: int = 10,
    quantity_col: str = "grams",
    score_field: str = "reranker_score",
    dietary_day: int = 1,                 # 1 = Day 1 recall (use all rows for respondent)
    normalize_quantity: bool = False,     # divide grams by max serving for that food code
    verbose: bool = True,
    crosswalk: Dict[int, int] = None,     # maps NHANES food code → similarity JSON food_id
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], dict]:
    """
    Build the S-MGIL observation vector for one respondent.

    Parameters
    ----------
    nhanes_csv_path : str
        Path to the NHANES dietary recall CSV (one row per food item eaten).
    similarity_json : str
        Path to the food-similarity JSON.
    respondent_id : int
        Value of `respondent_sequence_number` to filter.
    K : int
        Number of top similar neighbors to add per observed food item.
    quantity_col : str
        Column in the CSV to use as the quantity signal for observed items.
        Common choices: "grams", "energy_kcal", "protein_gm".
        Default "grams" means x[f_i] = total grams consumed of food f_i.
    score_field : str
        Similarity score to use as substitutability signal.
        "reranker_score" (recommended) or "final_score".
    dietary_day : int
        Placeholder for multi-day support; currently all rows for the
        respondent are used (NHANES Day 1 recall = one continuous block).
    normalize_quantity : bool
        If True, divide each observed quantity by the respondent's total
        energy intake, producing a fractional allocation vector.
    verbose : bool
        Print a summary of the constructed vector.

    Returns
    -------
    x_vector : np.ndarray, shape (n,)
        Observation vector.  x[i] = quantity for observed items, 0 for neighbors.
    W_S : np.ndarray, shape (n,)
        Diagonal of the similarity weight matrix W_S.
        W_S[i] = 1 for observed items, (1 - similarity_score) for neighbors.
    item_index : dict {food_code (int) → column index (int)}
        Maps each USDA food code to its position in x_vector / W_S.
    metadata : dict
        Structured information about the augmented item space:
        {
          "respondent_id": int,
          "observed_items": [{"food_code", "food_name", "quantity", "index"}, ...],
          "neighbor_items": [{"food_code", "food_name", "parent_code",
                              "score", "switching_cost", "index"}, ...],
          "n_obs": int,
          "n_aug": int,
          "missing_from_similarity": [food_code, ...],
        }
    """

    # ------------------------------------------------------------------
    # 1. Load NHANES CSV and filter respondent
    # ------------------------------------------------------------------
    df = pd.read_csv(nhanes_csv_path, low_memory=False)

    # Normalise column names (tolerate spaces / mixed case)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    id_col = "respondent_sequence_number"
    if id_col not in df.columns:
        raise KeyError(
            f"Column '{id_col}' not found. Available: {list(df.columns[:10])} ..."
        )

    respondent_df = df[df[id_col] == respondent_id].copy()
    if respondent_df.empty:
        raise ValueError(
            f"Respondent {respondent_id} not found in {nhanes_csv_path}."
        )

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

    # ------------------------------------------------------------------
    # 2. Aggregate quantities per food code (a respondent may eat the
    #    same USDA code at different meal occasions → sum grams)
    # ------------------------------------------------------------------
    agg = (
        respondent_df
        .groupby(usda_col)[qty_col_lower]
        .sum()
        .reset_index()
    )
    agg.columns = ["food_code", "quantity"]
    agg["food_code"] = agg["food_code"].astype(int)

    # Drop rows with 0 / NaN quantity (water-only entries, condiments)
    agg = agg[agg["quantity"] > 0].reset_index(drop=True)

    if normalize_quantity:
        total = agg["quantity"].sum()
        if total > 0:
            agg["quantity"] /= total

    observed_codes = set(agg["food_code"].tolist())

    if verbose:
        print(f"\n{'='*60}")
        print(f"Respondent {respondent_id}:  {len(observed_codes)} distinct food items")
        print(f"  Quantity column : '{quantity_col}'")
        print(f"  Neighbors K     : {K}")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 3. Load similarity index and retrieve neighbors
    # ------------------------------------------------------------------
    similarity_index = _load_similarity_index(similarity_json)
    missing_from_similarity: List[int] = []

    # Neighbor items: list of dicts
    all_neighbor_dicts: List[dict] = []
    # Track which codes are already claimed as neighbors to avoid duplicates
    claimed_neighbor_codes: set = set(observed_codes)  # observed codes are excluded too

    for _, row in agg.iterrows():
        fc = int(row["food_code"])

        # ── crosswalk: translate NHANES code → similarity JSON food_id ──
        lookup_fc = fc
        if crosswalk is not None:
            mapped = crosswalk.get(fc)
            if mapped is not None:
                lookup_fc = mapped
            else:
                missing_from_similarity.append(fc)
                if verbose:
                    warnings.warn(
                        f"  NHANES code {fc} has no crosswalk entry — "
                        f"no neighbors added for this item."
                    )
                continue

        if lookup_fc not in similarity_index:
            missing_from_similarity.append(fc)
            if verbose:
                warnings.warn(
                    f"  USDA code {fc} (→ similarity id {lookup_fc}) not found in similarity JSON — "
                    f"no neighbors added for this item."
                )
            continue

        nbrs = _get_top_k_neighbors(
            food_code=lookup_fc,          # use the translated JSON key, not the NHANES code
            similarity_index=similarity_index,
            exclude_codes=claimed_neighbor_codes,
            K=K,
            score_field=score_field,
        )
        for nbr in nbrs:
            claimed_neighbor_codes.add(nbr["food_code"])
        all_neighbor_dicts.extend(nbrs)

    # ------------------------------------------------------------------
    # 4. Build ordered item list and index map
    # ------------------------------------------------------------------
    # Observed items come first (preserves interpretability)
    ordered_items: List[int] = agg["food_code"].tolist()
    # Append neighbors not already in observed set (deduplication already done above)
    for nbr in all_neighbor_dicts:
        if nbr["food_code"] not in observed_codes:
            ordered_items.append(nbr["food_code"])

    # Deduplicate while preserving order (neighbors may overlap across parents)
    seen: set = set()
    unique_items: List[int] = []
    for code in ordered_items:
        if code not in seen:
            unique_items.append(code)
            seen.add(code)

    item_index: Dict[int, int] = {code: idx for idx, code in enumerate(unique_items)}
    n = len(unique_items)

    # ------------------------------------------------------------------
    # 5. Build x_vector and W_S
    # ------------------------------------------------------------------
    x_vector = np.zeros(n, dtype=float)
    W_S = np.ones(n, dtype=float)          # default weight = 1 (observed items)

    # Fill observed quantities
    for _, row in agg.iterrows():
        fc = int(row["food_code"])
        if fc in item_index:
            x_vector[item_index[fc]] = row["quantity"]
        # W_S[idx] remains 1.0 for observed items

    # Fill neighbor weights  (switching cost = 1 - similarity_score)
    for nbr in all_neighbor_dicts:
        fc = nbr["food_code"]
        if fc in item_index and fc not in observed_codes:
            switching_cost = max(0.0, 1.0 - float(nbr["score"]))
            W_S[item_index[fc]] = switching_cost
            # x_vector stays 0.0 for neighbor items ✓

    # ------------------------------------------------------------------
    # 6. Build metadata for interpretability
    # ------------------------------------------------------------------
    # Lookup food names from CSV for observed items
    name_map: Dict[int, str] = {}
    if "food_name" in respondent_df.columns or "description" in respondent_df.columns:
        name_col = "food_name" if "food_name" in respondent_df.columns else "description"
        for _, row in respondent_df.drop_duplicates(usda_col).iterrows():
            name_map[int(row[usda_col])] = str(row[name_col])

    observed_meta = []
    for _, row in agg.iterrows():
        fc = int(row["food_code"])
        observed_meta.append({
            "food_code": fc,
            "food_name": name_map.get(fc, str(fc)),
            "quantity": float(row["quantity"]),
            "unit": quantity_col,
            "index": item_index[fc],
        })

    neighbor_meta = []
    seen_nbr: set = set()
    for nbr in all_neighbor_dicts:
        fc = nbr["food_code"]
        if fc in item_index and fc not in observed_codes and fc not in seen_nbr:
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
            print(f"  ⚠ Not in similarity JSON ({len(missing_from_similarity)}): "
                  f"{missing_from_similarity}")
        print(f"\n  Top observed foods:")
        for item in sorted(observed_meta, key=lambda r: r["quantity"], reverse=True)[:5]:
            print(f"    [{item['index']:3d}] {item['food_code']}  "
                  f"{item['food_name'][:40]:<40}  "
                  f"{item['quantity']:8.1f} {quantity_col}")
        print(f"\n  Sample neighbors (highest similarity):")
        for nbr in sorted(neighbor_meta,
                          key=lambda r: r["similarity_score"], reverse=True)[:5]:
            parent_name = name_map.get(nbr["parent_code"], str(nbr["parent_code"]))
            print(f"    [{nbr['index']:3d}] {nbr['food_code']}  "
                  f"{nbr['food_name'][:35]:<35}  "
                  f"sim={nbr['similarity_score']:.3f}  "
                  f"cost={nbr['switching_cost_W_S']:.3f}  "
                  f"← {parent_name[:20]}")
        print()

    return x_vector, W_S, item_index, metadata


# ---------------------------------------------------------------------------
# Convenience: build X matrix for a list of respondents (multi-observation IL)
# ---------------------------------------------------------------------------

def build_common_observation_matrix(
    nhanes_csv_path: str,
    similarity_json: str,
    respondent_ids: List[int],
    K: int = 10,
    quantity_col: str = "grams",
    score_field: str = "reranker_score",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], List[dict]]:
    """
    Build a shared augmented item space across multiple respondents and
    return an X matrix of shape (len(respondent_ids), n_aug_shared).

    This is useful when running IL/MGIL on a *population* of respondents
    sharing a common feasible set (e.g. same DASH constraints) — the
    K-independence property means that only x_bar (centroid) enters the IL
    objective, so we can stack all respondents into one matrix.

    Note: The shared item space is the union of all respondents' augmented
    spaces.  Respondents who did not eat a given item get quantity = 0 for
    that item in their row.

    Returns
    -------
    X_matrix       : np.ndarray (K_respondents × n_aug_shared)
    W_S_avg        : np.ndarray (n_aug_shared,)  — average switching costs
    shared_index   : dict {food_code → column index}
    metadata_list  : list of per-respondent metadata dicts
    """
    # First pass: collect all augmented item spaces
    per_respondent = []
    for rid in respondent_ids:
        try:
            x_vec, w_s, idx, meta = build_observation_vector(
                nhanes_csv_path, similarity_json, rid,
                K=K, quantity_col=quantity_col,
                score_field=score_field, verbose=verbose,
            )
            per_respondent.append((rid, x_vec, w_s, idx, meta))
        except Exception as exc:
            warnings.warn(f"Skipping respondent {rid}: {exc}")

    # Build union item space
    all_codes: List[int] = []
    seen_codes: set = set()
    for _, _, _, idx, _ in per_respondent:
        for code in sorted(idx, key=lambda c: idx[c]):
            if code not in seen_codes:
                all_codes.append(code)
                seen_codes.add(code)

    shared_index: Dict[int, int] = {c: i for i, c in enumerate(all_codes)}
    n_shared = len(all_codes)
    K_resp = len(per_respondent)

    X_matrix = np.zeros((K_resp, n_shared), dtype=float)
    W_S_sum = np.ones(n_shared, dtype=float)   # initialise at 1; items not seen by
                                                 # anyone retain W_S = 1

    for row_idx, (rid, x_vec, w_s, idx, meta) in enumerate(per_respondent):
        for code, col in idx.items():
            shared_col = shared_index[code]
            X_matrix[row_idx, shared_col] = x_vec[col]
            # Average W_S across respondents who have this code
            W_S_sum[shared_col] = (W_S_sum[shared_col] + w_s[col]) / 2.0

    metadata_list = [meta for _, _, _, _, meta in per_respondent]

    if verbose:
        print(f"\nShared augmented space: {n_shared} items across "
              f"{K_resp} respondents")

    return X_matrix, W_S_sum, shared_index, metadata_list


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python nhanes_to_smgil.py  <nhanes_csv>  <similarity_json>"
            "  <respondent_id>  [K=10]  [quantity_col=grams]"
        )
        sys.exit(1)

    csv_path   = sys.argv[1]
    sim_json   = sys.argv[2]
    resp_id    = int(sys.argv[3])
    K_arg      = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    qty_arg    = sys.argv[5]      if len(sys.argv) > 5 else "grams"

    x, W, idx, meta = build_observation_vector(
        csv_path, sim_json, resp_id, K=K_arg, quantity_col=qty_arg,
    )

    print(f"\nx_vector  (first 10): {x[:10]}")
    print(f"W_S       (first 10): {W[:10]}")
    print(f"\nx_vector shape : {x.shape}")
    print(f"Non-zero entries (observed foods): {int((x > 0).sum())}")
    print(f"Zero entries (neighbor slots)    : {int((x == 0).sum())}")

    # Reshape for IL model input: X = x_vector as K=1 observation
    X_for_IL = x.reshape(1, -1)
    print(f"\nX_for_IL shape for IO/IL/MGIL: {X_for_IL.shape}")
    print("\nDone. x_vector and W_S are ready for S-MGIL.")
