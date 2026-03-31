"""AI4FoodDB dietary photo logs → S-MGIL observation vectors.

Operates in servings space (not grams): x_vector[i] = average daily servings
of food i across training dates.  Each food label in a photo = 1 serving.
All nutrient data comes from USDA FNDDS via the similarity mapping.

ID scheme: A4F observed items use key ``A4F_ID_OFFSET + a4f_json_id`` in
item_index.  USDA neighbor items use their raw ``food_id`` as key.  This avoids
collisions between A4F JSON IDs and USDA food_ids (0–7K).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


A4F_ID_OFFSET = 2_000_000


# ---------------------------------------------------------------------------
# Crosswalk builder  (A4F food_name → similarity-JSON a4f_json_id)
# ---------------------------------------------------------------------------

def build_a4f_crosswalk(
	a4f_similarity_json: str | Path,
	verbose: bool = True,
) -> tuple[dict[str, int], dict[int, dict]]:
	"""Build A4F food name → JSON ID mapping and similarity index.

	Parameters
	----------
	a4f_similarity_json : Path to ai4fooddb_to_usda.json.

	Returns
	-------
	name_to_id       : {food_name (subcategory) → a4f_json_id}
	similarity_index : {a4f_json_id → {"name": str, "neighbors": [...]}}
	"""
	path = Path(a4f_similarity_json)
	if not path.exists():
		raise FileNotFoundError(f"A4F similarity JSON not found: {path}")

	if verbose:
		print(f"[build_a4f_crosswalk] Loading {path.name} ...")

	with open(path) as f:
		raw = json.load(f)

	name_to_id: dict[str, int] = {}
	similarity_index: dict[int, dict] = {}

	for key, value in raw.items():
		a4f_id = int(key)
		name = value["name"].strip()
		neighbors = value.get("neighbors", [])
		neighbors = sorted(
			neighbors,
			key=lambda n: n.get("final_score", 0.0),
			reverse=True,
		)
		name_to_id[name] = a4f_id
		similarity_index[a4f_id] = {"name": name, "neighbors": neighbors}

	if verbose:
		print(
			f"[build_a4f_crosswalk] {len(name_to_id)} food names mapped, "
			f"{len(similarity_index)} entries with neighbors"
		)

	return name_to_id, similarity_index


# ---------------------------------------------------------------------------
# Observation vector builder
# ---------------------------------------------------------------------------

def build_a4f_observation_vector(
	user_df: pd.DataFrame,
	similarity_index: dict[int, dict],
	name_to_id: dict[str, int],
	user_id: str,
	train_dates: list[str] | None = None,
	K: int = 5,
	score_field: str = "final_score",
	verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict]:
	"""Build the S-MGIL observation vector for one AI4FoodDB user.

	Parameters
	----------
	user_df         : DataFrame from ai4fooddb_foods.csv (all users or
	                  pre-filtered to this user).
	similarity_index: From build_a4f_crosswalk().
	name_to_id      : From build_a4f_crosswalk().
	user_id         : A4F user ID (e.g., "A4F_20573").
	train_dates     : If given, restrict to these dates only.
	K               : Number of USDA neighbors per observed food.
	score_field     : Similarity score field to use for ranking.

	Returns
	-------
	x_vector   : (n,) average daily servings (observed foods), 0 for neighbors
	W_S        : (n,) weight vector — 1 for observed, (1-sim) for neighbors
	item_index : {item_key -> column index}
	metadata   : structured info about the augmented item space
	"""
	# 1. Filter to user + train dates
	if "user_id" in user_df.columns:
		df = user_df[user_df["user_id"] == user_id].copy()
	else:
		df = user_df.copy()

	if train_dates is not None:
		df = df[df["date"].isin(train_dates)]

	if df.empty:
		raise ValueError(f"No data for user {user_id} (after date filter).")

	n_train_days = df["date"].nunique()

	# 2. Aggregate: count servings per food_name (each row = 1 serving)
	food_agg = df.groupby("food_name").agg(
		total_servings=("date", "count"),
	).reset_index()

	# 3. Look up A4F JSON IDs and build item space
	observed_items: list[dict] = []
	missing_from_similarity: list[str] = []

	for _, row in food_agg.iterrows():
		food_name = str(row["food_name"]).strip()
		a4f_id = name_to_id.get(food_name)

		if a4f_id is None:
			missing_from_similarity.append(food_name)
			continue

		item_key = A4F_ID_OFFSET + a4f_id
		avg_daily_servings = row["total_servings"] / n_train_days

		observed_items.append({
			"food_code": item_key,
			"food_name": food_name,
			"a4f_json_id": a4f_id,
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
		a4f_id = item["a4f_json_id"]
		entry = similarity_index.get(a4f_id)
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
				"parent_name": nbr["parent_name"],
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
		"n_train_days": n_train_days,
		"unit": "servings",
		"quantity_col": "servings",
		"score_field": score_field,
		"K": K,
	}

	if verbose:
		print(f"\n{'=' * 60}")
		print(f"A4F User {user_id}:  {len(observed_meta)} distinct foods, "
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

def build_a4f_daily_matrix(
	user_df: pd.DataFrame,
	similarity_index: dict[int, dict],
	name_to_id: dict[str, int],
	user_id: str,
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
	Same as build_a4f_observation_vector().

	Returns
	-------
	X_daily    : (n_days, n) matrix — one row per training day (servings)
	x_vector   : (n,) mean across days (same as single-obs builder)
	W_S        : (n,) weight vector
	item_index : {item_key -> column index}
	metadata   : dict (includes 'daily_dates' listing the date per row)
	"""
	# Build item space using the standard builder
	x_vector, W_S, item_index, meta = build_a4f_observation_vector(
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
			a4f_id = name_to_id.get(food_name_stripped)
			if a4f_id is None:
				continue
			item_key = A4F_ID_OFFSET + a4f_id
			if item_key in item_index:
				X_daily[day_idx, item_index[item_key]] = len(group)

	meta["daily_dates"] = dates

	if verbose:
		print(f"\n{'=' * 60}")
		print(f"A4F User {user_id} — Multi-Observation Matrix")
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
