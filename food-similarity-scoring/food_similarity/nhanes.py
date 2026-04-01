from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from food_similarity.config import AppConfig, NhanesConfig, UsdaSourceConfig
from food_similarity.dietary_rules import DIET_FLAG_NAMES

logger = logging.getLogger(__name__)

EATING_OCCASION_MAP: dict[int, str] = {
    1: "Breakfast",
    2: "Lunch",
    3: "Dinner",
    4: "Dinner",
    5: "Breakfast",
}
# 6-19 are all snacks
for _i in range(6, 20):
    EATING_OCCASION_MAP[_i] = "Snack"

USDA_NUTRIENT_MAP: dict[str, str] = {
    "energy_kcal_100g": "energy_kcal",
    "fat_100g": "total_fat_gm",
    "carbohydrates_100g": "carbohydrate_gm",
    "proteins_100g": "protein_gm",
    "sugars_100g": "total_sugars_gm",
    "fiber_100g": "dietary_fiber_gm",
    "saturated_fat_100g": "total_saturated_fatty_acids_gm",
}


@dataclass
class FoodLogEntry:
    food_code: int
    eating_occasion: int
    grams: float
    energy_kcal: float


@dataclass
class UserProfile:
    user_id: int
    gender: int
    age: int
    diet_flags: dict[str, bool]
    food_log: list[FoodLogEntry]
    category_freqs: dict[str, float] = field(default_factory=dict)
    centroid: np.ndarray | None = None


class NhanesStore:
    def __init__(self) -> None:
        self._users: dict[int, UserProfile] = {}
        self._food_code_to_row_id: dict[int, int] = {}
        self._food_code_to_name: dict[int, str] = {}
        self._food_code_to_category: dict[int, list[str]] = {}
        self._food_code_to_macros: dict[int, dict[str, float | None]] = {}
        self._embeddings: np.ndarray | None = None
        self._user_ids_sorted: list[int] = []

    def load(self, config: AppConfig, usda_config: UsdaSourceConfig) -> None:
        nhanes_cfg = config.nhanes
        self._load_usda_linkage(config, usda_config)
        self._load_nhanes_data(nhanes_cfg)
        self._compute_centroids()
        self._compute_category_freqs()
        logger.info(
            "NhanesStore ready: %d users, %d food_code linkages, embeddings %s",
            len(self._users),
            len(self._food_code_to_row_id),
            self._embeddings.shape if self._embeddings is not None else "None",
        )

    def _load_usda_linkage(
        self, config: AppConfig, usda_config: UsdaSourceConfig
    ) -> None:
        desc_df = pl.read_csv(usda_config.descriptions_path)
        nutr_df = pl.read_csv(usda_config.nutrients_path)
        joined = desc_df.join(nutr_df, on="food_code", how="left")
        joined = joined.filter(
            pl.col("food_desc").str.len_chars() >= usda_config.min_name_length
        )

        for row_id, row in enumerate(joined.iter_rows(named=True)):
            fc_int = int(row["food_code"])
            name = str(row["food_desc"])
            cat = row.get("category_desc")
            self._food_code_to_row_id[fc_int] = row_id
            self._food_code_to_name[fc_int] = name
            self._food_code_to_category[fc_int] = [str(cat)] if cat else []

            macros: dict[str, float | None] = {}
            for canonical, usda_col in USDA_NUTRIENT_MAP.items():
                val = row.get(usda_col)
                macros[canonical] = float(val) if val is not None else None
            sodium = row.get("sodium_mg")
            macros["salt_100g"] = (
                float(sodium) * 2.5 / 1000 if sodium is not None else None
            )
            self._food_code_to_macros[fc_int] = macros

        # Load embeddings from checkpoint
        index_dir = Path(config.index.path) / "usda" / config.index.checkpoint_dir
        chunk_files = sorted(index_dir.glob("chunk_*.npy"))
        if chunk_files:
            embeddings_list = [np.load(path) for path in chunk_files]
            self._embeddings = np.vstack(embeddings_list)
            logger.info(
                "Loaded USDA embeddings from %d chunks: %s",
                len(chunk_files),
                self._embeddings.shape,
            )
        else:
            logger.warning(
                "USDA embeddings not found in checkpoint directory %s", index_dir
            )

    def _load_nhanes_data(self, nhanes_cfg: NhanesConfig) -> None:
        subjects_df = pl.read_csv(nhanes_cfg.subjects_path)
        foods_df = pl.read_csv(nhanes_cfg.foods_path)

        # Build food log per user
        food_logs: dict[int, list[FoodLogEntry]] = {}
        for row in foods_df.iter_rows(named=True):
            uid = row["respondent_sequence_number"]
            if uid is None:
                continue
            uid = int(uid)
            fc = row.get("usda_food_code")
            if fc is None:
                continue
            fc = int(fc)
            occasion = row.get("name_of_eating_occasion")
            occasion = int(occasion) if occasion is not None else 0
            grams = float(row.get("grams") or 0)
            kcal = float(row.get("energy_kcal") or 0)
            food_logs.setdefault(uid, []).append(
                FoodLogEntry(
                    food_code=fc,
                    eating_occasion=occasion,
                    grams=grams,
                    energy_kcal=kcal,
                )
            )

        # Build user profiles
        for row in subjects_df.iter_rows(named=True):
            uid = row["respondent_sequence_number"]
            if uid is None:
                continue
            uid = int(uid)
            gender = int(row.get("gender") or 0)
            age = int(row.get("age_in_years_at_screening") or 0)
            diet_flags: dict[str, bool] = {}
            for flag in DIET_FLAG_NAMES:
                val = row.get(flag)
                diet_flags[flag] = bool(val) if val is not None else False

            self._users[uid] = UserProfile(
                user_id=uid,
                gender=gender,
                age=age,
                diet_flags=diet_flags,
                food_log=food_logs.get(uid, []),
            )

        self._user_ids_sorted = sorted(self._users.keys())
        logger.info("Loaded %d NHANES users, %d food log entries",
                     len(self._users), sum(len(v) for v in food_logs.values()))

    def _compute_centroids(self) -> None:
        if self._embeddings is None:
            return
        dim = self._embeddings.shape[1]
        computed = 0
        for user in self._users.values():
            vecs: list[np.ndarray] = []
            for entry in user.food_log:
                row_id = self._food_code_to_row_id.get(entry.food_code)
                if row_id is not None and row_id < len(self._embeddings):
                    vecs.append(self._embeddings[row_id].astype(np.float32))
            if vecs:
                centroid = np.mean(vecs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                user.centroid = centroid
                computed += 1
            else:
                user.centroid = np.zeros(dim, dtype=np.float32)
        logger.info("Computed centroids for %d / %d users", computed, len(self._users))

    def _compute_category_freqs(self) -> None:
        for user in self._users.values():
            cat_counts: dict[str, int] = {}
            total = 0
            for entry in user.food_log:
                cats = self._food_code_to_category.get(entry.food_code, [])
                for cat in cats:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                    total += 1
            if total > 0:
                user.category_freqs = {
                    cat: count / total for cat, count in cat_counts.items()
                }

    def get_user(self, user_id: int) -> UserProfile | None:
        return self._users.get(user_id)

    def list_users(
        self,
        *,
        gender: int | None = None,
        min_age: int | None = None,
        max_age: int | None = None,
        diet_flag: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> tuple[list[UserProfile], int]:
        filtered: list[UserProfile] = []
        for uid in self._user_ids_sorted:
            user = self._users[uid]
            if gender is not None and user.gender != gender:
                continue
            if min_age is not None and user.age < min_age:
                continue
            if max_age is not None and user.age > max_age:
                continue
            if diet_flag and not user.diet_flags.get(diet_flag, False):
                continue
            filtered.append(user)

        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        return filtered[start:end], total

    def get_food_macros(self, food_code: int) -> dict[str, float | None]:
        return self._food_code_to_macros.get(food_code, {})

    def get_food_categories(self, food_code: int) -> list[str]:
        return self._food_code_to_category.get(food_code, [])

    def get_food_name(self, food_code: int) -> str:
        return self._food_code_to_name.get(food_code, f"Unknown ({food_code})")

    def get_food_embedding(self, food_code: int) -> np.ndarray | None:
        if self._embeddings is None:
            return None
        row_id = self._food_code_to_row_id.get(food_code)
        if row_id is None or row_id >= len(self._embeddings):
            return None
        return self._embeddings[row_id]

    def get_user_meals(
        self, user_id: int
    ) -> dict[str, list[FoodLogEntry]]:
        user = self._users.get(user_id)
        if user is None:
            return {}
        meals: dict[str, list[FoodLogEntry]] = {}
        for entry in user.food_log:
            meal_name = EATING_OCCASION_MAP.get(entry.eating_occasion, "Snack")
            meals.setdefault(meal_name, []).append(entry)
        return meals

    def get_candidate_embedding(self, row_id: int) -> np.ndarray | None:
        if self._embeddings is None:
            return None
        if row_id < 0 or row_id >= len(self._embeddings):
            return None
        return self._embeddings[row_id].astype(np.float32)

    @property
    def user_count(self) -> int:
        return len(self._users)

    def search_foods(
        self, query: str, limit: int = 20
    ) -> list[dict[str, object]]:
        q_lower = query.strip().lower()
        if not q_lower:
            return []
        if not hasattr(self, "_food_search_index"):
            self._food_search_index: list[tuple[int, str, str]] = [
                (fc, name.lower(), name)
                for fc, name in self._food_code_to_name.items()
            ]
        results: list[dict[str, object]] = []
        for fc, name_lower, name_original in self._food_search_index:
            if q_lower in name_lower:
                results.append({"food_code": fc, "food_name": name_original})
                if len(results) >= limit:
                    break
        return results

    def food_code_known(self, food_code: int) -> bool:
        return food_code in self._food_code_to_name
