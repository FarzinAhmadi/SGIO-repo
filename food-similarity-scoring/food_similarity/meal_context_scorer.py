from __future__ import annotations

import logging

import numpy as np

from food_similarity.config import MealContextScoringConfig
from food_similarity.nhanes import NhanesStore
from food_similarity.pipeline import Candidate, Scorer
from food_similarity.search_request import SearchRequest
from food_similarity.signals import MACRO_FIELDS

logger = logging.getLogger(__name__)

# Standard macro split for meal target estimation (fraction of daily total)
_MACRO_DAILY_FRACTIONS: dict[str, float] = {
    "energy_kcal_100g": 1.0,
    "fat_100g": 0.30,       # 30% of kcal from fat
    "carbohydrates_100g": 0.50,  # 50% of kcal from carbs
    "proteins_100g": 0.20,  # 20% of kcal from protein
    "sugars_100g": 0.10,    # ~10% of kcal from sugars
    "fiber_100g": 0.014,    # ~28g/day
    "salt_100g": 0.003,     # ~6g/day
    "saturated_fat_100g": 0.10,  # 10% of kcal from sat fat
}

# Conversion factors: kcal per gram for macro
_KCAL_PER_GRAM: dict[str, float] = {
    "fat_100g": 9.0,
    "carbohydrates_100g": 4.0,
    "proteins_100g": 4.0,
    "sugars_100g": 4.0,
    "saturated_fat_100g": 9.0,
}


class MealContextScorer(Scorer):
    def __init__(
        self,
        nhanes_store: NhanesStore,
        config: MealContextScoringConfig,
        macro_scales: dict[str, float],
    ) -> None:
        self._store = nhanes_store
        self._config = config
        self._macro_scales = macro_scales

    @property
    def name(self) -> str:
        return "meal_context"

    @property
    def weight(self) -> float:
        return self._config.weight

    def is_active(self, request: SearchRequest) -> bool:
        return bool(request.meal_food_codes)

    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]:
        # Accumulate macros from current meal items
        accumulated = self._accumulate_meal_macros(request.meal_food_codes)
        # Compute per-meal targets
        per_meal_target = self._compute_meal_target()
        # Collect categories already in meal
        meal_categories: dict[str, int] = {}
        for fc in request.meal_food_codes:
            for cat in self._store.get_food_categories(fc):
                meal_categories[cat] = meal_categories.get(cat, 0) + 1

        for candidate in candidates:
            sub_scores: list[tuple[float, float]] = []

            # Sub-score 1: Nutritional gap
            gap_score = self._nutritional_gap_score(
                candidate, accumulated, per_meal_target
            )
            sub_scores.append(
                (gap_score, self._config.nutritional_gap_weight)
            )

            # Sub-score 2: Category diversity
            diversity_score = self._diversity_score(candidate, meal_categories)
            sub_scores.append(
                (diversity_score, self._config.diversity_weight)
            )

            total_weight = sum(w for _, w in sub_scores)
            weighted_sum = sum(s * w for s, w in sub_scores)
            candidate.scores["meal_context"] = (
                weighted_sum / total_weight if total_weight > 0 else 0.5
            )

        return candidates

    def _accumulate_meal_macros(
        self, food_codes: list[int]
    ) -> dict[str, float]:
        # Sums per-100g nutrient values, implicitly assuming a 100g portion per
        # food item.  Meal targets in _compute_meal_target use the same unit
        # convention so the comparison in _nutritional_gap_score is consistent.
        accumulated: dict[str, float] = {f: 0.0 for f in MACRO_FIELDS}
        for fc in food_codes:
            macros = self._store.get_food_macros(fc)
            for field_name in MACRO_FIELDS:
                val = macros.get(field_name)
                if val is not None:
                    accumulated[field_name] += val
        return accumulated

    def _compute_meal_target(self) -> dict[str, float]:
        daily_kcal = self._config.default_daily_kcal
        meals = max(self._config.meals_per_day, 1)
        target: dict[str, float] = {}
        for field_name in MACRO_FIELDS:
            frac = _MACRO_DAILY_FRACTIONS.get(field_name, 0.0)
            if field_name == "energy_kcal_100g":
                target[field_name] = daily_kcal * frac / meals
            elif field_name in ("fiber_100g", "salt_100g"):
                # These are absolute amounts per day
                target[field_name] = daily_kcal * frac / meals
            elif field_name in _KCAL_PER_GRAM:
                # Convert kcal fraction to grams
                kcal_from_macro = daily_kcal * frac
                target[field_name] = kcal_from_macro / _KCAL_PER_GRAM[field_name] / meals
            else:
                target[field_name] = 0.0
        return target

    def _nutritional_gap_score(
        self,
        candidate: Candidate,
        accumulated: dict[str, float],
        per_meal_target: dict[str, float],
    ) -> float:
        parts: list[float] = []
        for field_name in MACRO_FIELDS:
            cand_val = candidate.macros.get(field_name)
            if cand_val is None:
                continue
            remaining = max(0.0, per_meal_target.get(field_name, 0.0) - accumulated.get(field_name, 0.0))
            scale = self._macro_scales.get(field_name, 1.0)
            distance = abs(cand_val - remaining) / max(scale, 1e-6)
            parts.append(1.0 / (1.0 + distance))
        return float(np.mean(parts)) if parts else 0.5

    def _diversity_score(
        self,
        candidate: Candidate,
        meal_categories: dict[str, int],
    ) -> float:
        if not candidate.categories:
            return 0.5
        scores: list[float] = []
        for cat in candidate.categories:
            count = meal_categories.get(cat, 0)
            if count == 0:
                scores.append(1.0)
            else:
                scores.append(1.0 / (1.0 + count))
        return float(np.mean(scores))
