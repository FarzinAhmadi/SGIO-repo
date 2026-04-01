from __future__ import annotations

import logging

import numpy as np

from food_similarity.config import PreferenceScoringConfig
from food_similarity.dietary_rules import DIET_RULES
from food_similarity.nhanes import NhanesStore
from food_similarity.pipeline import Candidate, Scorer
from food_similarity.search_request import SearchRequest

logger = logging.getLogger(__name__)


class PreferenceScorer(Scorer):
    def __init__(
        self,
        nhanes_store: NhanesStore,
        config: PreferenceScoringConfig,
        macro_scales: dict[str, float],
    ) -> None:
        self._store = nhanes_store
        self._config = config
        self._macro_scales = macro_scales

    @property
    def name(self) -> str:
        return "preference"

    @property
    def weight(self) -> float:
        return self._config.weight

    def is_active(self, request: SearchRequest) -> bool:
        return request.user_id is not None

    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]:
        user = self._store.get_user(request.user_id)  # type: ignore[arg-type]
        if user is None:
            return candidates

        for candidate in candidates:
            sub_scores: list[tuple[float, float]] = []

            # Sub-score 1: Centroid similarity
            if user.centroid is not None and candidate.source == "usda":
                cand_emb = self._store.get_candidate_embedding(candidate.row_id)
                if cand_emb is not None:
                    cos_sim = float(np.dot(user.centroid, cand_emb) / (
                        np.linalg.norm(user.centroid) * np.linalg.norm(cand_emb) + 1e-8
                    ))
                    # Map [-1, 1] to [0, 1]
                    centroid_score = (cos_sim + 1.0) / 2.0
                    sub_scores.append(
                        (centroid_score, self._config.centroid_weight)
                    )

            # Sub-score 2: Category affinity
            if user.category_freqs and candidate.categories:
                max_freq = max(user.category_freqs.values()) if user.category_freqs else 1.0
                cat_score = 0.0
                for cat in candidate.categories:
                    freq = user.category_freqs.get(cat, 0.0)
                    cat_score = max(cat_score, freq / max_freq)
                sub_scores.append(
                    (cat_score, self._config.category_weight)
                )

            # Sub-score 3: Dietary alignment
            active_flags = [
                flag for flag, val in user.diet_flags.items() if val
            ]
            if active_flags:
                diet_score = self._compute_dietary_score(candidate, active_flags)
                sub_scores.append(
                    (diet_score, self._config.dietary_weight)
                )

            if sub_scores:
                total_weight = sum(w for _, w in sub_scores)
                weighted_sum = sum(s * w for s, w in sub_scores)
                candidate.scores["preference"] = (
                    weighted_sum / total_weight if total_weight > 0 else 0.5
                )
            else:
                candidate.scores["preference"] = 0.5

        return candidates

    def _compute_dietary_score(
        self, candidate: Candidate, active_flags: list[str]
    ) -> float:
        rule_scores: list[float] = []
        for flag in active_flags:
            rules = DIET_RULES.get(flag, [])
            for rule in rules:
                value = candidate.macros.get(rule.macro)
                if value is None:
                    continue
                scale = self._macro_scales.get(rule.macro, 1.0)
                if rule.direction == "low":
                    score = 1.0 / (1.0 + max(0.0, value - scale) / max(scale, 1e-6))
                else:  # "high"
                    score = min(1.0, value / max(scale, 1e-6))
                rule_scores.append(score)
        return float(np.mean(rule_scores)) if rule_scores else 0.5
