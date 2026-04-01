from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from food_similarity.signals import (
    LIST_METADATA_FIELDS,
    MACRO_FIELDS,
    SCALAR_METADATA_FIELDS,
)


@dataclass
class SearchRequest:
    query: str
    macro_targets: dict[str, float] = field(default_factory=dict)
    metadata_tags: dict[str, set[str]] = field(default_factory=dict)
    metadata_scalars: dict[str, str | int] = field(default_factory=dict)
    source_filter: set[str] = field(default_factory=set)
    user_id: int | None = None
    meal_food_codes: list[int] = field(default_factory=list)
    weight_overrides: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_query(cls, query: str) -> SearchRequest:
        return cls(query=query.strip())

    def has_macro_targets(self) -> bool:
        return bool(self.macro_targets)

    def has_metadata_preferences(self) -> bool:
        return bool(self.metadata_tags or self.metadata_scalars)

    def applied_criteria(self) -> dict[str, object]:
        out: dict[str, object] = {}
        if self.macro_targets:
            out["macro_targets"] = self.macro_targets
        if self.metadata_tags:
            out["metadata_tags"] = {
                k: sorted(v) for k, v in self.metadata_tags.items()
            }
        if self.metadata_scalars:
            out["metadata_scalars"] = self.metadata_scalars
        if self.source_filter:
            out["source_filter"] = sorted(self.source_filter)
        if self.user_id is not None:
            out["user_id"] = self.user_id
        if self.meal_food_codes:
            out["meal_food_codes"] = self.meal_food_codes
        if self.weight_overrides:
            out["weight_overrides"] = self.weight_overrides
        return out


WEIGHT_PARAM_NAMES: tuple[str, ...] = (
    "embedding",
    "reranker",
    "macronutrient",
    "metadata",
    "edit_distance",
)


def _normalize_tag(value: str) -> str:
    tag = re.sub(r"^[a-z]{2}:", "", value.strip().lower())
    return tag.replace("-", " ")


def _parse_tag_list(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {_normalize_tag(part) for part in raw.split(",") if _normalize_tag(part)}


def _parse_optional_float(raw: str, field_name: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for '{field_name}': '{raw}'") from exc
    if not math.isfinite(value):
        raise ValueError(f"Invalid numeric value for '{field_name}': '{raw}'")
    return value


def _parse_optional_int(raw: str, field_name: str) -> int | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for '{field_name}': '{raw}'") from exc


def parse_search_request_params(
    getter: Callable[[str], str | None],
) -> tuple[SearchRequest, dict[str, str]]:
    query = (getter("q") or "").strip()
    raw_inputs = {field: (getter(field) or "").strip() for field in MACRO_FIELDS}
    raw_inputs.update(
        {field: (getter(field) or "").strip() for field in LIST_METADATA_FIELDS}
    )
    raw_inputs.update(
        {field: (getter(field) or "").strip() for field in SCALAR_METADATA_FIELDS}
    )

    macro_targets: dict[str, float] = {}
    for field_name in MACRO_FIELDS:
        parsed = _parse_optional_float(raw_inputs[field_name], field_name)
        if parsed is not None:
            macro_targets[field_name] = parsed

    metadata_tags: dict[str, set[str]] = {}
    for field_name in LIST_METADATA_FIELDS:
        tags = _parse_tag_list(raw_inputs[field_name])
        if tags:
            metadata_tags[field_name] = tags

    metadata_scalars: dict[str, str | int] = {}
    nutriscore_grade = raw_inputs["nutriscore_grade"].lower()
    if nutriscore_grade:
        metadata_scalars["nutriscore_grade"] = nutriscore_grade

    nova_group = _parse_optional_int(raw_inputs["nova_group"], "nova_group")
    if nova_group is not None:
        if not (1 <= nova_group <= 4):
            raise ValueError("Invalid value for 'nova_group': expected 1-4")
        metadata_scalars["nova_group"] = nova_group

    source_raw = (getter("source") or "").strip()
    source_filter: set[str] = set()
    if source_raw:
        source_filter = {s.strip().lower() for s in source_raw.split(",") if s.strip()}
    raw_inputs["source"] = source_raw

    user_id_raw = (getter("user_id") or "").strip()
    user_id = _parse_optional_int(user_id_raw, "user_id") if user_id_raw else None
    raw_inputs["user_id"] = user_id_raw

    meal_raw = (getter("meal") or "").strip()
    meal_food_codes: list[int] = []
    if meal_raw:
        for part in meal_raw.split(","):
            part = part.strip()
            if part:
                code = _parse_optional_int(part, "meal")
                if code is not None:
                    meal_food_codes.append(code)
    raw_inputs["meal"] = meal_raw

    weight_overrides: dict[str, float] = {}
    for scorer_name in WEIGHT_PARAM_NAMES:
        param_name = f"w_{scorer_name}"
        raw_weight = (getter(param_name) or "").strip()
        if raw_weight:
            parsed = _parse_optional_float(raw_weight, param_name)
            if parsed is not None:
                if parsed < 0.0:
                    raise ValueError(f"Weight for '{scorer_name}' must be >= 0")
                weight_overrides[scorer_name] = parsed
        raw_inputs[param_name] = raw_weight

    return (
        SearchRequest(
            query=query,
            macro_targets=macro_targets,
            metadata_tags=metadata_tags,
            metadata_scalars=metadata_scalars,
            source_filter=source_filter,
            user_id=user_id,
            meal_food_codes=meal_food_codes,
            weight_overrides=weight_overrides,
        ),
        raw_inputs,
    )
