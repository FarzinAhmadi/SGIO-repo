from __future__ import annotations

from typing import Final

MACRO_FIELD_TO_NUTRIMENT_KEY: Final[dict[str, str]] = {
    "energy_kcal_100g": "energy-kcal_100g",
    "fat_100g": "fat_100g",
    "carbohydrates_100g": "carbohydrates_100g",
    "proteins_100g": "proteins_100g",
    "sugars_100g": "sugars_100g",
    "fiber_100g": "fiber_100g",
    "salt_100g": "salt_100g",
    "saturated_fat_100g": "saturated-fat_100g",
}

MACRO_FIELDS: Final[tuple[str, ...]] = tuple(MACRO_FIELD_TO_NUTRIMENT_KEY.keys())

LIST_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "categories_tags",
    "food_groups_tags",
    "pnns_groups_1_tags",
    "pnns_groups_2_tags",
    "labels_tags",
)

SCALAR_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "nutriscore_grade",
    "nova_group",
)

KNOWN_SOURCES: Final[tuple[str, ...]] = ("off", "usda", "myfitnesspal", "ai4fooddb")

DEFAULT_MACRO_SCALES: Final[dict[str, float]] = {
    "energy_kcal_100g": 100.0,
    "fat_100g": 10.0,
    "carbohydrates_100g": 15.0,
    "proteins_100g": 8.0,
    "sugars_100g": 10.0,
    "fiber_100g": 4.0,
    "salt_100g": 1.0,
    "saturated_fat_100g": 5.0,
}

