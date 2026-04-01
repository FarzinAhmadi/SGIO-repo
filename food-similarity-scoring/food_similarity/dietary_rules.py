from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DietRule:
    macro: str
    direction: str  # "low" (penalize high values) or "high" (reward high values)


DIET_RULES: dict[str, list[DietRule]] = {
    "low_salt_or_low_sodium_diet": [DietRule("salt_100g", "low")],
    "sugar_free_or_low_sugar_diet": [DietRule("sugars_100g", "low")],
    "diabetic_diet": [
        DietRule("sugars_100g", "low"),
        DietRule("fiber_100g", "high"),
    ],
    "low_fat_or_low_cholesterol_diet": [
        DietRule("fat_100g", "low"),
        DietRule("saturated_fat_100g", "low"),
    ],
    "high_protein_diet": [DietRule("proteins_100g", "high")],
    "low_carbohydrate_diet": [DietRule("carbohydrates_100g", "low")],
    "high_fiber_diet": [DietRule("fiber_100g", "high")],
    "weight_loss_or_low_calorie_diet": [DietRule("energy_kcal_100g", "low")],
    "renal_or_kidney_diet": [
        DietRule("proteins_100g", "low"),
        DietRule("salt_100g", "low"),
    ],
}

DIET_FLAG_NAMES: tuple[str, ...] = (
    "weight_loss_or_low_calorie_diet",
    "low_fat_or_low_cholesterol_diet",
    "low_salt_or_low_sodium_diet",
    "sugar_free_or_low_sugar_diet",
    "low_fiber_diet",
    "high_fiber_diet",
    "diabetic_diet",
    "weight_gain_or_muscle_building_diet",
    "low_carbohydrate_diet",
    "high_protein_diet",
    "gluten_free_or_celiac_diet",
    "renal_or_kidney_diet",
    "other_special_diet",
)
