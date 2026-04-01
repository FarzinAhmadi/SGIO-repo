from __future__ import annotations

import logging
from collections.abc import Iterator

import polars as pl

from food_similarity.config import UsdaSourceConfig
from food_similarity.data import ProductChunk, _build_document_text
from food_similarity.signals import MACRO_FIELD_TO_NUTRIMENT_KEY

logger = logging.getLogger(__name__)

# FNDDS 2017-2018 Appendix H — food code grouping by first 2 digits.
# Source: https://www.ars.usda.gov/ARSUserFiles/80400530/pdf/fndds/2017_2018_FNDDS_Doc.pdf
FNDDS_MAIN_GROUP: dict[str, str] = {
    "1": "Milk and Milk Products",
    "2": "Meat, Poultry, Fish, and Mixtures",
    "3": "Eggs",
    "4": "Dry Beans, Peas, Other Legumes, Nuts, and Seeds",
    "5": "Grain Products",
    "6": "Fruits",
    "7": "Vegetables",
    "8": "Fats, Oils, and Salad Dressings",
    "9": "Sugars, Sweets, and Beverages",
}

FNDDS_SUBGROUP: dict[str, str] = {
    "11": "Milks, milk drinks, yogurts, infant formulas",
    "12": "Creams and cream substitutes",
    "13": "Milk desserts and sauces",
    "14": "Cheeses",
    "20": "Meat",
    "21": "Beef",
    "22": "Pork",
    "23": "Lamb, veal, game",
    "24": "Poultry",
    "25": "Organ meats, frankfurters, sausages, lunchmeats",
    "26": "Fish, shellfish",
    "27": "Meat, poultry, fish mixtures",
    "28": "Frozen meals, soups, gravies",
    "31": "Eggs",
    "32": "Egg mixtures",
    "33": "Egg substitutes",
    "41": "Legumes",
    "42": "Nuts, nut butters, nut mixtures",
    "43": "Seeds and seed mixtures",
    "44": "Carob products",
    "50": "Flour and dry mixes",
    "51": "Yeast breads, rolls",
    "52": "Quick breads",
    "53": "Cakes, cookies, pies, pastries, bars",
    "54": "Crackers, snack products",
    "55": "Pancakes, waffles, French toast, other grain products",
    "56": "Pastas, rice, cooked cereals",
    "57": "Cereals, not cooked",
    "58": "Grain mixtures, frozen meals, soups",
    "59": "Meat substitutes",
    "61": "Citrus fruits, juices",
    "62": "Dried fruits",
    "63": "Other fruits",
    "64": "Fruit juices and nectars excluding citrus",
    "67": "Fruits and juices baby food",
    "71": "White potatoes, starchy vegetables",
    "72": "Dark-green vegetables",
    "73": "Orange vegetables",
    "74": "Tomatoes, tomato mixtures",
    "75": "Other vegetables",
    "76": "Vegetables and mixtures mostly vegetables baby food",
    "77": "Vegetables with meat, poultry, fish",
    "78": "Mixtures mostly vegetables without meat, poultry, fish",
    "81": "Fats",
    "82": "Oils",
    "83": "Salad dressings",
    "89": "For use with a sandwich or vegetable",
    "91": "Sugars, sweets",
    "92": "Nonalcoholic beverages",
    "93": "Alcoholic beverages",
    "94": "Noncarbonated water",
    "95": "Formulated nutrition beverages, energy drinks, sports drinks",
    "99": "Used as an ingredient, not for coding",
}


def _food_code_to_groups(food_code: int | str) -> list[str]:
    """Derive FNDDS main-group and subgroup tags from an 8-digit food code."""
    code_str = str(food_code)
    tags: list[str] = []
    if len(code_str) >= 2:
        prefix2 = code_str[:2]
        prefix1 = code_str[0]
        if main := FNDDS_MAIN_GROUP.get(prefix1):
            tags.append(main)
        if sub := FNDDS_SUBGROUP.get(prefix2):
            tags.append(sub)
    return tags


USDA_NUTRIENT_MAP: dict[str, str] = {
    "energy_kcal_100g": "energy_kcal",
    "fat_100g": "total_fat_gm",
    "carbohydrates_100g": "carbohydrate_gm",
    "proteins_100g": "protein_gm",
    "sugars_100g": "total_sugars_gm",
    "fiber_100g": "dietary_fiber_gm",
    "saturated_fat_100g": "total_saturated_fatty_acids_gm",
}


def iter_usda_chunks(
    config: UsdaSourceConfig,
    chunk_size: int = 50_000,
    *,
    max_products: int | None = None,
    skip_chunks: int = 0,
) -> Iterator[ProductChunk]:
    logger.info("Reading USDA data from %s", config.descriptions_path)

    desc_df = pl.read_csv(config.descriptions_path)
    nutr_df = pl.read_csv(config.nutrients_path)
    joined = desc_df.join(nutr_df, on="food_code", how="left")

    # Filter by min name length
    joined = joined.filter(pl.col("food_desc").str.len_chars() >= config.min_name_length)

    if max_products is not None:
        joined = joined.head(max_products)

    total = len(joined)
    logger.info("USDA: %d products after filtering", total)

    chunk_idx = 0
    for start in range(0, total, chunk_size):
        if chunk_idx < skip_chunks:
            chunk_idx += 1
            continue

        end = min(start + chunk_size, total)
        batch = joined.slice(start, end - start)

        product_names: list[str] = batch.get_column("food_desc").to_list()
        food_codes: list[int] = batch.get_column("food_code").to_list()
        categories: list[list[str]] = [
            [cat] if cat else []
            for cat in batch.get_column("category_desc").to_list()
        ]
        food_groups: list[list[str]] = [
            _food_code_to_groups(fc) for fc in food_codes
        ]

        doc_texts = [
            _build_document_text(name, "", cats)
            for name, cats in zip(product_names, categories)
        ]

        macro_values: dict[str, list[float | None]] = {}
        for canonical, usda_col in USDA_NUTRIENT_MAP.items():
            if usda_col in batch.columns:
                macro_values[canonical] = [
                    float(v) if v is not None else None
                    for v in batch.get_column(usda_col).to_list()
                ]
            else:
                macro_values[canonical] = [None] * len(batch)

        # salt_100g from sodium_mg: salt_g = sodium_mg * 2.5 / 1000
        if "sodium_mg" in batch.columns:
            macro_values["salt_100g"] = [
                float(v) * 2.5 / 1000 if v is not None else None
                for v in batch.get_column("sodium_mg").to_list()
            ]
        else:
            macro_values["salt_100g"] = [None] * len(batch)

        size = len(product_names)
        yield ProductChunk(
            chunk_idx=chunk_idx,
            row_id_start=start,
            product_names=product_names,
            brands=[""] * size,
            categories=categories,
            food_groups_tags=food_groups,
            pnns_groups_1_tags=[[] for _ in range(size)],
            pnns_groups_2_tags=[[] for _ in range(size)],
            labels_tags=[[] for _ in range(size)],
            nutriscore_grades=[""] * size,
            nova_groups=[None] * size,
            macro_values=macro_values,
            document_texts=doc_texts,
            source="usda",
        )
        chunk_idx += 1

    logger.info("USDA streaming complete: %d products", total)
