from __future__ import annotations

import logging
from collections.abc import Iterator

import polars as pl

from food_similarity.config import MfpSourceConfig
from food_similarity.data import ProductChunk, _build_document_text

logger = logging.getLogger(__name__)

# MFP columns → canonical macro field names.
# NOTE: MFP macros are per-serving (not per-100g), but the serving size is
# free-text and not reliably parseable.  We store per-serving values; macro
# scoring will still be useful for within-source comparison.
MFP_NUTRIENT_MAP: dict[str, str] = {
    "energy_kcal_100g": "calories",
    "fat_100g": "fat",
    "carbohydrates_100g": "carbs",
    "proteins_100g": "protein",
    "sugars_100g": "sugar",
    "fiber_100g": "fiber",
    "saturated_fat_100g": "sat fat",
}


def iter_mfp_chunks(
    config: MfpSourceConfig,
    chunk_size: int = 50_000,
    *,
    max_products: int | None = None,
    skip_chunks: int = 0,
) -> Iterator[ProductChunk]:
    logger.info("Reading MyFitnessPal data from %s", config.path)

    df = pl.read_csv(config.path, infer_schema_length=10_000)

    # Deduplicate: group by (food_name, brand) and take median macros
    agg_exprs: list[pl.Expr] = []
    macro_cols = list(MFP_NUTRIENT_MAP.values()) + ["sodium"]
    for col in macro_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).median())

    deduped = (
        df.group_by(["food_name", "brand"])
        .agg(agg_exprs)
        .sort("food_name")
    )
    del df

    # Filter by min name length
    deduped = deduped.filter(
        pl.col("food_name").str.len_chars() >= config.min_name_length
    )

    if max_products is not None:
        deduped = deduped.head(max_products)

    total = len(deduped)
    logger.info("MyFitnessPal: %d unique products after deduplication", total)

    chunk_idx = 0
    for start in range(0, total, chunk_size):
        if chunk_idx < skip_chunks:
            chunk_idx += 1
            continue

        end = min(start + chunk_size, total)
        batch = deduped.slice(start, end - start)

        product_names: list[str] = batch.get_column("food_name").to_list()
        brands_raw: list[str | None] = batch.get_column("brand").to_list()
        brands: list[str] = [b if b is not None else "" for b in brands_raw]

        doc_texts = [
            _build_document_text(name, brand, None)
            for name, brand in zip(product_names, brands)
        ]

        macro_values: dict[str, list[float | None]] = {}
        for canonical, mfp_col in MFP_NUTRIENT_MAP.items():
            if mfp_col in batch.columns:
                macro_values[canonical] = [
                    float(v) if v is not None else None
                    for v in batch.get_column(mfp_col).to_list()
                ]
            else:
                macro_values[canonical] = [None] * len(batch)

        # salt_100g from sodium (mg): salt_g = sodium_mg * 2.5 / 1000
        if "sodium" in batch.columns:
            macro_values["salt_100g"] = [
                float(v) * 2.5 / 1000 if v is not None else None
                for v in batch.get_column("sodium").to_list()
            ]
        else:
            macro_values["salt_100g"] = [None] * len(batch)

        size = len(product_names)
        yield ProductChunk(
            chunk_idx=chunk_idx,
            row_id_start=start,
            product_names=product_names,
            brands=brands,
            categories=[[] for _ in range(size)],
            food_groups_tags=[[] for _ in range(size)],
            pnns_groups_1_tags=[[] for _ in range(size)],
            pnns_groups_2_tags=[[] for _ in range(size)],
            labels_tags=[[] for _ in range(size)],
            nutriscore_grades=[""] * size,
            nova_groups=[None] * size,
            macro_values=macro_values,
            document_texts=doc_texts,
            source="myfitnesspal",
        )
        chunk_idx += 1

    logger.info("MyFitnessPal streaming complete: %d unique products", total)
