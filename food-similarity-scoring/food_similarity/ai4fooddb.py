from __future__ import annotations

import logging
from collections.abc import Iterator

import polars as pl

from food_similarity.config import Ai4fooddbSourceConfig
from food_similarity.data import ProductChunk, _build_document_text
from food_similarity.signals import MACRO_FIELDS

logger = logging.getLogger(__name__)


def iter_ai4fooddb_chunks(
    config: Ai4fooddbSourceConfig,
    chunk_size: int = 50_000,
    *,
    max_products: int | None = None,
    skip_chunks: int = 0,
) -> Iterator[ProductChunk]:
    logger.info("Reading AI4FoodDB data from %s", config.path)

    df = pl.read_csv(config.path, infer_schema_length=10_000)

    # Filter by min name length
    df = df.filter(pl.col("food_name").str.len_chars() >= config.min_name_length)

    if max_products is not None:
        df = df.head(max_products)

    total = len(df)
    logger.info("AI4FoodDB: %d food items", total)

    chunk_idx = 0
    for start in range(0, total, chunk_size):
        if chunk_idx < skip_chunks:
            chunk_idx += 1
            continue

        end = min(start + chunk_size, total)
        batch = df.slice(start, end - start)

        product_names: list[str] = batch.get_column("food_name").to_list()
        categories: list[list[str]] = [
            [c] if c else [] for c in batch.get_column("category").to_list()
        ]

        doc_texts = [
            _build_document_text(name, "", cats)
            for name, cats in zip(product_names, categories)
        ]

        size = len(product_names)
        macro_values: dict[str, list[float | None]] = {
            field: [None] * size for field in MACRO_FIELDS
        }

        yield ProductChunk(
            chunk_idx=chunk_idx,
            row_id_start=start,
            product_names=product_names,
            brands=[""] * size,
            categories=categories,
            food_groups_tags=[[] for _ in range(size)],
            pnns_groups_1_tags=[[] for _ in range(size)],
            pnns_groups_2_tags=[[] for _ in range(size)],
            labels_tags=[[] for _ in range(size)],
            nutriscore_grades=[""] * size,
            nova_groups=[None] * size,
            macro_values=macro_values,
            document_texts=doc_texts,
            source="ai4fooddb",
        )
        chunk_idx += 1

    logger.info("AI4FoodDB streaming complete: %d food items", total)
