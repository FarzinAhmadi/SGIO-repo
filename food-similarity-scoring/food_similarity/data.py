from __future__ import annotations

import gzip
import json
import logging
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass

from food_similarity.config import DataConfig
from food_similarity.signals import MACRO_FIELD_TO_NUTRIMENT_KEY

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 500_000


@dataclass
class ProductChunk:
    """A chunk of parsed products ready for embedding + metadata storage."""

    chunk_idx: int
    row_id_start: int
    product_names: list[str]
    brands: list[str]
    categories: list[list[str]]
    food_groups_tags: list[list[str]]
    pnns_groups_1_tags: list[list[str]]
    pnns_groups_2_tags: list[list[str]]
    labels_tags: list[list[str]]
    nutriscore_grades: list[str]
    nova_groups: list[int | None]
    macro_values: dict[str, list[float | None]]
    document_texts: list[str]
    source: str = "off"

    @property
    def size(self) -> int:
        return len(self.product_names)


def _clean_category(tag: str) -> str:
    """Convert 'en:sweet-spreads' -> 'sweet spreads'."""
    tag = re.sub(r"^[a-z]{2}:", "", tag)
    return tag.replace("-", " ")


def _build_document_text(
    name: str, brand: str, categories: list[str] | None
) -> str:
    parts = [name]
    if brand:
        parts.append(f"by {brand}")
    if categories:
        cleaned = [_clean_category(c) for c in categories[:5]]
        parts.append(f"Categories: {', '.join(cleaned)}")
    return ". ".join(parts)


def _coerce_tag_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    tags = []
    for item in value:
        if isinstance(item, str):
            tag = item.strip().lower()
            if tag:
                tags.append(tag)
    return tags


def _coerce_macro_value(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return as_float


def _coerce_nova_group(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def iter_product_chunks(
    config: DataConfig,
    chunk_size: int = 50_000,
    *,
    max_products: int | None = None,
    skip_chunks: int = 0,
) -> Iterator[ProductChunk]:
    """Stream products from the gzipped JSONL, yielding fixed-size chunks.

    Only fields required for retrieval/reranking and optional structured
    scoring are extracted per row; the rest is discarded immediately, so
    memory usage stays proportional to chunk_size.

    If *skip_chunks* > 0, the first N chunks are fast-forwarded through
    (valid products are counted to maintain consistent chunk boundaries,
    but no data is accumulated or yielded).
    """
    logger.info("Streaming products from %s (chunk_size=%d)", config.path, chunk_size)
    if skip_chunks:
        logger.info("Skipping first %d chunks (%d products)...", skip_chunks, skip_chunks * chunk_size)

    name_field = config.name_field
    brand_field = config.brand_field
    cat_field = config.categories_field
    min_len = config.min_name_length

    names: list[str] = []
    brands: list[str] = []
    cats: list[list[str]] = []
    food_groups_tags: list[list[str]] = []
    pnns_groups_1_tags: list[list[str]] = []
    pnns_groups_2_tags: list[list[str]] = []
    labels_tags: list[list[str]] = []
    nutriscore_grades: list[str] = []
    nova_groups: list[int | None] = []
    macro_values = {name: [] for name in MACRO_FIELD_TO_NUTRIMENT_KEY}
    doc_texts: list[str] = []

    chunk_idx = 0
    row_id_start = 0
    total_kept = 0
    skipped_in_chunk = 0  # counter for products in the current skipped chunk
    lines_read = 0

    with gzip.open(config.path, "rt", encoding="utf-8") as f:
        for line in f:
            lines_read += 1
            if lines_read % PROGRESS_INTERVAL == 0:
                logger.info(
                    "Scanned %d lines, kept %d products so far...",
                    lines_read, total_kept,
                )

            if max_products is not None and total_kept >= max_products:
                break

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            product_name = row.get(name_field)
            if not product_name or not isinstance(product_name, str):
                continue
            if len(product_name) < min_len:
                continue

            total_kept += 1

            # Fast-forward: count valid products but don't accumulate data
            if chunk_idx < skip_chunks:
                skipped_in_chunk += 1
                if skipped_in_chunk >= chunk_size:
                    row_id_start += skipped_in_chunk
                    chunk_idx += 1
                    skipped_in_chunk = 0
                continue

            brand = row.get(brand_field) or ""
            raw_cats = row.get(cat_field)
            cat_list = _coerce_tag_list(raw_cats)
            nutriments = row.get("nutriments")
            nutriments_dict = nutriments if isinstance(nutriments, dict) else {}

            names.append(product_name)
            brands.append(brand)
            cats.append(cat_list)
            food_groups_tags.append(_coerce_tag_list(row.get("food_groups_tags")))
            pnns_groups_1_tags.append(_coerce_tag_list(row.get("pnns_groups_1_tags")))
            pnns_groups_2_tags.append(_coerce_tag_list(row.get("pnns_groups_2_tags")))
            labels_tags.append(_coerce_tag_list(row.get("labels_tags")))
            nutriscore_grades.append(
                row.get("nutriscore_grade", "").strip().lower()
                if isinstance(row.get("nutriscore_grade"), str)
                else ""
            )
            nova_groups.append(_coerce_nova_group(row.get("nova_group")))
            for macro_name, raw_key in MACRO_FIELD_TO_NUTRIMENT_KEY.items():
                macro_values[macro_name].append(
                    _coerce_macro_value(nutriments_dict.get(raw_key))
                )
            doc_texts.append(_build_document_text(product_name, brand, cat_list))

            # Yield a full chunk
            if len(names) >= chunk_size:
                yield ProductChunk(
                    chunk_idx=chunk_idx,
                    row_id_start=row_id_start,
                    product_names=names,
                    brands=brands,
                    categories=cats,
                    food_groups_tags=food_groups_tags,
                    pnns_groups_1_tags=pnns_groups_1_tags,
                    pnns_groups_2_tags=pnns_groups_2_tags,
                    labels_tags=labels_tags,
                    nutriscore_grades=nutriscore_grades,
                    nova_groups=nova_groups,
                    macro_values=macro_values,
                    document_texts=doc_texts,
                )
                row_id_start += len(names)
                chunk_idx += 1
                names, brands, cats, doc_texts = [], [], [], []
                food_groups_tags, pnns_groups_1_tags, pnns_groups_2_tags = [], [], []
                labels_tags, nutriscore_grades, nova_groups = [], [], []
                macro_values = {name: [] for name in MACRO_FIELD_TO_NUTRIMENT_KEY}

    # Handle leftover products from skipping (partial last skipped chunk)
    if chunk_idx < skip_chunks and skipped_in_chunk > 0:
        row_id_start += skipped_in_chunk
        chunk_idx += 1

    # Yield the final partial chunk
    if names:
        yield ProductChunk(
            chunk_idx=chunk_idx,
            row_id_start=row_id_start,
            product_names=names,
            brands=brands,
            categories=cats,
            food_groups_tags=food_groups_tags,
            pnns_groups_1_tags=pnns_groups_1_tags,
            pnns_groups_2_tags=pnns_groups_2_tags,
            labels_tags=labels_tags,
            nutriscore_grades=nutriscore_grades,
            nova_groups=nova_groups,
            macro_values=macro_values,
            document_texts=doc_texts,
        )

    logger.info(
        "Streaming complete: %d lines scanned, %d products kept",
        lines_read, total_kept,
    )
