"""Offline script: stream data, embed in chunks, and build FAISS index."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

from food_similarity.ai4fooddb import iter_ai4fooddb_chunks
from food_similarity.config import (
    Ai4fooddbSourceConfig,
    DataConfig,
    MfpSourceConfig,
    UsdaSourceConfig,
    load_config,
    resolve_device,
)
from food_similarity.data import iter_product_chunks
from food_similarity.embedding import EmbeddingModel
from food_similarity.index import FaissIndex, MetadataStore
from food_similarity.myfitnesspal import iter_mfp_chunks
from food_similarity.signals import DEFAULT_MACRO_SCALES, MACRO_FIELDS
from food_similarity.usda import iter_usda_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _compute_macro_stats(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for field_name in MACRO_FIELDS:
        if field_name not in df.columns:
            default_scale = DEFAULT_MACRO_SCALES[field_name]
            stats[field_name] = {
                "q25": 0.0,
                "q75": 0.0,
                "iqr": default_scale,
                "scale": default_scale,
            }
            continue
        series = df.get_column(field_name).drop_nulls()
        if len(series) == 0:
            default_scale = DEFAULT_MACRO_SCALES[field_name]
            stats[field_name] = {
                "q25": 0.0,
                "q75": 0.0,
                "iqr": default_scale,
                "scale": default_scale,
            }
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        if not math.isfinite(iqr) or iqr <= 0:
            iqr = DEFAULT_MACRO_SCALES[field_name]
        stats[field_name] = {
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "scale": iqr,
        }
    return stats


def _build_source_index(
    source_config: DataConfig | UsdaSourceConfig,
    config,
    embed_model: EmbeddingModel,
    device: str,
    *,
    chunk_size: int,
    max_products: int | None,
    resume: bool,
) -> None:
    source_name = source_config.name
    index_base = Path(config.index.path)
    source_dir = index_base / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = source_dir / config.index.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta_chunk_dir = checkpoint_dir / "metadata"
    meta_chunk_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Building index for source: %s ===", source_name)

    # Resume: load partial FAISS index + count completed chunks to skip
    partial_index_path = source_dir / "faiss.index.partial"
    faiss_index = FaissIndex(config.scoring.embedding.dimension)
    skip_chunks = 0

    if resume and partial_index_path.exists():
        faiss_index.load(partial_index_path)
        completed = sorted(checkpoint_dir.glob("chunk_*.npy"))
        skip_chunks = len(completed)
        logger.info(
            "Resumed: loaded partial index (%d vectors), skipping %d chunks",
            faiss_index.index.ntotal, skip_chunks,
        )
    else:
        faiss_index.create()

    # Dispatch to the right chunk iterator
    if isinstance(source_config, UsdaSourceConfig):
        chunk_iter = iter_usda_chunks(
            source_config,
            chunk_size=chunk_size,
            max_products=max_products,
            skip_chunks=skip_chunks,
        )
    elif isinstance(source_config, MfpSourceConfig):
        chunk_iter = iter_mfp_chunks(
            source_config,
            chunk_size=chunk_size,
            max_products=max_products,
            skip_chunks=skip_chunks,
        )
    elif isinstance(source_config, Ai4fooddbSourceConfig):
        chunk_iter = iter_ai4fooddb_chunks(
            source_config,
            chunk_size=chunk_size,
            max_products=max_products,
            skip_chunks=skip_chunks,
        )
    else:
        chunk_iter = iter_product_chunks(
            source_config,
            chunk_size=chunk_size,
            max_products=max_products,
            skip_chunks=skip_chunks,
        )

    total_embedded = faiss_index.index.ntotal
    total_start = time.time()

    for chunk in chunk_iter:
        emb_path = checkpoint_dir / f"chunk_{chunk.chunk_idx:05d}.npy"
        meta_path = meta_chunk_dir / f"chunk_{chunk.chunk_idx:05d}.parquet"

        # Save metadata chunk
        meta_dict: dict[str, object] = {
            "row_id": list(
                range(chunk.row_id_start, chunk.row_id_start + chunk.size)
            ),
            "product_name": chunk.product_names,
            "brands": chunk.brands,
            "categories_tags": chunk.categories,
            "food_groups_tags": chunk.food_groups_tags,
            "pnns_groups_1_tags": chunk.pnns_groups_1_tags,
            "pnns_groups_2_tags": chunk.pnns_groups_2_tags,
            "labels_tags": chunk.labels_tags,
            "nutriscore_grade": chunk.nutriscore_grades,
            "nova_group": chunk.nova_groups,
            "document_text": chunk.document_texts,
            "source": [chunk.source] * chunk.size,
        }
        for field_name in MACRO_FIELDS:
            if field_name in chunk.macro_values:
                meta_dict[field_name] = chunk.macro_values[field_name]
        meta_df = pl.DataFrame(meta_dict)
        meta_df.write_parquet(str(meta_path))

        # Embed
        logger.info(
            "Chunk %d: embedding %d docs (rows %d-%d)...",
            chunk.chunk_idx, chunk.size,
            chunk.row_id_start, chunk.row_id_start + chunk.size - 1,
        )
        chunk_start = time.time()
        embeddings = embed_model.encode_documents(
            chunk.document_texts, batch_size=config.index.batch_size
        )
        elapsed = time.time() - chunk_start

        faiss_index.add(embeddings)
        faiss_index.save(partial_index_path)
        np.save(emb_path, embeddings)
        del embeddings

        total_embedded += chunk.size
        docs_per_sec = chunk.size / elapsed if elapsed > 0 else 0
        logger.info(
            "Chunk %d done: %.1f docs/sec, %d products indexed",
            chunk.chunk_idx, docs_per_sec, total_embedded,
        )

    total_elapsed = time.time() - total_start
    logger.info(
        "Source %s: all chunks processed in %.1f min (%d products)",
        source_name, total_elapsed / 60, total_embedded,
    )

    # Save final FAISS index and clean up partial
    faiss_index.save(source_dir / config.index.faiss_file)
    if partial_index_path.exists():
        partial_index_path.unlink()

    # Combine metadata parquet chunks into one file
    logger.info("Combining metadata chunks for %s...", source_name)
    meta_parts = sorted(meta_chunk_dir.glob("chunk_*.parquet"))
    if not meta_parts:
        logger.warning("No metadata chunks found for %s, skipping", source_name)
        return
    combined_meta = pl.concat(
        [pl.read_parquet(str(p)) for p in meta_parts],
        how="diagonal_relaxed",
    )
    metadata_store = MetadataStore()
    metadata_store.save(combined_meta, source_dir / config.index.metadata_file)

    macro_stats = _compute_macro_stats(combined_meta)
    macro_stats_path = source_dir / config.index.macro_stats_file
    macro_stats_path.write_text(json.dumps(macro_stats, indent=2), encoding="utf-8")
    logger.info("Macro stats saved to %s", macro_stats_path)

    del combined_meta

    # Verification query
    logger.info("Running verification query for %s: 'milk'", source_name)
    query_vec = embed_model.encode_query("milk")
    scores, indices = faiss_index.search(query_vec, 5)

    metadata_store.load(source_dir / config.index.metadata_file)
    scored_results = []
    for score, row_id in zip(scores.tolist(), indices.tolist()):
        if row_id < 0:
            continue
        row = metadata_store.get(int(row_id))
        if row is not None:
            scored_results.append((row, score))

    logger.info("Top 5 results for 'milk' from %s:", source_name)
    for i, (row, score) in enumerate(scored_results):
        logger.info(
            "  %d. [%.4f] %s (by %s)",
            i + 1, score, row["product_name"], row.get("brands", ""),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for food similarity")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument(
        "--max-products", type=int, default=None,
        help="Limit number of products per source (for testing)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50_000,
        help="Products per chunk (default: 50000)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Build only a specific source (e.g. 'off' or 'usda')",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.scoring.embedding.device)
    logger.info("Embedding device: %s", device)

    # Load embedding model once (shared across sources)
    embed_model = EmbeddingModel(config.scoring.embedding, device=config.scoring.embedding.device)

    for source_config in config.sources:
        if not source_config.enabled:
            logger.info("Skipping disabled source: %s", source_config.name)
            continue
        if args.source and source_config.name != args.source:
            logger.info("Skipping source %s (--source=%s)", source_config.name, args.source)
            continue

        _build_source_index(
            source_config,
            config,
            embed_model,
            device,
            chunk_size=args.chunk_size,
            max_products=args.max_products,
            resume=args.resume,
        )

    logger.info("Index build complete!")


if __name__ == "__main__":
    main()
