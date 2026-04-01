"""Build a complete pairwise food similarity matrix for a food database.

Combines four fast, cheap scorers:
  embedding   - cosine similarity from pre-built FAISS embeddings (L2-normalised)
  macro       - nutrient profile similarity via mean scaled-L1 across nutrients
  category    - Jaccard similarity over WWEIA category tags
  food_group  - Jaccard similarity over FNDDS food-group tags (main group + subgroup)

The final score is a configurable weighted average of enabled scorers.
Output is a compressed NumPy matrix (.npz) plus a companion food-index parquet
for label lookups.

Usage:
    uv run python -m scripts.build_similarity_matrix --source usda
    uv run python -m scripts.build_similarity_matrix --source off --out data/similarity/off
    uv run python -m scripts.build_similarity_matrix --source usda --max-items 500  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from food_similarity.config import load_config
from food_similarity.signals import DEFAULT_MACRO_SCALES, MACRO_FIELDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class FoodData(NamedTuple):
    """All per-item data needed to compute the similarity matrix."""

    food_ids: np.ndarray        # (N,) int64  — FAISS row IDs
    names: list[str]            # (N,)        — display names
    embeddings: np.ndarray      # (N, D) float32, L2-normalised
    macros: np.ndarray          # (N, K) float32, may contain NaN
    macro_names: list[str]      # (K,)        — nutrient column names
    macro_scales: np.ndarray    # (K,) float32 — IQR normalisation scales
    categories: list[list[str]] # (N,)        — category tag lists
    food_groups: list[list[str]]  # (N,)      — FNDDS food-group tag lists


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_macro_scales(path: Path) -> dict[str, float]:
    scales = dict(DEFAULT_MACRO_SCALES)
    if not path.exists():
        return scales
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read macro stats %s; using defaults", path)
        return scales
    if isinstance(raw, dict):
        for field_name in MACRO_FIELDS:
            payload = raw.get(field_name)
            if not isinstance(payload, dict):
                continue
            scale = payload.get("scale")
            if isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0:
                scales[field_name] = float(scale)
    return scales


def load_source_data(
    index_dir: Path,
    *,
    max_items: int | None = None,
) -> FoodData:
    """Load metadata and pre-built embeddings for one source index.

    Expects the standard layout written by scripts/build_index.py:
        <index_dir>/metadata.parquet
        <index_dir>/macro_stats.json
        <index_dir>/checkpoints/chunk_*.npy   (embedding chunks)
    """
    meta_path = index_dir / "metadata.parquet"
    df = pl.read_parquet(str(meta_path))
    if max_items is not None:
        df = df.head(max_items)
    n = len(df)
    logger.info("Loaded %d food items from %s", n, meta_path)

    food_ids = df["row_id"].to_numpy().astype(np.int64)
    names: list[str] = df["product_name"].to_list()
    categories: list[list[str]] = df["categories_tags"].to_list()
    food_groups: list[list[str]] = (
        df["food_groups_tags"].to_list()
        if "food_groups_tags" in df.columns
        else [[] for _ in range(n)]
    )

    # Macronutrients
    scales_map = _load_macro_scales(index_dir / "macro_stats.json")
    macro_names = [f for f in MACRO_FIELDS if f in df.columns]
    if macro_names:
        macro_data = np.stack(
            [df[col].to_numpy().astype(np.float32) for col in macro_names], axis=1
        )
        macro_scales = np.array([scales_map[col] for col in macro_names], dtype=np.float32)
    else:
        macro_data = np.full((n, 0), np.nan, dtype=np.float32)
        macro_scales = np.empty(0, dtype=np.float32)

    # Embeddings from checkpoint .npy files
    checkpoint_dir = index_dir / "checkpoints"
    emb_files = sorted(checkpoint_dir.glob("chunk_*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No embedding checkpoints found in {checkpoint_dir}")

    chunks = [np.load(str(p)).astype(np.float32) for p in emb_files]
    embeddings_full = np.vstack(chunks)
    if len(embeddings_full) < n:
        raise ValueError(
            f"Not enough embeddings: {len(embeddings_full)} vectors but {n} metadata rows"
        )
    embeddings = embeddings_full[:n]
    logger.info("Loaded embeddings: shape=%s dtype=%s", embeddings.shape, embeddings.dtype)

    return FoodData(
        food_ids=food_ids,
        names=names,
        embeddings=embeddings,
        macros=macro_data,
        macro_names=macro_names,
        macro_scales=macro_scales,
        categories=categories,
        food_groups=food_groups,
    )


# ---------------------------------------------------------------------------
# Similarity scorers  —  each returns (N, N) float32 in [0, 1]
# ---------------------------------------------------------------------------

def embedding_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity via matrix multiply (embeddings are L2-normalised)."""
    n = len(embeddings)
    logger.info("Computing embedding similarity (%d×%d dot product)...", n, n)
    sim = embeddings @ embeddings.T
    np.clip(sim, 0.0, 1.0, out=sim)
    return sim


def macro_similarity(macros: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Pairwise nutrient profile similarity via mean scaled-L1 distance.

    Each nutrient is normalised by its IQR scale before computing
    ``1 / (1 + |xi - xj| / scale)``.  Pairs where either item is missing a
    nutrient exclude that nutrient from the average; pairs with no shared
    nutrients receive score 0.
    """
    n, k = macros.shape
    logger.info("Computing macro similarity (%d×%d, %d nutrients)...", n, n, k)
    if k == 0:
        return np.zeros((n, n), dtype=np.float32)

    total = np.zeros((n, n), dtype=np.float32)
    count = np.zeros((n, n), dtype=np.float32)

    for j in range(k):
        v = macros[:, j]                          # (N,) float32, may have NaN
        valid = ~np.isnan(v)
        if valid.sum() < 2:
            continue

        scale = max(float(scales[j]), 1e-6)
        diff = np.abs(v[:, None] - v[None, :]) / scale  # (N, N); NaN where invalid
        pair_valid = (valid[:, None] & valid[None, :]).astype(np.float32)
        sim_j = np.where(pair_valid.astype(bool), 1.0 / (1.0 + diff), 0.0).astype(np.float32)

        total += sim_j
        count += pair_valid

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(count > 0, total / count, 0.0)

    return result.astype(np.float32)


def category_similarity(categories: list[list[str]]) -> np.ndarray:
    """Jaccard similarity over food-category tag sets.

    Items with no categories get score 0 with all other items (undefined,
    not trivially equal).
    """
    n = len(categories)
    logger.info("Computing category similarity (Jaccard, %d items)...", n)

    # Build vocabulary from all non-empty tags
    vocab: dict[str, int] = {}
    for cats in categories:
        if cats:
            for c in cats:
                if c and c not in vocab:
                    vocab[c] = len(vocab)

    if not vocab:
        return np.zeros((n, n), dtype=np.float32)

    c = len(vocab)
    B = np.zeros((n, c), dtype=np.float32)
    for i, cats in enumerate(categories):
        if cats:
            for cat in cats:
                idx = vocab.get(cat)
                if idx is not None:
                    B[i, idx] = 1.0

    intersection = B @ B.T                                # (N, N)
    row_sums = B.sum(axis=1)                              # (N,)
    union = row_sums[:, None] + row_sums[None, :] - intersection

    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)

    return jaccard.astype(np.float32)


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------

def build_similarity_matrix(
    data: FoodData,
    *,
    embedding_weight: float = 1.0,
    macro_weight: float = 0.5,
    category_weight: float = 0.3,
    food_group_weight: float = 0.3,
) -> np.ndarray:
    """Compute the weighted similarity matrix, shape (N, N), values in [0, 1]."""
    scorer_specs: list[tuple[str, float]] = [
        ("embedding", embedding_weight),
        ("macro", macro_weight),
        ("category", category_weight),
        ("food_group", food_group_weight),
    ]
    active = [(name, w) for name, w in scorer_specs if w > 0]
    if not active:
        raise ValueError("All scorer weights are 0")

    total_weight = sum(w for _, w in active)
    n = len(data.food_ids)
    result = np.zeros((n, n), dtype=np.float32)

    for name, weight in active:
        if name == "embedding":
            sim = embedding_similarity(data.embeddings)
        elif name == "macro":
            if data.macros.shape[1] == 0:
                logger.warning("No macro columns available; skipping macro scorer")
                continue
            sim = macro_similarity(data.macros, data.macro_scales)
        elif name == "category":
            sim = category_similarity(data.categories)
        elif name == "food_group":
            sim = category_similarity(data.food_groups)
        else:
            raise ValueError(f"Unknown scorer: {name!r}")

        result += (weight / total_weight) * sim
        logger.info("Applied scorer '%s' (weight=%.2f / total=%.2f)", name, weight, total_weight)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a pairwise food similarity matrix from a pre-built index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.toml", help="App config file")
    parser.add_argument(
        "--source", default="usda",
        help="Source name (must match a source directory under the index path)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path prefix (default: data/similarity/<source>)",
    )
    parser.add_argument(
        "--embedding-weight", type=float, default=1.0,
        help="Weight for embedding (cosine) similarity",
    )
    parser.add_argument(
        "--macro-weight", type=float, default=0.5,
        help="Weight for macronutrient profile similarity",
    )
    parser.add_argument(
        "--category-weight", type=float, default=0.3,
        help="Weight for Jaccard category similarity (WWEIA categories)",
    )
    parser.add_argument(
        "--food-group-weight", type=float, default=0.3,
        help="Weight for Jaccard food-group similarity (FNDDS main/subgroup)",
    )
    parser.add_argument(
        "--dtype", choices=["float16", "float32"], default="float16",
        help="Matrix storage dtype (float16 saves ~50%% disk space)",
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Limit number of items (for smoke testing)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    index_dir = Path(config.index.path) / args.source

    out_prefix = Path(args.out) if args.out else Path("data/similarity") / args.source
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load
    data = load_source_data(index_dir, max_items=args.max_items)

    # Build
    matrix = build_similarity_matrix(
        data,
        embedding_weight=args.embedding_weight,
        macro_weight=args.macro_weight,
        category_weight=args.category_weight,
        food_group_weight=args.food_group_weight,
    )

    # Cast to output dtype
    save_dtype = np.float16 if args.dtype == "float16" else np.float32
    matrix = matrix.astype(save_dtype)

    # Save matrix + food IDs together
    out_npz = out_prefix.parent / (out_prefix.name + ".npz")
    np.savez_compressed(str(out_npz), similarity=matrix, food_ids=data.food_ids)
    size_mb = out_npz.stat().st_size / 1e6
    logger.info("Saved similarity matrix → %s (%.1f MB)", out_npz, size_mb)

    # Save full food index for label lookups
    out_index = out_prefix.parent / (out_prefix.name + "_index.parquet")
    pl.DataFrame({
        "food_id": data.food_ids.tolist(),
        "name": data.names,
        "categories": data.categories,
        "food_groups": data.food_groups,
        **{col: data.macros[:, j].tolist() for j, col in enumerate(data.macro_names)},
    }).write_parquet(str(out_index))
    logger.info("Saved food index → %s", out_index)

    # Summary statistics
    # Sample upper triangle (avoid loading entire matrix if large)
    tri = matrix[np.triu_indices(len(matrix), k=1)]
    logger.info(
        "Similarity stats (upper triangle, N=%d pairs): "
        "mean=%.4f  p50=%.4f  p90=%.4f  p99=%.4f  max=%.4f",
        len(tri),
        float(np.mean(tri)),
        float(np.percentile(tri, 50)),
        float(np.percentile(tri, 90)),
        float(np.percentile(tri, 99)),
        float(np.max(tri)),
    )

    logger.info(
        "Done. Shape=%s dtype=%s elapsed=%.1fs",
        matrix.shape, matrix.dtype, time.time() - t0,
    )


if __name__ == "__main__":
    main()
