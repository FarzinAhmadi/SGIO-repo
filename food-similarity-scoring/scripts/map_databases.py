"""Map food items from one database to another via nearest-neighbour lookup.

For each item in the source database, finds the k nearest items in the target
database using a two-stage pipeline:
  1. Embedding retrieval — cosine similarity (dot product on L2-normalised
     vectors) selects 2*k candidates per source item.
  2. Reranker refinement — a cross-encoder re-scores the candidates; the final
     score is a weighted average of embedding and reranker scores, keeping
     the top k.

Memory-efficient: target embeddings are held in memory while source embeddings
are streamed in batches from checkpoint files.  No full N×M matrix is ever
materialised.

Usage:
    uv run python -m scripts.map_databases --source myfitnesspal --target usda
    uv run python -m scripts.map_databases --source myfitnesspal --target usda --resume
    uv run python -m scripts.map_databases --source myfitnesspal --target usda --k 10

Output:
    data/similarity/myfitnesspal_to_usda.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from food_similarity.config import load_config, resolve_device
from food_similarity.reranker import RerankerModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class TargetData(NamedTuple):
    """Target database — fully loaded into memory."""

    food_ids: np.ndarray    # (M,) int64
    names: list[str]        # (M,)
    doc_texts: list[str]    # (M,) for reranker input
    embeddings: np.ndarray  # (M, D) float32, L2-normalised


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_target(index_dir: Path) -> TargetData:
    """Load the full target database into memory."""
    df = pl.read_parquet(str(index_dir / "metadata.parquet"))
    food_ids = df["row_id"].to_numpy().astype(np.int64)
    names: list[str] = df["product_name"].to_list()
    doc_texts: list[str] = df["document_text"].to_list()

    emb_files = sorted((index_dir / "checkpoints").glob("chunk_*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No embedding checkpoints in {index_dir / 'checkpoints'}")
    chunks = [np.load(str(p)).astype(np.float32) for p in emb_files]
    embeddings = np.vstack(chunks)[: len(df)]

    logger.info("Target: %d items, embeddings %s", len(df), embeddings.shape)
    return TargetData(food_ids, names, doc_texts, embeddings)


def iter_source_batches(
    index_dir: Path,
    batch_size: int,
    *,
    already_done: set[int] | None = None,
) -> tuple[int, ...]:
    """Yield (food_ids, names, embeddings) batches from source checkpoint files.

    Streams one checkpoint .npy at a time so peak memory is bounded by
    max(checkpoint_size) + batch_size rather than total source size.
    Items whose food_id is in *already_done* are skipped.
    """
    df = pl.read_parquet(str(index_dir / "metadata.parquet"))
    all_food_ids = df["row_id"].to_numpy().astype(np.int64)
    all_names: list[str] = df["product_name"].to_list()
    n = len(df)
    del df  # free DataFrame memory

    emb_files = sorted((index_dir / "checkpoints").glob("chunk_*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No embedding checkpoints in {index_dir / 'checkpoints'}")

    # Accumulate rows into batches across checkpoint boundaries
    buf_ids: list[int] = []
    buf_names: list[str] = []
    buf_embs: list[np.ndarray] = []
    global_offset = 0

    for emb_file in emb_files:
        chunk_embs = np.load(str(emb_file)).astype(np.float32)
        chunk_len = min(len(chunk_embs), n - global_offset)

        for local_i in range(chunk_len):
            global_i = global_offset + local_i
            fid = int(all_food_ids[global_i])

            if already_done and fid in already_done:
                continue

            buf_ids.append(fid)
            buf_names.append(all_names[global_i])
            buf_embs.append(chunk_embs[local_i])

            if len(buf_ids) == batch_size:
                yield (
                    np.array(buf_ids, dtype=np.int64),
                    buf_names,
                    np.stack(buf_embs),
                )
                buf_ids, buf_names, buf_embs = [], [], []

        global_offset += chunk_len
        # chunk_embs goes out of scope here, freeing memory before next file

    # Flush remaining items
    if buf_ids:
        yield (
            np.array(buf_ids, dtype=np.int64),
            buf_names,
            np.stack(buf_embs),
        )


# ---------------------------------------------------------------------------
# Retrieval + reranking
# ---------------------------------------------------------------------------

def find_candidates(
    source_embs: np.ndarray,
    target: TargetData,
    candidate_k: int,
) -> list[list[tuple[int, float]]]:
    """Return top candidate_k (target_index, embedding_score) per source item."""
    # (B, D) @ (D, M) → (B, M)  — cosine sim for L2-normed vectors
    sims = source_embs @ target.embeddings.T
    np.clip(sims, 0.0, 1.0, out=sims)

    capped_k = min(candidate_k, sims.shape[1])
    results: list[list[tuple[int, float]]] = []

    for i in range(len(source_embs)):
        row = sims[i]
        if capped_k >= len(row):
            top_idxs = np.argsort(row)[::-1][:capped_k]
        else:
            part = np.argpartition(row, -capped_k)[-capped_k:]
            top_idxs = part[np.argsort(row[part])[::-1]]
        results.append([(int(j), float(row[j])) for j in top_idxs])

    return results


def rerank_and_merge(
    source_names: list[str],
    candidates: list[list[tuple[int, float]]],
    target: TargetData,
    reranker: RerankerModel,
    *,
    k: int,
    embedding_weight: float,
    reranker_weight: float,
    batch_size: int,
) -> list[list[dict]]:
    """Rerank candidates with the cross-encoder and return top-k per item."""
    total_w = embedding_weight + reranker_weight

    # Flatten all (query_name, target_doc_text) pairs for batch scoring
    pairs: list[tuple[str, str]] = []
    for i, cands in enumerate(candidates):
        for target_idx, _ in cands:
            pairs.append((source_names[i], target.doc_texts[target_idx]))

    reranker_scores = reranker.score_pairs(pairs, batch_size=batch_size)

    # Regroup into per-item neighbour lists, sorted by final score
    all_neighbors: list[list[dict]] = []
    pos = 0
    for cands in candidates:
        neighbors: list[dict] = []
        for target_idx, emb_score in cands:
            rr_score = reranker_scores[pos]
            final = (embedding_weight * emb_score + reranker_weight * rr_score) / total_w
            neighbors.append({
                "food_id": int(target.food_ids[target_idx]),
                "name": target.names[target_idx],
                "embedding_score": round(emb_score, 6),
                "reranker_score": round(rr_score, 6),
                "final_score": round(final, 6),
            })
            pos += 1
        neighbors.sort(key=lambda x: x["final_score"], reverse=True)
        all_neighbors.append(neighbors[:k])

    return all_neighbors


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_partial(path: Path) -> dict[str, dict]:
    """Load an in-progress output file, returning {} if absent or corrupt."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Resuming from %s (%d items already done)", path, len(data))
        return data
    except Exception:
        logger.warning("Could not parse partial file %s; starting fresh", path)
        return {}


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map food items from one database to another via nearest-neighbour lookup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.toml", help="App config file")
    parser.add_argument("--source", default="myfitnesspal", help="Source database name")
    parser.add_argument("--target", default="usda", help="Target database name")
    parser.add_argument("--k", type=int, default=5, help="Neighbours to keep per item")
    parser.add_argument(
        "--embedding-weight", type=float, default=1.0,
        help="Weight for embedding similarity score",
    )
    parser.add_argument(
        "--reranker-weight", type=float, default=1.0,
        help="Weight for reranker score",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Reranker batch size (default: from config)",
    )
    parser.add_argument(
        "--retrieval-batch-size", type=int, default=1024,
        help="Source items per embedding-retrieval batch",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=500,
        help="Save partial output every N source items",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from partial output")
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: data/similarity/<source>_to_<target>.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Reranker device: %s", resolve_device(config.scoring.reranker.device))

    index_base = Path(config.index.path)
    source_dir = index_base / args.source
    target_dir = index_base / args.target

    out_path = (
        Path(args.out) if args.out
        else Path("data/similarity") / f"{args.source}_to_{args.target}.json"
    )
    partial_path = out_path.with_suffix(".partial.json")

    candidate_k = 2 * args.k

    # Load target database fully (small enough to fit in memory)
    target = load_target(target_dir)

    # Load existing progress if resuming
    results: dict[str, dict] = _load_partial(partial_path) if args.resume else {}
    already_done = {int(k) for k in results.keys()}

    # Count total source items to report progress
    source_total = len(pl.read_parquet(str(source_dir / "metadata.parquet"), columns=["row_id"]))
    n_pending = source_total - len(already_done)
    logger.info(
        "%d items to process (%d already done, %d total)",
        n_pending, len(already_done), source_total,
    )

    if n_pending == 0:
        logger.info("Nothing to do — writing final output.")
        _save_json(results, out_path)
        return

    # Load reranker
    reranker = RerankerModel(config.scoring.reranker, device=config.scoring.reranker.device)
    reranker_batch = args.batch_size or config.scoring.reranker.batch_size

    t0 = time.time()
    processed = 0
    items_since_checkpoint = 0

    for batch_ids, batch_names, batch_embs in iter_source_batches(
        source_dir, args.retrieval_batch_size, already_done=already_done,
    ):
        # Stage 1: fast embedding retrieval → 2*k candidates per item
        candidates = find_candidates(batch_embs, target, candidate_k)

        # Stage 2: cross-encoder reranker → top k
        neighbor_lists = rerank_and_merge(
            batch_names,
            candidates,
            target,
            reranker,
            k=args.k,
            embedding_weight=args.embedding_weight,
            reranker_weight=args.reranker_weight,
            batch_size=reranker_batch,
        )

        for i, neighbors in enumerate(neighbor_lists):
            results[str(int(batch_ids[i]))] = {
                "name": batch_names[i],
                "neighbors": neighbors,
            }

        processed += len(batch_ids)
        items_since_checkpoint += len(batch_ids)

        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = n_pending - processed
        eta = remaining / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d | %.1f items/s | ETA %.0fs",
            len(already_done) + processed, source_total, rate, eta,
        )

        if items_since_checkpoint >= args.checkpoint_every:
            _save_json(results, partial_path)
            items_since_checkpoint = 0

    # Write final output and clean up partial file
    _save_json(results, out_path)
    logger.info("Saved mapping → %s", out_path)
    if partial_path.exists():
        partial_path.unlink()

    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        "Done. %d source → %d target | top-%d | %.1f MB | elapsed %.1fs",
        len(results), len(target.food_ids), args.k, size_mb, time.time() - t0,
    )


if __name__ == "__main__":
    main()
