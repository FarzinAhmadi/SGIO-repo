"""Refine the top-k similarity neighbours for each food item using the reranker.

Loads the pre-built pairwise similarity matrix, selects the top-k candidates
per food item (excluding the item itself), then scores every (query, candidate)
pair with the cross-encoder reranker.  The final score is a weighted average
of the original similarity score and the reranker score.

The result is written as a JSON file mapping each food_id (string key) to its
re-ranked neighbour list, e.g.:

    {
      "0": {
        "name": "Milk, human",
        "neighbors": [
          {"food_id": 1, "name": "Milk, NFS",
           "original_score": 0.9541, "reranker_score": 0.87, "final_score": 0.9121},
          ...
        ]
      },
      ...
    }

Usage:
    uv run python -m scripts.refine_similarity --source usda
    uv run python -m scripts.refine_similarity --source usda --resume
    uv run python -m scripts.refine_similarity --source usda --top-k 10 \\
        --original-weight 1.0 --reranker-weight 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from food_similarity.config import RerankerScoringConfig, load_config, resolve_device
from food_similarity.reranker import RerankerModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_matrix(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (similarity float32 N×N, food_ids int64 N)."""
    d = np.load(str(npz_path))
    sim = d["similarity"].astype(np.float32)
    food_ids = d["food_ids"].astype(np.int64)
    logger.info("Loaded similarity matrix %s from %s", sim.shape, npz_path)
    return sim, food_ids


def _load_metadata(meta_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(str(meta_path))
    logger.info("Loaded %d food items from %s", len(df), meta_path)
    return df


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
# Top-k selection
# ---------------------------------------------------------------------------

def top_k_indices(sim_row: np.ndarray, self_idx: int, k: int) -> np.ndarray:
    """Return indices of the k highest-scoring items, excluding self_idx."""
    row = sim_row.copy()
    row[self_idx] = -1.0
    # argpartition is O(N) vs argsort's O(N log N), then sort only the small slice
    part = np.argpartition(row, -k)[-k:]
    return part[np.argsort(row[part])[::-1]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine top-k food similarity with the cross-encoder reranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.toml", help="App config file")
    parser.add_argument("--source", default="usda", help="Source name")
    parser.add_argument(
        "--similarity-dir", default=None,
        help="Directory containing <source>.npz (default: data/similarity)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Neighbours per item")
    parser.add_argument(
        "--original-weight", type=float, default=1.0,
        help="Weight for the original (matrix) similarity score",
    )
    parser.add_argument(
        "--reranker-weight", type=float, default=1.0,
        help="Weight for the reranker score",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Reranker batch size (default: from config)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=200,
        help="Save partial output every N food items",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing partial output file",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: data/similarity/<source>_refined.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Reranker device: %s", resolve_device(config.scoring.reranker.device))

    sim_dir = Path(args.similarity_dir) if args.similarity_dir else Path("data/similarity")
    index_dir = Path(config.index.path) / args.source

    npz_path = sim_dir / f"{args.source}.npz"
    meta_path = index_dir / "metadata.parquet"
    out_path = Path(args.out) if args.out else sim_dir / f"{args.source}_refined.json"
    partial_path = out_path.with_suffix(".partial.json")

    # Load data
    sim_matrix, food_ids = _load_matrix(npz_path)
    df = _load_metadata(meta_path)

    # Build fast lookup: food_id → (name, document_text)
    id_to_name: dict[int, str] = {}
    id_to_doc: dict[int, str] = {}
    for row in df.iter_rows(named=True):
        fid = int(row["row_id"])
        id_to_name[fid] = row["product_name"]
        id_to_doc[fid] = row["document_text"]

    n = len(food_ids)
    top_k = args.top_k

    # Load existing progress if resuming
    results: dict[str, dict] = _load_partial(partial_path) if args.resume else {}
    already_done: set[str] = set(results.keys())

    # Determine which items still need processing
    pending_indices = [
        i for i in range(n) if str(int(food_ids[i])) not in already_done
    ]
    logger.info(
        "%d items to process (%d already done, %d total)",
        len(pending_indices), len(already_done), n,
    )

    if not pending_indices:
        logger.info("Nothing to do — writing final output.")
        _save_json(results, out_path)
        return

    # Load reranker
    reranker = RerankerModel(config.scoring.reranker, device=config.scoring.reranker.device)
    batch_size = args.batch_size or config.scoring.reranker.batch_size

    orig_w = args.original_weight
    rerank_w = args.reranker_weight
    total_w = orig_w + rerank_w

    t0 = time.time()
    processed = 0

    # Process in checkpoint-sized chunks
    for chunk_start in range(0, len(pending_indices), args.checkpoint_every):
        chunk = pending_indices[chunk_start : chunk_start + args.checkpoint_every]

        # Build flat list of (food_idx, neighbor_idx, original_score) for the chunk
        chunk_triplets: list[tuple[int, int, float]] = []
        for i in chunk:
            fid = int(food_ids[i])
            nbr_idxs = top_k_indices(sim_matrix[i], self_idx=i, k=top_k)
            for j in nbr_idxs:
                chunk_triplets.append((i, int(j), float(sim_matrix[i, j])))

        # Flat-batch all (query_name, doc_text) pairs in this chunk
        pairs: list[tuple[str, str]] = [
            (id_to_name[int(food_ids[i])], id_to_doc[int(food_ids[j])])
            for i, j, _ in chunk_triplets
        ]
        reranker_scores = reranker.score_pairs(pairs, batch_size=batch_size)

        # Regroup results by food item
        pos = 0
        for i in chunk:
            fid = int(food_ids[i])
            neighbors: list[dict] = []
            for _ in range(top_k):
                _, j, orig_score = chunk_triplets[pos]
                rr_score = reranker_scores[pos]
                final_score = (orig_w * orig_score + rerank_w * rr_score) / total_w
                neighbors.append({
                    "food_id": int(food_ids[j]),
                    "name": id_to_name[int(food_ids[j])],
                    "original_score": round(orig_score, 6),
                    "reranker_score": round(rr_score, 6),
                    "final_score": round(final_score, 6),
                })
                pos += 1

            # Re-sort by final score descending
            neighbors.sort(key=lambda x: x["final_score"], reverse=True)
            results[str(fid)] = {
                "name": id_to_name[fid],
                "neighbors": neighbors,
            }

        processed += len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed
        remaining = len(pending_indices) - processed
        eta = remaining / rate if rate > 0 else 0

        logger.info(
            "Progress: %d/%d items | %.1f items/s | ETA %.0fs",
            len(already_done) + processed,
            n,
            rate,
            eta,
        )

        # Checkpoint
        _save_json(results, partial_path)

    # Write final output and clean up partial
    _save_json(results, out_path)
    logger.info("Saved refined similarity → %s", out_path)
    if partial_path.exists():
        partial_path.unlink()

    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        "Done. %d items × top-%d neighbours | %.1f MB | elapsed %.1fs",
        len(results), top_k, size_mb, time.time() - t0,
    )


if __name__ == "__main__":
    main()
