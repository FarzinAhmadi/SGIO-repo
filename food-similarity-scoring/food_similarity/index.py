from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index: faiss.Index | None = None

    def create(self) -> None:
        """Create an empty IndexFlatIP ready for incremental .add() calls."""
        logger.info("Creating empty FAISS IndexFlatIP (dim=%d)", self.dimension)
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.omp_set_num_threads(16)

    def add(self, embeddings: np.ndarray) -> None:
        """Add a batch of vectors to the index."""
        embeddings = embeddings.astype(np.float32)
        assert embeddings.shape[1] == self.dimension
        self.index.add(embeddings)
        logger.info(
            "Added %d vectors (total: %d)", len(embeddings), self.index.ntotal
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("Index saved to %s", path)

    def load(self, path: str | Path, device: str = "cpu") -> None:
        path = Path(path)
        logger.info("Loading FAISS index from %s", path)
        self.index = faiss.read_index(str(path))
        faiss.omp_set_num_threads(16)

        if device == "cuda":
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception:
                logger.warning("Failed to move FAISS index to GPU, using CPU")

        logger.info("Index loaded: %d vectors", self.index.ntotal)

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices) arrays of shape (top_k,)."""
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        return scores[0], indices[0]


class MetadataStore:
    def __init__(self) -> None:
        self.df: pl.DataFrame | None = None
        self.row_id_to_idx: dict[int, int] = {}
        self.row_count: int = 0

    def save(self, df: pl.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(path))
        logger.info("Metadata saved to %s (%d rows)", path, len(df))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.df = pl.read_parquet(str(path))
        row_ids = self.df.get_column("row_id").to_list()
        self.row_id_to_idx = {int(row_id): idx for idx, row_id in enumerate(row_ids)}
        self.row_count = len(row_ids)
        logger.info("Metadata loaded: %d rows", self.row_count)

    def lookup(self, row_ids: np.ndarray) -> list[dict]:
        """Look up metadata rows by FAISS row IDs."""
        if self.df is None:
            raise RuntimeError("Metadata store is not loaded")
        row_indices: list[int] = []
        for row_id in row_ids.tolist():
            if row_id < 0:
                continue
            row_index = self.row_id_to_idx.get(int(row_id))
            if row_index is not None:
                row_indices.append(row_index)
        if not row_indices:
            return []
        return self.df[row_indices].to_dicts()

    def get(self, row_id: int) -> dict | None:
        if self.df is None:
            raise RuntimeError("Metadata store is not loaded")
        row_index = self.row_id_to_idx.get(row_id)
        if row_index is None:
            return None
        return self.df.row(row_index, named=True)

    def unique_tags(self, field_name: str, limit: int = 500) -> list[str]:
        """Return the most common tag values for a list column."""
        if self.df is None:
            raise RuntimeError("Metadata store is not loaded")
        if field_name not in self.df.columns:
            return []
        col = self.df.select(pl.col(field_name))
        tags = (
            col.explode(field_name)
            .drop_nulls()
            .group_by(field_name)
            .len()
            .sort("len", descending=True)
            .head(limit)
            .get_column(field_name)
            .to_list()
        )
        return sorted(tags)
