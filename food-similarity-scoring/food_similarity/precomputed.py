"""Load and query pre-computed similarity and mapping data."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from food_similarity.signals import MACRO_FIELDS

logger = logging.getLogger(__name__)

PER_PAGE = 50


@dataclass
class _SimilarityData:
    source: str
    foods: dict[str, dict]  # food_id_str -> {name, neighbors}
    names: list[tuple[str, str]]  # (food_id_str, name_lower) for search


@dataclass
class _MappingData:
    source: str
    target: str
    foods: dict[str, dict]
    names: list[tuple[str, str]]


def _build_name_index(data: dict[str, dict]) -> list[tuple[str, str]]:
    return [(fid, entry["name"].lower()) for fid, entry in data.items()]


def _macros_for_row(row: dict) -> dict[str, float | None]:
    return {f: row.get(f) for f in MACRO_FIELDS}


class PrecomputedStore:
    """Discovers, loads, and queries pre-computed similarity/mapping JSON files."""

    def __init__(self, data_dir: Path) -> None:
        self.similarity_dir = data_dir / "similarity"
        self.index_dir = data_dir / "index-v1"
        self.similarities: dict[str, _SimilarityData] = {}
        self.mappings: dict[str, _MappingData] = {}
        # Polars DataFrames keyed by source name for macro enrichment
        self.metadata: dict[str, pl.DataFrame] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        if not self.similarity_dir.exists():
            logger.info("No similarity directory at %s — skipping", self.similarity_dir)
            return

        for path in sorted(self.similarity_dir.glob("*_refined.json")):
            source = path.stem.removesuffix("_refined")
            self._load_similarity(source, path)

        for path in sorted(self.similarity_dir.glob("*_to_*.json")):
            stem = path.stem
            parts = stem.split("_to_")
            if len(parts) == 2:
                self._load_mapping(parts[0], parts[1], path)

    def unload(self) -> None:
        """Release all loaded data to free memory."""
        self.similarities.clear()
        self.mappings.clear()
        self.metadata.clear()
        logger.info("PrecomputedStore unloaded to free memory")

    @property
    def is_loaded(self) -> bool:
        return bool(self.similarities or self.mappings)

    def _load_similarity(self, source: str, path: Path) -> None:
        logger.info("Loading similarity data: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
        with open(path) as f:
            data: dict[str, dict] = json.load(f)
        names = _build_name_index(data)
        self.similarities[source] = _SimilarityData(source, data, names)
        logger.info("  loaded %d foods for source %s", len(data), source)

        # Load index parquet for macro enrichment
        idx_path = self.similarity_dir / f"{source}_index.parquet"
        if idx_path.exists():
            self._ensure_metadata(source, idx_path, id_col="food_id")

    def _load_mapping(self, source: str, target: str, path: Path) -> None:
        logger.info("Loading mapping data: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
        with open(path) as f:
            data: dict[str, dict] = json.load(f)
        names = _build_name_index(data)
        key = f"{source}_to_{target}"
        self.mappings[key] = _MappingData(source, target, data, names)
        logger.info("  loaded %d foods for mapping %s -> %s", len(data), source, target)

        # Ensure metadata for source and target
        for src in (source, target):
            if src not in self.metadata:
                # Try similarity index parquet first, then full metadata
                idx_path = self.similarity_dir / f"{src}_index.parquet"
                meta_path = self.index_dir / src / "metadata.parquet"
                if idx_path.exists():
                    self._ensure_metadata(src, idx_path, id_col="food_id")
                elif meta_path.exists():
                    self._ensure_metadata(src, meta_path, id_col="row_id")

    def _ensure_metadata(self, source: str, path: Path, *, id_col: str) -> None:
        if source in self.metadata:
            return
        logger.info("Loading metadata: %s", path.name)
        df = pl.read_parquet(path)
        # Normalize id column to "food_id"
        if id_col != "food_id" and id_col in df.columns:
            df = df.rename({id_col: "food_id"})
        # Ensure we have name column
        if "name" not in df.columns and "product_name" in df.columns:
            df = df.rename({"product_name": "name"})
        self.metadata[source] = df
        logger.info("  metadata for %s: %d rows", source, len(df))

    # ------------------------------------------------------------------
    # Queries — Similarity
    # ------------------------------------------------------------------

    def available_similarities(self) -> list[str]:
        return sorted(self.similarities.keys())

    def search_similarity(
        self, source: str, query: str, page: int = 1
    ) -> tuple[list[dict], int, int]:
        """Return (results, total_count, total_pages) for list mode."""
        sim = self.similarities.get(source)
        if sim is None:
            return [], 0, 0
        q = query.lower().strip()
        if q:
            matches = [(fid, sim.foods[fid]["name"]) for fid, name in sim.names if q in name]
        else:
            matches = [(fid, sim.foods[fid]["name"]) for fid, _ in sim.names]
        total = len(matches)
        total_pages = max(1, math.ceil(total / PER_PAGE))
        start = (page - 1) * PER_PAGE
        page_items = matches[start : start + PER_PAGE]
        results = []
        for fid, name in page_items:
            row = {"food_id": fid, "name": name}
            row["macros"] = self._get_macros(source, int(fid))
            row["categories"] = self._get_categories(source, int(fid))
            results.append(row)
        return results, total, total_pages

    def get_similarity_detail(self, source: str, food_id: str) -> dict | None:
        sim = self.similarities.get(source)
        if sim is None or food_id not in sim.foods:
            return None
        entry = sim.foods[food_id]
        fid_int = int(food_id)
        result = {
            "food_id": food_id,
            "name": entry["name"],
            "macros": self._get_macros(source, fid_int),
            "categories": self._get_categories(source, fid_int),
            "neighbors": [],
        }
        for nb in entry["neighbors"]:
            nb_id = nb["food_id"]
            result["neighbors"].append({
                "food_id": str(nb_id),
                "name": nb["name"],
                "original_score": round(nb["original_score"], 4),
                "reranker_score": round(nb["reranker_score"], 4),
                "final_score": round(nb["final_score"], 4),
                "macros": self._get_macros(source, nb_id),
                "categories": self._get_categories(source, nb_id),
            })
        return result

    # ------------------------------------------------------------------
    # Queries — Mapping
    # ------------------------------------------------------------------

    def available_mappings(self) -> list[tuple[str, str]]:
        return sorted(
            (m.source, m.target) for m in self.mappings.values()
        )

    def search_mapping(
        self, source: str, target: str, query: str, page: int = 1
    ) -> tuple[list[dict], int, int]:
        key = f"{source}_to_{target}"
        mp = self.mappings.get(key)
        if mp is None:
            return [], 0, 0
        q = query.lower().strip()
        if q:
            matches = [(fid, mp.foods[fid]["name"]) for fid, name in mp.names if q in name]
        else:
            matches = [(fid, mp.foods[fid]["name"]) for fid, _ in mp.names]
        total = len(matches)
        total_pages = max(1, math.ceil(total / PER_PAGE))
        start = (page - 1) * PER_PAGE
        page_items = matches[start : start + PER_PAGE]
        results = []
        for fid, name in page_items:
            row = {"food_id": fid, "name": name}
            row["macros"] = self._get_macros(source, int(fid))
            results.append(row)
        return results, total, total_pages

    def get_mapping_detail(self, source: str, target: str, food_id: str) -> dict | None:
        key = f"{source}_to_{target}"
        mp = self.mappings.get(key)
        if mp is None or food_id not in mp.foods:
            return None
        entry = mp.foods[food_id]
        fid_int = int(food_id)
        result = {
            "food_id": food_id,
            "name": entry["name"],
            "macros": self._get_macros(source, fid_int),
            "categories": self._get_categories(source, fid_int),
            "neighbors": [],
        }
        for nb in entry["neighbors"]:
            nb_id = nb["food_id"]
            result["neighbors"].append({
                "food_id": str(nb_id),
                "name": nb["name"],
                "embedding_score": round(nb["embedding_score"], 4),
                "reranker_score": round(nb["reranker_score"], 4),
                "final_score": round(nb["final_score"], 4),
                "macros": self._get_macros(target, nb_id),
                "categories": self._get_categories(target, nb_id),
            })
        return result

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _get_macros(self, source: str, food_id: int) -> dict[str, float | None]:
        df = self.metadata.get(source)
        if df is None:
            return {f: None for f in MACRO_FIELDS}
        row = df.filter(pl.col("food_id") == food_id)
        if row.is_empty():
            return {f: None for f in MACRO_FIELDS}
        r = row.row(0, named=True)
        return {f: round(r[f], 2) if r.get(f) is not None else None for f in MACRO_FIELDS}

    def _get_categories(self, source: str, food_id: int) -> list[str]:
        df = self.metadata.get(source)
        if df is None:
            return []
        cat_col = "categories" if "categories" in df.columns else "categories_tags"
        if cat_col not in df.columns:
            return []
        row = df.filter(pl.col("food_id") == food_id)
        if row.is_empty():
            return []
        val = row[cat_col][0]
        if val is None:
            return []
        return val.to_list() if hasattr(val, "to_list") else list(val)
