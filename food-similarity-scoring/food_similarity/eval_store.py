"""SQLite storage and sampling logic for manual food similarity evaluation."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
from pathlib import Path
from uuid import uuid4

import polars as pl

from food_similarity.precomputed import PrecomputedStore
from food_similarity.signals import MACRO_FIELDS

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    browser_cookie TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_cookie ON sessions(browser_cookie);

CREATE TABLE IF NOT EXISTS binary_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    query_food_id INTEGER NOT NULL,
    candidate_food_id INTEGER NOT NULL,
    candidate_rank INTEGER NOT NULL,
    response TEXT NOT NULL CHECK(response IN ('yes', 'no')),
    similarity_score REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bestswap_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    query_food_id INTEGER NOT NULL,
    candidate_food_ids TEXT NOT NULL,
    chosen_food_id INTEGER NOT NULL,
    candidate_ranks TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS likert_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    food_id_a INTEGER NOT NULL,
    food_id_b INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    similarity_bin INTEGER NOT NULL CHECK(similarity_bin BETWEEN 1 AND 5),
    similarity_score REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS goodswaps_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    query_food_id INTEGER NOT NULL,
    candidate_food_ids TEXT NOT NULL,
    chosen_food_ids TEXT NOT NULL,
    candidate_ranks TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class EvalStore:
    """Manages evaluation SQLite database and food pair sampling."""

    def __init__(self, db_path: Path, precomputed: PrecomputedStore) -> None:
        self._precomputed = precomputed
        self._lock = threading.Lock()
        self._loaded = False

        # Open database
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Initialize data fields (populated by _load_data)
        self._sim_data: dict[str, dict] = {}
        self._food_ids: list[str] = []
        self._metadata_df: pl.DataFrame | None = None
        self._fg_col: str = ""
        self._cat_col: str = ""
        self._food_groups_index: dict[str, list[int]] = {}
        self._all_groups: list[str] = []
        self._neighbor_set: dict[int, set[int]] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Build all in-memory lookup indexes from precomputed data."""
        precomputed = self._precomputed

        if "usda" not in precomputed.similarities:
            raise KeyError("USDA similarity data not loaded in PrecomputedStore")

        self._sim_data = precomputed.similarities["usda"].foods
        self._food_ids = list(self._sim_data.keys())

        # Load the full USDA metadata (7338 rows) instead of the 500-row
        # similarity index subset that PrecomputedStore uses
        full_meta_path = precomputed.index_dir / "usda" / "metadata.parquet"
        if full_meta_path.exists():
            self._metadata_df = pl.read_parquet(full_meta_path)
            logger.info("Loaded full USDA metadata: %d rows", len(self._metadata_df))
        else:
            self._metadata_df = precomputed.metadata["usda"]
            logger.warning("Full USDA metadata not found, using %d-row subset", len(self._metadata_df))

        # Normalize column names
        if "food_id" not in self._metadata_df.columns and "row_id" in self._metadata_df.columns:
            self._metadata_df = self._metadata_df.rename({"row_id": "food_id"})
        if "name" not in self._metadata_df.columns and "product_name" in self._metadata_df.columns:
            self._metadata_df = self._metadata_df.rename({"product_name": "name"})
        self._fg_col = "food_groups_tags" if "food_groups_tags" in self._metadata_df.columns else "food_groups"
        self._cat_col = "categories_tags" if "categories_tags" in self._metadata_df.columns else "categories"

        # food_group -> [food_id ints]
        self._food_groups_index = {}
        if self._fg_col in self._metadata_df.columns:
            for row in self._metadata_df.select(["food_id", self._fg_col]).iter_rows(named=True):
                fid = row["food_id"]
                groups = row[self._fg_col]
                if groups is not None:
                    group_list = groups if isinstance(groups, list) else groups.to_list()
                    for g in group_list:
                        if g:
                            self._food_groups_index.setdefault(g, []).append(fid)
        self._all_groups = [g for g, ids in self._food_groups_index.items() if len(ids) >= 2]

        # food_id int -> set of neighbor food_id ints
        self._neighbor_set = {}
        for fid_str, entry in self._sim_data.items():
            fid = int(fid_str)
            self._neighbor_set[fid] = {nb["food_id"] for nb in entry["neighbors"]}

        self._loaded = True
        logger.info(
            "EvalStore data loaded: %d foods, %d food groups",
            len(self._food_ids), len(self._all_groups),
        )

    def unload(self) -> None:
        """Release in-memory data to free memory during idle eviction."""
        with self._lock:
            self._sim_data = {}
            self._food_ids = []
            self._metadata_df = None
            self._food_groups_index = {}
            self._all_groups = []
            self._neighbor_set = {}
            self._loaded = False
        logger.info("EvalStore data unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Reload data if it was evicted. PrecomputedStore must be reloaded first."""
        with self._lock:
            if self._loaded:
                return
            self._load_data()

    # ------------------------------------------------------------------
    # Food info helper
    # ------------------------------------------------------------------

    def _food_info(self, food_id: int) -> dict:
        fid_str = str(food_id)
        entry = self._sim_data.get(fid_str)
        name = entry["name"] if entry else f"Food #{food_id}"
        macros = self._get_macros(food_id)
        categories = self._get_list_field(food_id, self._cat_col)
        food_groups = self._get_list_field(food_id, self._fg_col)
        return {
            "food_id": food_id,
            "name": name,
            "macros": macros,
            "categories": categories,
            "food_groups": food_groups,
        }

    def _get_macros(self, food_id: int) -> dict[str, float | None]:
        row = self._metadata_df.filter(pl.col("food_id") == food_id)
        if row.is_empty():
            return {f: None for f in MACRO_FIELDS}
        r = row.row(0, named=True)
        return {f: round(r[f], 2) if r.get(f) is not None else None for f in MACRO_FIELDS}

    def _get_list_field(self, food_id: int, col: str) -> list[str]:
        if col not in self._metadata_df.columns:
            return []
        row = self._metadata_df.filter(pl.col("food_id") == food_id)
        if row.is_empty():
            return []
        val = row[col][0]
        if val is None:
            return []
        return val.to_list() if hasattr(val, "to_list") else list(val)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_binary(self) -> dict:
        query_fid = random.choice(self._food_ids)
        neighbors = self._sim_data[query_fid]["neighbors"]
        top_k = min(5, len(neighbors))
        idx = random.randint(0, top_k - 1)
        nb = neighbors[idx]
        return {
            "query": self._food_info(int(query_fid)),
            "candidate": self._food_info(nb["food_id"]),
            "candidate_rank": idx + 1,
            "similarity_score": round(nb["final_score"], 4),
        }

    def sample_bestswap(self) -> dict:
        query_fid = random.choice(self._food_ids)
        neighbors = self._sim_data[query_fid]["neighbors"]
        n = min(10, len(neighbors))
        indices = random.sample(range(n), min(4, n))
        candidates = []
        for idx in indices:
            nb = neighbors[idx]
            info = self._food_info(nb["food_id"])
            info["rank"] = idx + 1
            info["similarity_score"] = round(nb["final_score"], 4)
            candidates.append(info)
        random.shuffle(candidates)
        return {
            "query": self._food_info(int(query_fid)),
            "candidates": candidates,
        }

    def sample_goodswaps(self) -> dict:
        """Like bestswap but presents top-10 neighbors for multi-select."""
        query_fid = random.choice(self._food_ids)
        neighbors = self._sim_data[query_fid]["neighbors"][:10]
        candidates = []
        for idx, nb in enumerate(neighbors):
            info = self._food_info(nb["food_id"])
            info["rank"] = idx + 1
            info["similarity_score"] = round(nb["final_score"], 4)
            candidates.append(info)
        random.shuffle(candidates)
        return {
            "query": self._food_info(int(query_fid)),
            "candidates": candidates,
        }

    def sample_likert(self) -> dict:
        bin_choice = random.randint(1, 5)

        if bin_choice >= 3:
            # Bins 3-5: pick a food and one of its neighbors
            query_fid = random.choice(self._food_ids)
            neighbors = self._sim_data[query_fid]["neighbors"]
            if bin_choice == 5:
                lo, hi = 0, min(2, len(neighbors))
            elif bin_choice == 4:
                lo, hi = min(2, len(neighbors)), min(5, len(neighbors))
            else:  # bin 3
                lo, hi = min(5, len(neighbors)), min(10, len(neighbors))
            # Fallback if range is empty
            if lo >= hi:
                lo, hi = 0, min(2, len(neighbors))
            idx = random.randint(lo, hi - 1)
            nb = neighbors[idx]
            food_a = self._food_info(int(query_fid))
            food_b = self._food_info(nb["food_id"])
            score = round(nb["final_score"], 4)
        elif bin_choice == 2:
            # Same food group, not neighbors
            food_a, food_b, score = self._sample_same_group_pair()
        else:
            # Different food groups, not neighbors
            food_a, food_b, score = self._sample_different_group_pair()

        # Randomize order so evaluators can't guess which is "query"
        if random.random() < 0.5:
            food_a, food_b = food_b, food_a

        return {
            "food_a": food_a,
            "food_b": food_b,
            "similarity_bin": bin_choice,
            "similarity_score": score,
        }

    def _sample_same_group_pair(self) -> tuple[dict, dict, float | None]:
        for _ in range(20):
            group = random.choice(self._all_groups)
            members = self._food_groups_index[group]
            if len(members) < 2:
                continue
            a, b = random.sample(members, 2)
            if b not in self._neighbor_set.get(a, set()):
                return self._food_info(a), self._food_info(b), None
        # Fallback: just return any same-group pair
        group = random.choice(self._all_groups)
        a, b = random.sample(self._food_groups_index[group], 2)
        return self._food_info(a), self._food_info(b), None

    def _sample_different_group_pair(self) -> tuple[dict, dict, float | None]:
        for _ in range(20):
            g1, g2 = random.sample(self._all_groups, 2)
            a = random.choice(self._food_groups_index[g1])
            b = random.choice(self._food_groups_index[g2])
            if b not in self._neighbor_set.get(a, set()):
                return self._food_info(a), self._food_info(b), None
        # Fallback
        g1, g2 = random.sample(self._all_groups, 2)
        a = random.choice(self._food_groups_index[g1])
        b = random.choice(self._food_groups_index[g2])
        return self._food_info(a), self._food_info(b), None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_or_get_session(self, username: str, browser_cookie: str) -> str:
        with self._lock:
            cur = self._conn.execute(
                "SELECT session_id, username FROM sessions WHERE browser_cookie = ?",
                (browser_cookie,),
            )
            row = cur.fetchone()
            if row:
                session_id, existing_name = row
                if existing_name != username:
                    self._conn.execute(
                        "UPDATE sessions SET username = ? WHERE session_id = ?",
                        (username, session_id),
                    )
                    self._conn.commit()
                return session_id
            session_id = str(uuid4())
            self._conn.execute(
                "INSERT INTO sessions (session_id, username, browser_cookie) VALUES (?, ?, ?)",
                (session_id, username, browser_cookie),
            )
            self._conn.commit()
            return session_id

    # ------------------------------------------------------------------
    # Recording ratings
    # ------------------------------------------------------------------

    def record_binary(
        self,
        session_id: str,
        query_food_id: int,
        candidate_food_id: int,
        candidate_rank: int,
        response: str,
        similarity_score: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO binary_ratings
                   (session_id, query_food_id, candidate_food_id, candidate_rank, response, similarity_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, query_food_id, candidate_food_id, candidate_rank, response, similarity_score),
            )
            self._conn.commit()

    def record_bestswap(
        self,
        session_id: str,
        query_food_id: int,
        candidate_food_ids: list[int],
        chosen_food_id: int,
        candidate_ranks: list[int],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO bestswap_ratings
                   (session_id, query_food_id, candidate_food_ids, chosen_food_id, candidate_ranks)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, query_food_id, json.dumps(candidate_food_ids), chosen_food_id, json.dumps(candidate_ranks)),
            )
            self._conn.commit()

    def record_likert(
        self,
        session_id: str,
        food_id_a: int,
        food_id_b: int,
        rating: int,
        similarity_bin: int,
        similarity_score: float | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO likert_ratings
                   (session_id, food_id_a, food_id_b, rating, similarity_bin, similarity_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, food_id_a, food_id_b, rating, similarity_bin, similarity_score),
            )
            self._conn.commit()

    def record_goodswaps(
        self,
        session_id: str,
        query_food_id: int,
        candidate_food_ids: list[int],
        chosen_food_ids: list[int],
        candidate_ranks: list[int],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO goodswaps_ratings
                   (session_id, query_food_id, candidate_food_ids, chosen_food_ids, candidate_ranks)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, query_food_id, json.dumps(candidate_food_ids),
                 json.dumps(chosen_food_ids), json.dumps(candidate_ranks)),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, session_id: str) -> dict:
        counts = {}
        for table in ("binary_ratings", "bestswap_ratings", "likert_ratings", "goodswaps_ratings"):
            cur = self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id = ?",  # noqa: S608
                (session_id,),
            )
            key = table.replace("_ratings", "_count")
            counts[key] = cur.fetchone()[0]
        return counts

    def close(self) -> None:
        self._conn.close()
