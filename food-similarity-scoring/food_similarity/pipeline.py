from __future__ import annotations

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from food_similarity.config import AppConfig, resolve_device
from food_similarity.embedding import EmbeddingModel
from food_similarity.index import FaissIndex, MetadataStore
from food_similarity.llm import LlmBackend, create_llm_backend
from food_similarity.reranker import RerankerModel
from food_similarity.search_request import SearchRequest
from food_similarity.signals import (
    DEFAULT_MACRO_SCALES,
    LIST_METADATA_FIELDS,
    MACRO_FIELDS,
)

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    row_id: int
    product_name: str
    brands: str
    categories: list[str]
    metadata_lists: dict[str, set[str]]
    metadata_scalars: dict[str, str | int | None]
    macros: dict[str, float | None]
    document_text: str
    source: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0


@dataclass
class SourceIndex:
    name: str
    faiss_index: FaissIndex
    metadata_store: MetadataStore
    macro_scales: dict[str, float]


class Scorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def weight(self) -> float: ...

    def is_active(self, request: SearchRequest) -> bool:
        return True

    @abstractmethod
    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]: ...


class EmbeddingRetriever:
    """First stage: fast embedding-based retrieval via FAISS."""

    def __init__(
        self, embedding_model: EmbeddingModel, faiss_index: FaissIndex,
        metadata_store: MetadataStore, top_k: int, source: str = "",
    ) -> None:
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.metadata_store = metadata_store
        self.top_k = top_k
        self.source = source

    def retrieve(self, query: str, query_vec: np.ndarray | None = None) -> list[Candidate]:
        if query_vec is None:
            query_vec = self.embedding_model.encode_query(query)
        scores, indices = self.faiss_index.search(query_vec, self.top_k)
        candidates = []
        for score, row_id in zip(scores.tolist(), indices.tolist()):
            if row_id < 0:
                continue
            row = self.metadata_store.get(int(row_id))
            if row is None:
                continue
            cats = _normalize_tag_list(row.get("categories_tags"))
            metadata_lists = {
                field_name: _normalize_tag_list(row.get(field_name))
                for field_name in LIST_METADATA_FIELDS
            }
            metadata_scalars: dict[str, str | int | None] = {
                "nutriscore_grade": _normalize_scalar_grade(
                    row.get("nutriscore_grade")
                ),
                "nova_group": _coerce_int(row.get("nova_group")),
            }
            macros = {field_name: _coerce_float(row.get(field_name)) for field_name in MACRO_FIELDS}
            candidate = Candidate(
                row_id=row["row_id"],
                product_name=row["product_name"],
                brands=row.get("brands", ""),
                categories=sorted(cats),
                metadata_lists=metadata_lists,
                metadata_scalars=metadata_scalars,
                macros=macros,
                document_text=row["document_text"],
                source=self.source,
                scores={"embedding": float(score)},
            )
            candidates.append(candidate)
        return candidates


class RerankerScorer(Scorer):
    """Second stage: cross-encoder reranking."""

    def __init__(self, reranker_model: RerankerModel, weight: float) -> None:
        self._reranker = reranker_model
        self._weight = weight

    @property
    def name(self) -> str:
        return "reranker"

    @property
    def weight(self) -> float:
        return self._weight

    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]:
        documents = [c.document_text for c in candidates]
        reranker_scores = self._reranker.score(request.query, documents)
        for candidate, s in zip(candidates, reranker_scores):
            candidate.scores["reranker"] = s
        return candidates


class MacronutrientScorer(Scorer):
    def __init__(
        self, *, weight: float, fields: tuple[str, ...], scales: dict[str, float]
    ) -> None:
        self._weight = weight
        self._fields = fields
        self._scales = scales

    @property
    def name(self) -> str:
        return "macronutrient"

    @property
    def weight(self) -> float:
        return self._weight

    def is_active(self, request: SearchRequest) -> bool:
        return bool(set(request.macro_targets) & set(self._fields))

    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]:
        active_targets = {
            field_name: value
            for field_name, value in request.macro_targets.items()
            if field_name in self._fields
        }
        for candidate in candidates:
            parts: list[float] = []
            for field_name, target in active_targets.items():
                value = candidate.macros.get(field_name)
                if value is None:
                    continue
                scale = self._scales.get(field_name, DEFAULT_MACRO_SCALES[field_name])
                distance = abs(value - target) / max(scale, 1e-6)
                parts.append(1.0 / (1.0 + distance))
            candidate.scores[self.name] = float(np.mean(parts)) if parts else 0.0
        return candidates


class MetadataScorer(Scorer):
    def __init__(
        self, *, weight: float, list_fields: tuple[str, ...], scalar_fields: tuple[str, ...]
    ) -> None:
        self._weight = weight
        self._list_fields = list_fields
        self._scalar_fields = scalar_fields

    @property
    def name(self) -> str:
        return "metadata"

    @property
    def weight(self) -> float:
        return self._weight

    def is_active(self, request: SearchRequest) -> bool:
        has_list_fields = bool(set(request.metadata_tags) & set(self._list_fields))
        has_scalar_fields = bool(
            set(request.metadata_scalars) & set(self._scalar_fields)
        )
        return has_list_fields or has_scalar_fields

    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]:
        active_list_criteria = {
            field_name: tags
            for field_name, tags in request.metadata_tags.items()
            if field_name in self._list_fields and tags
        }
        active_scalar_criteria = {
            field_name: value
            for field_name, value in request.metadata_scalars.items()
            if field_name in self._scalar_fields
        }

        for candidate in candidates:
            parts: list[float] = []
            for field_name, query_tags in active_list_criteria.items():
                candidate_tags = candidate.metadata_lists.get(field_name, set())
                if not candidate_tags:
                    parts.append(0.0)
                    continue
                overlap = len(candidate_tags & query_tags)
                parts.append(overlap / len(query_tags))

            for field_name, query_value in active_scalar_criteria.items():
                candidate_value = candidate.metadata_scalars.get(field_name)
                parts.append(1.0 if candidate_value == query_value else 0.0)

            candidate.scores[self.name] = float(np.mean(parts)) if parts else 0.0
        return candidates


class EditDistanceScorer(Scorer):
    def __init__(self, weight: float) -> None:
        self._weight = weight

    @property
    def name(self) -> str:
        return "edit_distance"

    @property
    def weight(self) -> float:
        return self._weight

    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]:
        query_lower = request.query.strip().lower()
        for candidate in candidates:
            name_lower = candidate.product_name.strip().lower()
            candidate.scores[self.name] = SequenceMatcher(
                None, query_lower, name_lower
            ).ratio()
        return candidates


class LlmScorer(Scorer):
    def __init__(self, llm_backend: LlmBackend, weight: float) -> None:
        self._llm = llm_backend
        self._weight = weight

    @property
    def name(self) -> str:
        return "llm"

    @property
    def weight(self) -> float:
        return self._weight

    def score(
        self, request: SearchRequest, candidates: list[Candidate]
    ) -> list[Candidate]:
        product_names = [c.product_name for c in candidates]
        llm_scores = self._llm.score(request.query, product_names)
        for candidate, s in zip(candidates, llm_scores):
            candidate.scores["llm"] = s
        return candidates


def _effective_weight(
    name: str, config_weight: float, overrides: dict[str, float],
) -> float:
    """Return the per-request weight for a scorer, falling back to config."""
    return overrides.get(name, config_weight)


class SearchPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedding_model: EmbeddingModel | None = None
        self.source_indices: list[SourceIndex] = []
        self.retrievers: list[EmbeddingRetriever] = []
        self.scorers: list[Scorer] = []
        self.nhanes_store = None  # NhanesStore | None
        self._macro_scales: dict[str, float] = dict(DEFAULT_MACRO_SCALES)
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        config = self.config
        index_device = resolve_device(config.index.device)
        logger.info("Loading search pipeline (index_device=%s)", index_device)

        # Load embedding model (shared across all sources)
        self.embedding_model = EmbeddingModel(
            config.scoring.embedding,
            device=config.scoring.embedding.device,
        )

        # Load per-source indices
        index_base = Path(config.index.path)
        self.source_indices = []
        self.retrievers = []

        for src in config.sources:
            if not src.enabled:
                continue
            source_dir = index_base / src.name
            faiss_path = source_dir / config.index.faiss_file
            meta_path = source_dir / config.index.metadata_file
            if not faiss_path.exists() or not meta_path.exists():
                logger.warning(
                    "Index not found for source %s (missing %s or %s), skipping",
                    src.name, faiss_path, meta_path,
                )
                continue

            fi = FaissIndex(config.scoring.embedding.dimension)
            fi.load(faiss_path, device=index_device)
            ms = MetadataStore()
            ms.load(meta_path)
            scales = _load_macro_scales(
                source_dir / config.index.macro_stats_file
            )
            si = SourceIndex(
                name=src.name, faiss_index=fi,
                metadata_store=ms, macro_scales=scales,
            )
            self.source_indices.append(si)
            self.retrievers.append(
                EmbeddingRetriever(
                    self.embedding_model, fi, ms,
                    top_k=config.scoring.embedding.top_k,
                    source=src.name,
                )
            )

        # Merge macro scales across sources (last wins; similar values expected)
        self._macro_scales = dict(DEFAULT_MACRO_SCALES)
        for si in self.source_indices:
            self._macro_scales.update(si.macro_scales)

        # Build scorer chain
        self.scorers = []
        if config.scoring.reranker.enabled:
            reranker_model = RerankerModel(
                config.scoring.reranker, device=config.scoring.reranker.device,
            )
            self.scorers.append(
                RerankerScorer(reranker_model, config.scoring.reranker.weight)
            )
        if config.scoring.macronutrient.enabled:
            self.scorers.append(
                MacronutrientScorer(
                    weight=config.scoring.macronutrient.weight,
                    fields=config.scoring.macronutrient.fields,
                    scales=self._macro_scales,
                )
            )
        if config.scoring.metadata.enabled:
            self.scorers.append(
                MetadataScorer(
                    weight=config.scoring.metadata.weight,
                    list_fields=config.scoring.metadata.list_fields,
                    scalar_fields=config.scoring.metadata.scalar_fields,
                )
            )
        if config.scoring.edit_distance.enabled:
            self.scorers.append(
                EditDistanceScorer(weight=config.scoring.edit_distance.weight)
            )
        if config.scoring.llm.enabled:
            llm_backend = create_llm_backend(
                config.scoring.llm, device=config.scoring.llm.device,
            )
            self.scorers.append(
                LlmScorer(llm_backend, config.scoring.llm.weight)
            )

        # NHANES-based personalization scorers
        self.nhanes_store = None
        if config.nhanes.enabled:
            usda_config = next(
                (s for s in config.sources if s.name == "usda"), None
            )
            if usda_config is not None:
                from food_similarity.config import UsdaSourceConfig

                if isinstance(usda_config, UsdaSourceConfig):
                    from food_similarity.nhanes import NhanesStore

                    try:
                        store = NhanesStore()
                        store.load(config, usda_config)
                    except Exception:
                        logger.warning(
                            "Failed to load NHANES data; "
                            "NHANES-based personalization scorers will be disabled.",
                            exc_info=True,
                        )
                    else:
                        self.nhanes_store = store

                        if config.scoring.preference.enabled:
                            from food_similarity.preference_scorer import (
                                PreferenceScorer,
                            )

                            self.scorers.append(
                                PreferenceScorer(
                                    store, config.scoring.preference,
                                    self._macro_scales,
                                )
                            )
                        if config.scoring.meal_context.enabled:
                            from food_similarity.meal_context_scorer import (
                                MealContextScorer,
                            )

                            self.scorers.append(
                                MealContextScorer(
                                    store, config.scoring.meal_context,
                                    self._macro_scales,
                                )
                            )

        self._loaded = True
        logger.info(
            "Pipeline ready: %d sources loaded, %d scorers enabled",
            len(self.source_indices), len(self.scorers),
        )

    def unload(self) -> None:
        """Release heavy resources (models, indices, metadata) to free memory."""
        if not self._loaded:
            return
        self.embedding_model = None
        self.source_indices.clear()
        self.retrievers.clear()
        self.scorers.clear()
        self.nhanes_store = None
        self._loaded = False

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Pipeline unloaded to free memory")

    def ensure_loaded(self) -> None:
        """Reload the pipeline if it was previously unloaded."""
        if not self._loaded:
            logger.info("Reloading pipeline after idle eviction...")
            self._load()

    def search(self, request: SearchRequest | str) -> list[Candidate]:
        if isinstance(request, str):
            request = SearchRequest.from_query(request)
        if not request.query:
            return []

        # Stage 1: retrieve candidates from all sources
        query_vec = self.embedding_model.encode_query(request.query)
        candidates: list[Candidate] = []
        for retriever in self.retrievers:
            candidates.extend(retriever.retrieve(request.query, query_vec=query_vec))
        logger.info("Retrieved %d candidates for '%s'", len(candidates), request.query)

        # Filter by source if requested
        if request.source_filter:
            candidates = [
                c for c in candidates if c.source in request.source_filter
            ]

        # Stage 2+: apply scorers (using per-request weight overrides)
        overrides = request.weight_overrides
        active_scorers = [
            s for s in self.scorers
            if _effective_weight(s.name, s.weight, overrides) > 0
            and s.is_active(request)
        ]
        for scorer in active_scorers:
            candidates = scorer.score(request, candidates)

        # Compute final weighted score
        emb_weight = _effective_weight(
            "embedding", self.config.scoring.embedding.weight, overrides,
        )
        total_weight = emb_weight + sum(
            _effective_weight(s.name, s.weight, overrides)
            for s in active_scorers
        )
        for c in candidates:
            weighted_sum = c.scores.get("embedding", 0.0) * emb_weight
            for scorer in active_scorers:
                w = _effective_weight(scorer.name, scorer.weight, overrides)
                weighted_sum += c.scores.get(scorer.name, 0.0) * w
            c.final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Deduplicate (if enabled), sort, and return top results
        if self.config.search.deduplicate:
            candidates = _deduplicate(candidates)
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[: self.config.search.final_top_k]


def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    """Keep highest-scoring candidate per (product_name, brands, source) tuple."""
    seen: dict[tuple[str, str, str], Candidate] = {}
    for c in candidates:
        key = (c.product_name.strip().lower(), c.brands.strip().lower(), c.source)
        existing = seen.get(key)
        if existing is None or c.final_score > existing.final_score:
            seen[key] = c
    deduped = list(seen.values())
    if len(deduped) < len(candidates):
        logger.info("Deduplicated %d -> %d candidates", len(candidates), len(deduped))
    return deduped


def _clean_tag(tag: str) -> str:
    """Strip locale prefix (e.g. 'en:') and replace hyphens with spaces."""
    tag = re.sub(r"^[a-z]{2}:", "", tag.strip().lower())
    return tag.replace("-", " ")


def _normalize_tag_list(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    out = set()
    for item in value:
        if isinstance(item, str):
            normalized = _clean_tag(item)
            if normalized:
                out.add(normalized)
    return out


def _normalize_scalar_grade(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized or None
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        as_float = float(value)
        if math.isfinite(as_float):
            return as_float
    return None


def _load_macro_scales(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_MACRO_SCALES)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read macro stats file %s; using defaults", path)
        return dict(DEFAULT_MACRO_SCALES)

    scales = dict(DEFAULT_MACRO_SCALES)
    if isinstance(raw, dict):
        for field_name in MACRO_FIELDS:
            payload = raw.get(field_name)
            if not isinstance(payload, dict):
                continue
            scale = payload.get("scale")
            if isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0:
                scales[field_name] = float(scale)
    return scales
