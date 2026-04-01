from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from food_similarity.signals import (
    KNOWN_SOURCES,
    LIST_METADATA_FIELDS,
    MACRO_FIELDS,
    SCALAR_METADATA_FIELDS,
)


@dataclass(frozen=True)
class DataConfig:
    name: str = "off"
    path: str = "data/openfoodfacts-products.jsonl.gz"
    name_field: str = "product_name"
    brand_field: str = "brands"
    categories_field: str = "categories_tags"
    min_name_length: int = 2
    enabled: bool = True


@dataclass(frozen=True)
class UsdaSourceConfig:
    name: str = "usda"
    descriptions_path: str = "data/usda/2017-2018/processed/descriptions.csv"
    nutrients_path: str = "data/usda/2017-2018/processed/nutrient_values.csv"
    min_name_length: int = 2
    enabled: bool = True


@dataclass(frozen=True)
class MfpSourceConfig:
    name: str = "myfitnesspal"
    path: str = "data/myfitnesspal/processed/myfitnesspal_foods.csv"
    min_name_length: int = 2
    enabled: bool = True


@dataclass(frozen=True)
class Ai4fooddbSourceConfig:
    name: str = "ai4fooddb"
    path: str = "data/ai4fooddb/processed/ai4fooddb_food_index.csv"
    min_name_length: int = 2
    enabled: bool = True


SourceConfig = DataConfig | UsdaSourceConfig | MfpSourceConfig | Ai4fooddbSourceConfig


@dataclass(frozen=True)
class IndexConfig:
    path: str = "data/index"
    faiss_file: str = "faiss.index"
    metadata_file: str = "metadata.parquet"
    macro_stats_file: str = "macro_stats.json"
    checkpoint_dir: str = "data/index/checkpoints"
    batch_size: int = 256
    device: str = "auto"


@dataclass(frozen=True)
class EmbeddingScoringConfig:
    enabled: bool = True
    weight: float = 1.0
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    instruction: str = "Given a food item name, retrieve similar food products"
    top_k: int = 50
    dimension: int = 1024
    device: str = "auto"


@dataclass(frozen=True)
class RerankerScoringConfig:
    enabled: bool = True
    weight: float = 1.0
    model: str = "Qwen/Qwen3-Reranker-0.6B"
    instruction: str = (
        "Given a food item name, determine if the document describes "
        "a similar food product"
    )
    max_length: int = 8192
    batch_size: int = 16
    device: str = "auto"


@dataclass(frozen=True)
class MacronutrientScoringConfig:
    enabled: bool = True
    weight: float = 0.5
    fields: tuple[str, ...] = MACRO_FIELDS
    similarity: str = "scaled_l1"


@dataclass(frozen=True)
class MetadataScoringConfig:
    enabled: bool = True
    weight: float = 0.5
    list_fields: tuple[str, ...] = LIST_METADATA_FIELDS
    scalar_fields: tuple[str, ...] = SCALAR_METADATA_FIELDS
    list_similarity: str = "query_coverage"


@dataclass(frozen=True)
class EditDistanceScoringConfig:
    enabled: bool = True
    weight: float = 0.3


@dataclass(frozen=True)
class LlmScoringConfig:
    enabled: bool = False
    weight: float = 1.0
    backend: str = "local"  # "local" or "api"
    model: str = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    max_length: int = 2048
    batch_size: int = 8
    api_base: str = ""
    api_key: str = ""
    device: str = "auto"


@dataclass(frozen=True)
class PreferenceScoringConfig:
    enabled: bool = True
    weight: float = 0.5
    centroid_weight: float = 0.4
    category_weight: float = 0.3
    dietary_weight: float = 0.3


@dataclass(frozen=True)
class MealContextScoringConfig:
    enabled: bool = True
    weight: float = 0.4
    nutritional_gap_weight: float = 0.6
    diversity_weight: float = 0.4
    default_daily_kcal: float = 2000.0
    meals_per_day: int = 3


@dataclass(frozen=True)
class ScoringConfig:
    embedding: EmbeddingScoringConfig = None  # type: ignore[assignment]
    reranker: RerankerScoringConfig = None  # type: ignore[assignment]
    macronutrient: MacronutrientScoringConfig = None  # type: ignore[assignment]
    metadata: MetadataScoringConfig = None  # type: ignore[assignment]
    edit_distance: EditDistanceScoringConfig = None  # type: ignore[assignment]
    llm: LlmScoringConfig = None  # type: ignore[assignment]
    preference: PreferenceScoringConfig = None  # type: ignore[assignment]
    meal_context: MealContextScoringConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.embedding is None:
            object.__setattr__(self, "embedding", EmbeddingScoringConfig())
        if self.reranker is None:
            object.__setattr__(self, "reranker", RerankerScoringConfig())
        if self.macronutrient is None:
            object.__setattr__(
                self, "macronutrient", MacronutrientScoringConfig()
            )
        if self.metadata is None:
            object.__setattr__(self, "metadata", MetadataScoringConfig())
        if self.edit_distance is None:
            object.__setattr__(
                self, "edit_distance", EditDistanceScoringConfig()
            )
        if self.llm is None:
            object.__setattr__(self, "llm", LlmScoringConfig())
        if self.preference is None:
            object.__setattr__(self, "preference", PreferenceScoringConfig())
        if self.meal_context is None:
            object.__setattr__(self, "meal_context", MealContextScoringConfig())


@dataclass(frozen=True)
class SearchConfig:
    final_top_k: int = 5
    deduplicate: bool = True


@dataclass(frozen=True)
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass(frozen=True)
class NhanesConfig:
    enabled: bool = False
    subjects_path: str = "data/nhanes/2017/processed/subjects.csv"
    foods_path: str = "data/nhanes/2017/processed/foods.csv"


@dataclass(frozen=True)
class MemoryConfig:
    idle_timeout_seconds: int = 300
    check_interval_seconds: int = 30


@dataclass(frozen=True)
class AppConfig:
    sources: tuple[SourceConfig, ...] = None  # type: ignore[assignment]
    index: IndexConfig = None  # type: ignore[assignment]
    scoring: ScoringConfig = None  # type: ignore[assignment]
    search: SearchConfig = None  # type: ignore[assignment]
    web: WebConfig = None  # type: ignore[assignment]
    nhanes: NhanesConfig = None  # type: ignore[assignment]
    memory: MemoryConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sources is None:
            object.__setattr__(self, "sources", (DataConfig(),))
        if self.index is None:
            object.__setattr__(self, "index", IndexConfig())
        if self.scoring is None:
            object.__setattr__(self, "scoring", ScoringConfig())
        if self.search is None:
            object.__setattr__(self, "search", SearchConfig())
        if self.web is None:
            object.__setattr__(self, "web", WebConfig())
        if self.nhanes is None:
            object.__setattr__(self, "nhanes", NhanesConfig())
        if self.memory is None:
            object.__setattr__(self, "memory", MemoryConfig())


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _validate(config: AppConfig) -> None:
    if not config.sources:
        raise ValueError("At least one source must be configured")
    seen_names: set[str] = set()
    for src in config.sources:
        if src.name in seen_names:
            raise ValueError(f"Duplicate source name: {src.name!r}")
        seen_names.add(src.name)
    if config.scoring.embedding.weight < 0:
        raise ValueError("Embedding weight must be >= 0")
    if config.scoring.reranker.weight < 0:
        raise ValueError("Reranker weight must be >= 0")
    if config.scoring.macronutrient.weight < 0:
        raise ValueError("Macronutrient weight must be >= 0")
    if config.scoring.metadata.weight < 0:
        raise ValueError("Metadata weight must be >= 0")
    if config.scoring.edit_distance.weight < 0:
        raise ValueError("Edit distance weight must be >= 0")
    if config.scoring.llm.weight < 0:
        raise ValueError("LLM weight must be >= 0")
    if config.scoring.llm.backend not in ("local", "api"):
        raise ValueError(
            f"LLM backend must be 'local' or 'api', got {config.scoring.llm.backend!r}"
        )
    if config.scoring.llm.enabled and config.scoring.llm.backend == "api":
        api_base = config.scoring.llm.api_base or os.environ.get("LLM_API_BASE", "")
        if not api_base:
            raise ValueError(
                "LLM api_base must be set (in config.toml or LLM_API_BASE env var) "
                "when backend is 'api'"
            )
    if config.scoring.preference.weight < 0:
        raise ValueError("Preference weight must be >= 0")
    if config.scoring.meal_context.weight < 0:
        raise ValueError("Meal context weight must be >= 0")
    if not (1 <= config.web.port <= 65535):
        raise ValueError(f"Port must be 1-65535, got {config.web.port}")
    if config.scoring.embedding.top_k < config.search.final_top_k:
        raise ValueError(
            f"Embedding top_k ({config.scoring.embedding.top_k}) must be >= "
            f"final_top_k ({config.search.final_top_k})"
        )
    invalid_macro_fields = set(config.scoring.macronutrient.fields) - set(MACRO_FIELDS)
    if invalid_macro_fields:
        raise ValueError(
            f"Unknown macronutrient fields: {sorted(invalid_macro_fields)}"
        )
    invalid_list_fields = set(config.scoring.metadata.list_fields) - set(
        LIST_METADATA_FIELDS
    )
    if invalid_list_fields:
        raise ValueError(f"Unknown metadata list fields: {sorted(invalid_list_fields)}")
    invalid_scalar_fields = set(config.scoring.metadata.scalar_fields) - set(
        SCALAR_METADATA_FIELDS
    )
    if invalid_scalar_fields:
        raise ValueError(
            f"Unknown metadata scalar fields: {sorted(invalid_scalar_fields)}"
        )
    if config.scoring.macronutrient.similarity != "scaled_l1":
        raise ValueError(
            "Macronutrient similarity must be 'scaled_l1'"
        )
    if config.scoring.metadata.list_similarity != "query_coverage":
        raise ValueError(
            "Metadata list_similarity must be 'query_coverage'"
        )


def _coerce_tuple(raw: dict, key: str) -> dict:
    out = dict(raw)
    if key in out and isinstance(out[key], list):
        out[key] = tuple(out[key])
    return out


def _parse_sources(raw: dict) -> tuple[SourceConfig, ...]:
    if "sources" in raw:
        sources: list[SourceConfig] = []
        for entry in raw["sources"]:
            entry = dict(entry)
            src_type = entry.pop("type", "off")
            if src_type == "usda":
                sources.append(UsdaSourceConfig(**entry))
            elif src_type == "myfitnesspal":
                sources.append(MfpSourceConfig(**entry))
            elif src_type == "ai4fooddb":
                sources.append(Ai4fooddbSourceConfig(**entry))
            else:
                sources.append(DataConfig(**entry))
        return tuple(sources)
    if "data" in raw:
        return (DataConfig(**raw["data"]),)
    return (DataConfig(),)


def load_config(path: str | Path = "config.toml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        config = AppConfig()
        _validate(config)
        return config

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    sources = _parse_sources(raw)
    index = IndexConfig(**raw.get("index", {}))

    scoring_raw = raw.get("scoring", {})
    embedding_scoring = EmbeddingScoringConfig(**scoring_raw.get("embedding", {}))
    reranker_scoring = RerankerScoringConfig(**scoring_raw.get("reranker", {}))
    macro_raw = _coerce_tuple(scoring_raw.get("macronutrient", {}), "fields")
    metadata_raw = _coerce_tuple(scoring_raw.get("metadata", {}), "list_fields")
    metadata_raw = _coerce_tuple(metadata_raw, "scalar_fields")
    macronutrient_scoring = MacronutrientScoringConfig(**macro_raw)
    metadata_scoring = MetadataScoringConfig(**metadata_raw)
    edit_distance_scoring = EditDistanceScoringConfig(
        **scoring_raw.get("edit_distance", {})
    )
    llm_scoring = LlmScoringConfig(**scoring_raw.get("llm", {}))
    preference_scoring = PreferenceScoringConfig(
        **scoring_raw.get("preference", {})
    )
    meal_context_scoring = MealContextScoringConfig(
        **scoring_raw.get("meal_context", {})
    )
    scoring = ScoringConfig(
        embedding=embedding_scoring,
        reranker=reranker_scoring,
        macronutrient=macronutrient_scoring,
        metadata=metadata_scoring,
        edit_distance=edit_distance_scoring,
        llm=llm_scoring,
        preference=preference_scoring,
        meal_context=meal_context_scoring,
    )

    search = SearchConfig(**raw.get("search", {}))
    web = WebConfig(**raw.get("web", {}))
    nhanes = NhanesConfig(**raw.get("nhanes", {}))
    memory = MemoryConfig(**raw.get("memory", {}))

    config = AppConfig(
        sources=sources, index=index, scoring=scoring, search=search, web=web,
        nhanes=nhanes, memory=memory,
    )
    _validate(config)
    return config
