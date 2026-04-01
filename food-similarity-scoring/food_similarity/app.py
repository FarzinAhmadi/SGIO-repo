from __future__ import annotations

import asyncio
import gc
import logging
import math
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from food_similarity.config import AppConfig, load_config
from food_similarity.dietary_rules import DIET_FLAG_NAMES
from food_similarity.eval_routes import router as eval_router, set_eval_store
from food_similarity.eval_store import EvalStore
from food_similarity.pipeline import SearchPipeline
from food_similarity.precomputed import PrecomputedStore

if TYPE_CHECKING:
    from food_similarity.nhanes import NhanesStore
from food_similarity.search_request import (
    SearchRequest,
    parse_search_request_params,
)
from food_similarity.signals import (
    KNOWN_SOURCES,
    LIST_METADATA_FIELDS,
    MACRO_FIELDS,
    SCALAR_METADATA_FIELDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def _clean_tag(tag: str) -> str:
    """Strip locale prefix (e.g. 'en:') and replace hyphens with spaces."""
    tag = re.sub(r"^[a-z]{2}:", "", tag.strip().lower())
    return tag.replace("-", " ")


pipeline: SearchPipeline | None = None
nhanes_store: NhanesStore | None = None
tag_cache: dict[str, list[str]] = {}
precomputed: PrecomputedStore | None = None
_last_activity: float = 0.0
_app_config: AppConfig | None = None


def _touch_activity() -> None:
    global _last_activity
    _last_activity = time.monotonic()


def _ensure_resources_loaded() -> bool:
    """Reload pipeline and precomputed data if they were evicted.

    Returns True if any resources were reloaded.
    """
    global nhanes_store
    reloaded = False
    if pipeline is not None and not pipeline._loaded:
        logger.info("Request received — reloading evicted resources...")
        pipeline.ensure_loaded()
        nhanes_store = pipeline.nhanes_store
        reloaded = True
    if precomputed is not None and not precomputed.is_loaded:
        precomputed.load_all()
        reloaded = True
    from food_similarity.eval_routes import eval_store as _eval_store
    if _eval_store is not None and not _eval_store.is_loaded:
        _eval_store.ensure_loaded()
        reloaded = True
    return reloaded


async def _idle_evictor(config: AppConfig) -> None:
    """Background task: periodically check for idle and evict heavy resources."""
    timeout = config.memory.idle_timeout_seconds
    interval = config.memory.check_interval_seconds
    if timeout <= 0:
        logger.info("Idle eviction disabled (idle_timeout_seconds <= 0)")
        return
    logger.info(
        "Idle evictor started (timeout=%ds, check_interval=%ds)",
        timeout, interval,
    )
    while True:
        await asyncio.sleep(interval)
        if pipeline is None:
            break
        idle = time.monotonic() - _last_activity
        if idle >= timeout and pipeline._loaded:
            logger.info(
                "Server idle for %.0fs (threshold %ds) — evicting resources",
                idle, timeout,
            )
            pipeline.unload()
            if precomputed is not None:
                precomputed.unload()
            from food_similarity.eval_routes import eval_store as _eval_store
            if _eval_store is not None:
                _eval_store.unload()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Idle eviction complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, precomputed, nhanes_store, _app_config
    config = load_config()
    _app_config = config
    logger.info("Loading search pipeline...")
    pipeline = SearchPipeline(config)
    nhanes_store = pipeline.nhanes_store
    for field_name in LIST_METADATA_FIELDS:
        all_tags: set[str] = set()
        for si in pipeline.source_indices:
            raw_tags = si.metadata_store.unique_tags(field_name, limit=500)
            all_tags.update(_clean_tag(t) for t in raw_tags if _clean_tag(t))
        tag_cache[field_name] = sorted(all_tags)
        logger.info("Cached %d tags for %s", len(tag_cache[field_name]), field_name)
    logger.info("Pipeline ready")

    logger.info("Loading pre-computed similarity/mapping data...")
    store = PrecomputedStore(Path("data"))
    store.load_all()
    precomputed = store
    logger.info("Pre-computed data ready")

    eval_db: EvalStore | None = None
    try:
        logger.info("Initializing evaluation store...")
        eval_db = EvalStore(Path("data/eval/eval.db"), store)
        set_eval_store(eval_db)
        logger.info("Evaluation store ready")
    except Exception:
        logger.warning("Evaluation store init failed — eval endpoints will be unavailable", exc_info=True)

    _touch_activity()
    evictor_task = asyncio.create_task(_idle_evictor(config))

    yield

    evictor_task.cancel()
    try:
        await evictor_task
    except asyncio.CancelledError:
        pass
    if eval_db is not None:
        eval_db.close()
    set_eval_store(None)
    pipeline = None
    precomputed = None
    nhanes_store = None
    _app_config = None


app = FastAPI(title="Food Similarity Search", lifespan=lifespan)
app.include_router(eval_router)

templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)


@app.middleware("http")
async def activity_middleware(request: Request, call_next):
    _touch_activity()
    request.state.resources_reloaded = _ensure_resources_loaded()
    return await call_next(request)


def _require_pipeline() -> SearchPipeline:
    if pipeline is None:
        raise RuntimeError("Search pipeline is not initialized")
    return pipeline


def _serialize_results(candidates: list) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "food_id": c.row_id,
            "source": c.source,
            "product_name": c.product_name,
            "brands": c.brands,
            "categories": c.categories,
            "macros": {k: round(v, 2) if v is not None else None for k, v in c.macros.items()},
            "nutriscore_grade": c.metadata_scalars.get("nutriscore_grade"),
            "nova_group": c.metadata_scalars.get("nova_group"),
            "labels": sorted(c.metadata_lists.get("labels_tags", set())),
            "food_groups": sorted(c.metadata_lists.get("food_groups_tags", set())),
            "pnns_groups_1": sorted(c.metadata_lists.get("pnns_groups_1_tags", set())),
            "pnns_groups_2": sorted(c.metadata_lists.get("pnns_groups_2_tags", set())),
            "scores": c.scores,
            "final_score": round(c.final_score, 4),
        }
        for i, c in enumerate(candidates)
    ]


def _empty_input_values() -> dict[str, str]:
    fields = (*MACRO_FIELDS, *LIST_METADATA_FIELDS, *SCALAR_METADATA_FIELDS)
    out = {field: "" for field in fields}
    out["source"] = ""
    return out


def _available_sources() -> list[str]:
    p = _require_pipeline()
    return [si.name for si in p.source_indices]


@app.get("/", response_class=HTMLResponse)
async def search_page(request: Request):
    results = []
    error = None

    try:
        search_request, input_values = parse_search_request_params(
            request.query_params.get
        )
    except ValueError as exc:
        search_request = SearchRequest.from_query(request.query_params.get("q") or "")
        input_values = {
            field: (request.query_params.get(field) or "")
            for field in _empty_input_values()
        }
        error = str(exc)

    if search_request.query and error is None:
        try:
            candidates = _require_pipeline().search(search_request)
            results = _serialize_results(candidates)
        except Exception:
            logger.exception("Search failed for query: %s", search_request.query)
            error = "Search failed. Please try again."

    store = _require_precomputed()
    similarity_sources = set(store.available_similarities())
    # Build mapping lookup: source -> target for sources without own similarity data
    mapping_targets: dict[str, str] = {}
    for src, tgt in store.available_mappings():
        if src not in similarity_sources and src not in mapping_targets:
            mapping_targets[src] = tgt

    scorer_defaults: dict[str, float] = {}
    if _app_config is not None:
        scoring = _app_config.scoring
        if scoring.embedding.enabled:
            scorer_defaults["embedding"] = scoring.embedding.weight
        if scoring.reranker.enabled:
            scorer_defaults["reranker"] = scoring.reranker.weight
        if scoring.macronutrient.enabled:
            scorer_defaults["macronutrient"] = scoring.macronutrient.weight
        if scoring.metadata.enabled:
            scorer_defaults["metadata"] = scoring.metadata.weight
        if scoring.edit_distance.enabled:
            scorer_defaults["edit_distance"] = scoring.edit_distance.weight

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": search_request.query,
            "results": results,
            "error": error,
            "inputs": input_values,
            "available_sources": _available_sources(),
            "similarity_sources": similarity_sources,
            "mapping_targets": mapping_targets,
            "scorer_defaults": scorer_defaults,
            "resources_reloaded": request.state.resources_reloaded,
        },
    )


@app.get("/api/search")
async def api_search(request: Request):
    try:
        search_request, _ = parse_search_request_params(request.query_params.get)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not search_request.query:
        return {"query": "", "applied_criteria": {}, "results": []}

    candidates = _require_pipeline().search(search_request)
    return {
        "query": search_request.query,
        "applied_criteria": search_request.applied_criteria(),
        "results": _serialize_results(candidates),
    }


@app.get("/api/tags/{field_name}")
async def api_tags(field_name: str, prefix: str = "", limit: int = 50):
    if field_name not in tag_cache:
        raise HTTPException(status_code=404, detail=f"Unknown tag field: {field_name}")
    all_tags = tag_cache[field_name]
    if prefix:
        prefix_lower = prefix.strip().lower()
        all_tags = [t for t in all_tags if prefix_lower in t]
    return all_tags[:limit]


def _require_precomputed() -> PrecomputedStore:
    if precomputed is None:
        raise RuntimeError("Pre-computed data is not loaded")
    return precomputed


def _require_nhanes_store() -> NhanesStore:
    if nhanes_store is None:
        raise HTTPException(status_code=503, detail="NHANES data not loaded")
    return nhanes_store


def _serialize_user(user, store: NhanesStore) -> dict:
    active_diets = [k for k, v in user.diet_flags.items() if v]
    return {
        "user_id": user.user_id,
        "gender": user.gender,
        "gender_label": "Male" if user.gender == 1 else "Female",
        "age": user.age,
        "diet_flags": active_diets,
        "food_count": len(user.food_log),
    }


def _serialize_user_detail(user, store: NhanesStore) -> dict:
    info = _serialize_user(user, store)
    meals = store.get_user_meals(user.user_id)
    info["meals"] = {}
    for meal_name, entries in meals.items():
        info["meals"][meal_name] = [
            {
                "food_code": e.food_code,
                "food_name": store.get_food_name(e.food_code),
                "eating_occasion": e.eating_occasion,
                "grams": round(e.grams, 1),
                "energy_kcal": round(e.energy_kcal, 1),
            }
            for e in entries
        ]
    return info


@app.get("/api/users")
async def api_users(
    gender: int | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    diet: str | None = None,
    page: int = 1,
):
    store = _require_nhanes_store()
    page = max(1, page)
    users, total = store.list_users(
        gender=gender,
        min_age=min_age,
        max_age=max_age,
        diet_flag=diet,
        page=page,
    )
    per_page = 50
    total_pages = max(1, math.ceil(total / per_page))
    return {
        "users": [_serialize_user(u, store) for u in users],
        "total": total,
        "page": page,
        "total_pages": total_pages,
    }


@app.get("/api/users/{user_id}")
async def api_user_detail(user_id: int):
    store = _require_nhanes_store()
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _serialize_user_detail(user, store)


@app.get("/api/foods/search")
async def api_food_search(q: str = "", limit: int = 20):
    store = _require_nhanes_store()
    return store.search_foods(q, max(1, min(limit, 50)))


@app.get("/api/foods/lookup")
async def api_foods_lookup(codes: str = ""):
    store = _require_nhanes_store()
    result: dict[str, dict] = {}
    for part in codes.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            fc = int(part)
        except ValueError:
            continue
        macros = store.get_food_macros(fc)
        result[str(fc)] = {
            "food_name": store.get_food_name(fc),
            "energy_kcal": macros.get("energy_kcal_100g"),
            "protein": macros.get("proteins_100g"),
            "fat": macros.get("fat_100g"),
            "carbs": macros.get("carbohydrates_100g"),
        }
    return result


@app.get("/similarity", response_class=HTMLResponse)
async def similarity_page(
    request: Request,
    source: str = "",
    q: str = "",
    food_id: str = "",
    page: int = 1,
):
    store = _require_precomputed()
    available = store.available_similarities()
    if not source and available:
        source = available[0]

    selected_food = None
    foods: list[dict] = []
    total_count = 0
    total_pages = 1

    if food_id and source:
        selected_food = store.get_similarity_detail(source, food_id)
    elif source:
        foods, total_count, total_pages = store.search_similarity(source, q, page)

    return templates.TemplateResponse(
        "similarity.html",
        {
            "request": request,
            "available_sources": available,
            "source": source,
            "query": q,
            "page": page,
            "total_count": total_count,
            "total_pages": total_pages,
            "foods": foods,
            "selected_food": selected_food,
            "resources_reloaded": request.state.resources_reloaded,
        },
    )


@app.get("/mapping", response_class=HTMLResponse)
async def mapping_page(
    request: Request,
    source: str = "",
    target: str = "",
    q: str = "",
    food_id: str = "",
    page: int = 1,
):
    store = _require_precomputed()
    available = store.available_mappings()
    if not source and available:
        source, target = available[0]
    elif source and not target:
        # Find the target for this source
        for s, t in available:
            if s == source:
                target = t
                break

    selected_food = None
    foods: list[dict] = []
    total_count = 0
    total_pages = 1

    if food_id and source and target:
        selected_food = store.get_mapping_detail(source, target, food_id)
    elif source and target:
        foods, total_count, total_pages = store.search_mapping(
            source, target, q, page
        )

    return templates.TemplateResponse(
        "mapping.html",
        {
            "request": request,
            "available_mappings": available,
            "source": source,
            "target": target,
            "query": q,
            "page": page,
            "total_count": total_count,
            "total_pages": total_pages,
            "foods": foods,
            "selected_food": selected_food,
            "resources_reloaded": request.state.resources_reloaded,
        },
    )


@app.get("/personalized", response_class=HTMLResponse)
async def personalized_page(request: Request):
    results = []
    error = None
    selected_user = None
    meal_items: list[dict] = []

    try:
        search_request, input_values = parse_search_request_params(
            request.query_params.get
        )
    except ValueError as exc:
        search_request = SearchRequest.from_query(
            request.query_params.get("q") or ""
        )
        input_values = {
            field: (request.query_params.get(field) or "")
            for field in _empty_input_values()
        }
        input_values["user_id"] = request.query_params.get("user_id") or ""
        input_values["meal"] = request.query_params.get("meal") or ""
        error = str(exc)

    store = nhanes_store
    if store and search_request.user_id is not None:
        user = store.get_user(search_request.user_id)
        if user:
            selected_user = _serialize_user_detail(user, store)

    if store and search_request.meal_food_codes:
        for fc in search_request.meal_food_codes:
            macros = store.get_food_macros(fc)
            meal_items.append({
                "food_code": fc,
                "food_name": store.get_food_name(fc),
                "energy_kcal": macros.get("energy_kcal_100g"),
                "protein": macros.get("proteins_100g"),
                "fat": macros.get("fat_100g"),
                "carbs": macros.get("carbohydrates_100g"),
            })

    if search_request.query and error is None:
        try:
            candidates = _require_pipeline().search(search_request)
            results = _serialize_results(candidates)
        except Exception:
            logger.exception(
                "Search failed for query: %s", search_request.query
            )
            error = "Search failed. Please try again."

    source_str = input_values.get("source", "")
    if source_str:
        selected_sources = {
            s.strip().lower() for s in source_str.split(",") if s.strip()
        }
    else:
        selected_sources = {"usda"}

    return templates.TemplateResponse(
        "personalized.html",
        {
            "request": request,
            "query": search_request.query,
            "results": results,
            "error": error,
            "inputs": input_values,
            "selected_user": selected_user,
            "meal_items": meal_items,
            "diet_flag_names": DIET_FLAG_NAMES,
            "nhanes_enabled": store is not None,
            "available_sources": _available_sources(),
            "selected_sources": selected_sources,
            "resources_reloaded": request.state.resources_reloaded,
        },
    )


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(
        "food_similarity.app:app",
        host=config.web.host,
        port=config.web.port,
        reload=False,
    )
