"""FastAPI routes for the manual evaluation system."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator, model_validator

from food_similarity.eval_store import EvalStore

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

eval_store: EvalStore | None = None


def set_eval_store(store: EvalStore | None) -> None:
    global eval_store
    eval_store = store


def _require_eval_store() -> EvalStore:
    if eval_store is None:
        raise HTTPException(
            status_code=503,
            detail="Evaluation system unavailable (USDA similarity data may not be loaded)",
        )
    return eval_store


# ------------------------------------------------------------------
# HTML routes
# ------------------------------------------------------------------

@router.get("/eval")
async def eval_redirect():
    return RedirectResponse(url="/eval/binary", status_code=302)


@router.get("/eval/binary", response_class=HTMLResponse)
async def eval_binary(request: Request):
    return templates.TemplateResponse("eval.html", {
        "request": request, "active_tab": "binary",
        "resources_reloaded": getattr(request.state, "resources_reloaded", False),
    })


@router.get("/eval/bestswap", response_class=HTMLResponse)
async def eval_bestswap(request: Request):
    return templates.TemplateResponse("eval.html", {
        "request": request, "active_tab": "bestswap",
        "resources_reloaded": getattr(request.state, "resources_reloaded", False),
    })


@router.get("/eval/likert", response_class=HTMLResponse)
async def eval_likert(request: Request):
    return templates.TemplateResponse("eval.html", {
        "request": request, "active_tab": "likert",
        "resources_reloaded": getattr(request.state, "resources_reloaded", False),
    })


@router.get("/eval/goodswaps", response_class=HTMLResponse)
async def eval_goodswaps(request: Request):
    return templates.TemplateResponse("eval.html", {
        "request": request, "active_tab": "goodswaps",
        "resources_reloaded": getattr(request.state, "resources_reloaded", False),
    })


# ------------------------------------------------------------------
# Sampling API
# ------------------------------------------------------------------

@router.get("/api/eval/sample/binary")
async def api_sample_binary():
    return _require_eval_store().sample_binary()


@router.get("/api/eval/sample/bestswap")
async def api_sample_bestswap():
    return _require_eval_store().sample_bestswap()


@router.get("/api/eval/sample/likert")
async def api_sample_likert():
    return _require_eval_store().sample_likert()


@router.get("/api/eval/sample/goodswaps")
async def api_sample_goodswaps():
    return _require_eval_store().sample_goodswaps()


# ------------------------------------------------------------------
# Rating API
# ------------------------------------------------------------------

class BinaryRatingRequest(BaseModel):
    session_id: str
    query_food_id: int
    candidate_food_id: int
    candidate_rank: int
    response: str
    similarity_score: float

    @field_validator("response")
    @classmethod
    def response_must_be_yes_or_no(cls, v: str) -> str:
        if v not in ("yes", "no"):
            raise ValueError("response must be 'yes' or 'no'")
        return v


class BestSwapRatingRequest(BaseModel):
    session_id: str
    query_food_id: int
    candidate_food_ids: list[int]
    chosen_food_id: int
    candidate_ranks: list[int]

    @model_validator(mode="after")
    def check_consistency(self) -> BestSwapRatingRequest:
        if len(self.candidate_food_ids) != len(self.candidate_ranks):
            raise ValueError("candidate_food_ids and candidate_ranks must be the same length")
        if self.chosen_food_id not in self.candidate_food_ids:
            raise ValueError("chosen_food_id must be one of candidate_food_ids")
        return self


class LikertRatingRequest(BaseModel):
    session_id: str
    food_id_a: int
    food_id_b: int
    rating: int
    similarity_bin: int
    similarity_score: float | None = None

    @field_validator("rating")
    @classmethod
    def rating_must_be_1_to_5(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("rating must be between 1 and 5")
        return v

    @field_validator("similarity_bin")
    @classmethod
    def bin_must_be_1_to_5(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("similarity_bin must be between 1 and 5")
        return v


class GoodSwapsRatingRequest(BaseModel):
    session_id: str
    query_food_id: int
    candidate_food_ids: list[int]
    chosen_food_ids: list[int]
    candidate_ranks: list[int]

    @model_validator(mode="after")
    def check_consistency(self) -> GoodSwapsRatingRequest:
        if len(self.candidate_food_ids) != len(self.candidate_ranks):
            raise ValueError("candidate_food_ids and candidate_ranks must be the same length")
        cid_set = set(self.candidate_food_ids)
        if not set(self.chosen_food_ids).issubset(cid_set):
            raise ValueError("chosen_food_ids must be a subset of candidate_food_ids")
        return self


class SessionRequest(BaseModel):
    username: str
    browser_cookie: str


@router.post("/api/eval/rate/binary")
async def api_rate_binary(req: BinaryRatingRequest):
    store = _require_eval_store()
    store.record_binary(
        session_id=req.session_id,
        query_food_id=req.query_food_id,
        candidate_food_id=req.candidate_food_id,
        candidate_rank=req.candidate_rank,
        response=req.response,
        similarity_score=req.similarity_score,
    )
    return {"ok": True}


@router.post("/api/eval/rate/bestswap")
async def api_rate_bestswap(req: BestSwapRatingRequest):
    store = _require_eval_store()
    store.record_bestswap(
        session_id=req.session_id,
        query_food_id=req.query_food_id,
        candidate_food_ids=req.candidate_food_ids,
        chosen_food_id=req.chosen_food_id,
        candidate_ranks=req.candidate_ranks,
    )
    return {"ok": True}


@router.post("/api/eval/rate/goodswaps")
async def api_rate_goodswaps(req: GoodSwapsRatingRequest):
    store = _require_eval_store()
    store.record_goodswaps(
        session_id=req.session_id,
        query_food_id=req.query_food_id,
        candidate_food_ids=req.candidate_food_ids,
        chosen_food_ids=req.chosen_food_ids,
        candidate_ranks=req.candidate_ranks,
    )
    return {"ok": True}


@router.post("/api/eval/rate/likert")
async def api_rate_likert(req: LikertRatingRequest):
    store = _require_eval_store()
    store.record_likert(
        session_id=req.session_id,
        food_id_a=req.food_id_a,
        food_id_b=req.food_id_b,
        rating=req.rating,
        similarity_bin=req.similarity_bin,
        similarity_score=req.similarity_score,
    )
    return {"ok": True}


# ------------------------------------------------------------------
# Session & stats API
# ------------------------------------------------------------------

@router.post("/api/eval/session")
async def api_session(req: SessionRequest):
    store = _require_eval_store()
    session_id = store.create_or_get_session(req.username, req.browser_cookie)
    return {"session_id": session_id}


@router.get("/api/eval/stats")
async def api_stats(session_id: str = ""):
    if not session_id:
        return {"binary_count": 0, "bestswap_count": 0, "likert_count": 0, "goodswaps_count": 0}
    store = _require_eval_store()
    return store.get_stats(session_id)
