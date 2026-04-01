"""Automated LLM-based evaluation of food similarity.

Poses the same questions from the manual evaluation system to frontier LLMs
via the OpenAI Responses API, saving results to a separate database and raw
JSONL logs.

Usage:
    uv run python -m scripts.run_automated_eval
    uv run python -m scripts.run_automated_eval --model gpt-5.4 --num-samples 100
    uv run python -m scripts.run_automated_eval --eval-types binary likert --max-workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

from food_similarity.eval_store import EvalStore
from food_similarity.precomputed import PrecomputedStore

logger = logging.getLogger(__name__)

ALL_EVAL_TYPES = ("binary", "bestswap", "goodswaps", "likert")

# ---------------------------------------------------------------------------
# Macro labels / units — match eval.html JS constants
# ---------------------------------------------------------------------------

MACRO_LABELS: dict[str, str] = {
    "energy_kcal_100g": "Energy",
    "fat_100g": "Fat",
    "carbohydrates_100g": "Carbs",
    "proteins_100g": "Protein",
    "sugars_100g": "Sugars",
    "fiber_100g": "Fiber",
    "salt_100g": "Salt",
    "saturated_fat_100g": "Sat. fat",
}

MACRO_UNITS: dict[str, str] = {
    "energy_kcal_100g": "kcal",
    "fat_100g": "g",
    "carbohydrates_100g": "g",
    "proteins_100g": "g",
    "sugars_100g": "g",
    "fiber_100g": "g",
    "salt_100g": "g",
    "saturated_fat_100g": "g",
}

# ---------------------------------------------------------------------------
# JSON schemas for structured output
# ---------------------------------------------------------------------------

BINARY_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}

BESTSWAP_SCHEMA = {
    "type": "object",
    "properties": {"chosen_food_id": {"type": "integer"}},
    "required": ["chosen_food_id"],
    "additionalProperties": False,
}

GOODSWAPS_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_food_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["chosen_food_ids"],
    "additionalProperties": False,
}

LIKERT_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "string",
            "enum": [
                "not at all similar",
                "slightly similar",
                "somewhat similar",
                "very similar",
                "almost identical",
            ],
        },
    },
    "required": ["rating"],
    "additionalProperties": False,
}

LIKERT_TEXT_TO_INT: dict[str, int] = {
    "not at all similar": 1,
    "slightly similar": 2,
    "somewhat similar": 3,
    "very similar": 4,
    "almost identical": 5,
}

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Skips comments and blank lines."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


# ---------------------------------------------------------------------------
# Food formatting
# ---------------------------------------------------------------------------


def format_food(food: dict) -> str:
    """Render a food dict as plain text matching the human eval UI."""
    lines = [f"Name: {food['name']}"]

    macros = food.get("macros") or {}
    macro_parts = []
    for key, label in MACRO_LABELS.items():
        val = macros.get(key)
        if val is not None:
            unit = MACRO_UNITS.get(key, "")
            macro_parts.append(f"{label}: {val:.1f} {unit}")
    if macro_parts:
        lines.append("Nutrition (per 100g):")
        # Two rows of 4 macros each
        lines.append("  " + " | ".join(macro_parts[:4]))
        if len(macro_parts) > 4:
            lines.append("  " + " | ".join(macro_parts[4:]))

    cats = food.get("categories") or []
    if cats:
        lines.append(f"Categories: {', '.join(cats)}")

    groups = food.get("food_groups") or []
    if groups:
        lines.append(f"Food groups: {', '.join(groups)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders — one per eval type
# ---------------------------------------------------------------------------

_SYSTEM = "You are a food science expert evaluating food similarity."


def build_binary_prompt(sample: dict) -> tuple[str, str, str, dict]:
    """Return (instructions, user_input, schema_name, schema)."""
    user_input = (
        "Question: Could you swap the left food for the right one in a meal?\n\n"
        "Hint: Imagine someone is eating the Original Food. Would the Possible "
        "Replacement be a reasonable replacement with similar taste, texture, or "
        "nutritional value?\n\n"
        "=== Original Food ===\n"
        f"{format_food(sample['query'])}\n\n"
        "=== Possible Replacement ===\n"
        f"{format_food(sample['candidate'])}\n\n"
        'Answer with JSON: {"answer": "yes"} or {"answer": "no"}'
    )
    return _SYSTEM, user_input, "binary_eval", BINARY_SCHEMA


def build_bestswap_prompt(sample: dict) -> tuple[str, str, str, dict]:
    option_labels = "ABCDEFGHIJ"
    options_text = ""
    for i, c in enumerate(sample["candidates"]):
        label = option_labels[i] if i < len(option_labels) else str(i + 1)
        options_text += f"\n=== Option {label} (food_id: {c['food_id']}) ===\n{format_food(c)}\n"

    user_input = (
        "Question: Which of these 4 foods is the single best replacement?\n\n"
        "Hint: Look at the original food below, then pick the one option you think "
        "is the closest match in terms of taste, use, or nutrition.\n\n"
        "=== Original Food ===\n"
        f"{format_food(sample['query'])}\n"
        f"{options_text}\n"
        'Answer with JSON using the food_id of your chosen option: {"chosen_food_id": <integer>}'
    )
    return _SYSTEM, user_input, "bestswap_eval", BESTSWAP_SCHEMA


def build_goodswaps_prompt(sample: dict) -> tuple[str, str, str, dict]:
    candidates_text = ""
    for c in sample["candidates"]:
        candidates_text += f"\n--- food_id: {c['food_id']} ---\n{format_food(c)}\n"

    user_input = (
        "Question: Which of these foods would be acceptable replacements?\n\n"
        "Hint: Select all the foods below that you think could reasonably replace "
        "the original. You can select any number — from none to all of them.\n\n"
        "=== Original Food ===\n"
        f"{format_food(sample['query'])}\n\n"
        "=== Candidates ===\n"
        f"{candidates_text}\n"
        "Answer with JSON listing the food_ids of ALL acceptable replacements "
        '(empty list if none): {"chosen_food_ids": [<integer>, ...]}'
    )
    return _SYSTEM, user_input, "goodswaps_eval", GOODSWAPS_SCHEMA


def build_likert_prompt(sample: dict) -> tuple[str, str, str, dict]:
    user_input = (
        "Question: How similar are these two foods?\n\n"
        "Hint: Think about whether they taste similar, are used in similar meals, "
        "or have similar nutritional profiles. Then pick a rating.\n\n"
        "Choose one of these ratings:\n"
        '  "not at all similar"\n'
        '  "slightly similar"\n'
        '  "somewhat similar"\n'
        '  "very similar"\n'
        '  "almost identical"\n\n'
        "=== Food A ===\n"
        f"{format_food(sample['food_a'])}\n\n"
        "=== Food B ===\n"
        f"{format_food(sample['food_b'])}\n\n"
        'Answer with JSON: {"rating": "<your choice>"}'
    )
    return _SYSTEM, user_input, "likert_eval", LIKERT_SCHEMA


PROMPT_BUILDERS = {
    "binary": build_binary_prompt,
    "bestswap": build_bestswap_prompt,
    "goodswaps": build_goodswaps_prompt,
    "likert": build_likert_prompt,
}

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def parse_binary(data: dict, _sample: dict) -> str:
    answer = data["answer"]
    if answer not in ("yes", "no"):
        raise ValueError(f"Invalid binary answer: {answer!r}")
    return answer


def parse_bestswap(data: dict, sample: dict) -> int:
    chosen = data["chosen_food_id"]
    valid_ids = {c["food_id"] for c in sample["candidates"]}
    if chosen not in valid_ids:
        raise ValueError(f"chosen_food_id {chosen} not in candidates {valid_ids}")
    return chosen


def parse_goodswaps(data: dict, sample: dict) -> list[int]:
    chosen = data["chosen_food_ids"]
    valid_ids = {c["food_id"] for c in sample["candidates"]}
    invalid = set(chosen) - valid_ids
    if invalid:
        raise ValueError(f"Invalid food_ids in chosen_food_ids: {invalid}")
    return chosen


def parse_likert(data: dict, _sample: dict) -> int:
    text = data["rating"]
    rating = LIKERT_TEXT_TO_INT.get(text)
    if rating is None:
        raise ValueError(f"Invalid likert rating text: {text!r}")
    return rating


PARSERS = {
    "binary": parse_binary,
    "bestswap": parse_bestswap,
    "goodswaps": parse_goodswaps,
    "likert": parse_likert,
}

# ---------------------------------------------------------------------------
# LLM call via Responses API
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3


def call_llm(
    client: OpenAI,
    model: str,
    instructions: str,
    user_input: str,
    schema_name: str,
    schema: dict,
    reasoning_effort: str | None,
) -> dict:
    """Call the OpenAI Responses API with structured output and optional reasoning.

    Returns the raw response as a serialisable dict.
    """
    kwargs: dict = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
        "store": False,
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "concise"}

    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.responses.create(**kwargs)
            raw = response.to_dict()
            # output_text is a property, not serialised — stash it in the dict
            raw["output_text"] = response.output_text
            return raw
        except Exception as exc:
            last_err = exc
            err_msg = str(exc).lower()

            # If json_schema not supported, fall back to json_object
            if "json_schema" in err_msg or "response_format" in err_msg or "text.format" in err_msg:
                kwargs["text"] = {"format": {"type": "json_object"}}
                continue

            # If reasoning not supported, drop it
            if "reasoning" in err_msg and "reasoning" in kwargs:
                del kwargs["reasoning"]
                continue

            # Rate limit / transient — backoff
            if attempt < _MAX_RETRIES - 1:
                wait = 2**attempt
                logger.warning("API error (attempt %d/%d), retrying in %ds: %s", attempt + 1, _MAX_RETRIES, wait, exc)
                time.sleep(wait)
                continue
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Process a single sample end-to-end
# ---------------------------------------------------------------------------


def process_one(
    client: OpenAI,
    model: str,
    eval_type: str,
    sample: dict,
    reasoning_effort: str | None,
) -> dict:
    """Build prompt, call LLM, parse response. Returns a result dict."""
    instructions, user_input, schema_name, schema = PROMPT_BUILDERS[eval_type](sample)

    result: dict = {
        "eval_type": eval_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "sample": _extract_sample_metadata(eval_type, sample),
        "instructions": instructions,
        "input": user_input,
        "raw_response": None,
        "parsed_answer": None,
        "error": None,
    }

    try:
        raw = call_llm(client, model, instructions, user_input, schema_name, schema, reasoning_effort)
        result["raw_response"] = raw
        content = raw["output_text"]
        data = json.loads(content)
        PARSERS[eval_type](data, sample)
        # For likert, convert text choice to int for storage
        if eval_type == "likert":
            data["rating_int"] = LIKERT_TEXT_TO_INT[data["rating"]]
        result["parsed_answer"] = data
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("Failed to process %s sample: %s", eval_type, exc)

    return result


def _extract_sample_metadata(eval_type: str, sample: dict) -> dict:
    """Extract lightweight metadata from a sample for the JSONL log."""
    if eval_type == "binary":
        return {
            "query_food_id": sample["query"]["food_id"],
            "candidate_food_id": sample["candidate"]["food_id"],
            "candidate_rank": sample["candidate_rank"],
            "similarity_score": sample["similarity_score"],
        }
    if eval_type == "bestswap":
        return {
            "query_food_id": sample["query"]["food_id"],
            "candidate_food_ids": [c["food_id"] for c in sample["candidates"]],
            "candidate_ranks": [c["rank"] for c in sample["candidates"]],
        }
    if eval_type == "goodswaps":
        return {
            "query_food_id": sample["query"]["food_id"],
            "candidate_food_ids": [c["food_id"] for c in sample["candidates"]],
            "candidate_ranks": [c["rank"] for c in sample["candidates"]],
        }
    # likert
    return {
        "food_id_a": sample["food_a"]["food_id"],
        "food_id_b": sample["food_b"]["food_id"],
        "similarity_bin": sample["similarity_bin"],
        "similarity_score": sample["similarity_score"],
    }


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def _sample_key(eval_type: str, sample: dict) -> tuple:
    """Return a hashable key that identifies this sample's food pair/set."""
    if eval_type == "binary":
        return (sample["query"]["food_id"], sample["candidate"]["food_id"])
    if eval_type in ("bestswap", "goodswaps"):
        return (
            sample["query"]["food_id"],
            tuple(sorted(c["food_id"] for c in sample["candidates"])),
        )
    # likert — order-independent pair
    a, b = sample["food_a"]["food_id"], sample["food_b"]["food_id"]
    return (min(a, b), max(a, b))


def sample_unique(
    eval_db: EvalStore,
    eval_type: str,
    num_samples: int,
    max_attempts_factor: int = 5,
) -> list[dict]:
    """Sample num_samples unique evaluation items, retrying on duplicates."""
    sampler = SAMPLERS[eval_type]
    seen: set[tuple] = set()
    samples: list[dict] = []
    max_attempts = num_samples * max_attempts_factor
    attempts = 0

    while len(samples) < num_samples and attempts < max_attempts:
        s = sampler(eval_db)
        key = _sample_key(eval_type, s)
        attempts += 1
        if key in seen:
            continue
        seen.add(key)
        samples.append(s)

    if len(samples) < num_samples:
        logger.warning(
            "%s: only got %d unique samples out of %d requested (%d attempts)",
            eval_type, len(samples), num_samples, attempts,
        )
    return samples


# ---------------------------------------------------------------------------
# Record parsed answer to the database
# ---------------------------------------------------------------------------


def record_answer(
    eval_db: EvalStore,
    eval_type: str,
    session_id: str,
    sample: dict,
    result: dict,
) -> None:
    """Write a successfully parsed answer to the eval database."""
    if result["error"] is not None or result["parsed_answer"] is None:
        return

    answer = result["parsed_answer"]

    if eval_type == "binary":
        eval_db.record_binary(
            session_id=session_id,
            query_food_id=sample["query"]["food_id"],
            candidate_food_id=sample["candidate"]["food_id"],
            candidate_rank=sample["candidate_rank"],
            response=answer["answer"],
            similarity_score=sample["similarity_score"],
        )
    elif eval_type == "bestswap":
        eval_db.record_bestswap(
            session_id=session_id,
            query_food_id=sample["query"]["food_id"],
            candidate_food_ids=[c["food_id"] for c in sample["candidates"]],
            chosen_food_id=answer["chosen_food_id"],
            candidate_ranks=[c["rank"] for c in sample["candidates"]],
        )
    elif eval_type == "goodswaps":
        eval_db.record_goodswaps(
            session_id=session_id,
            query_food_id=sample["query"]["food_id"],
            candidate_food_ids=[c["food_id"] for c in sample["candidates"]],
            chosen_food_ids=answer["chosen_food_ids"],
            candidate_ranks=[c["rank"] for c in sample["candidates"]],
        )
    elif eval_type == "likert":
        eval_db.record_likert(
            session_id=session_id,
            food_id_a=sample["food_a"]["food_id"],
            food_id_b=sample["food_b"]["food_id"],
            rating=answer["rating_int"],
            similarity_bin=sample["similarity_bin"],
            similarity_score=sample["similarity_score"],
        )


# ---------------------------------------------------------------------------
# Run one eval type
# ---------------------------------------------------------------------------

SAMPLERS = {
    "binary": lambda store: store.sample_binary(),
    "bestswap": lambda store: store.sample_bestswap(),
    "goodswaps": lambda store: store.sample_goodswaps(),
    "likert": lambda store: store.sample_likert(),
}


def run_eval_type(
    eval_db: EvalStore,
    client: OpenAI,
    model: str,
    eval_type: str,
    session_id: str,
    num_samples: int,
    max_workers: int,
    jsonl_fh,
    reasoning_effort: str | None,
) -> dict[str, int]:
    """Sample, dispatch to thread pool, record results. Returns stats."""
    logger.info("Starting %s: %d samples with %d workers", eval_type, num_samples, max_workers)

    # Sample unique tasks upfront (serial)
    samples = sample_unique(eval_db, eval_type, num_samples)

    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_one, client, model, eval_type, sample, reasoning_effort): sample
            for sample in samples
        }

        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "eval_type": eval_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "sample": _extract_sample_metadata(eval_type, sample),
                    "instructions": None,
                    "input": None,
                    "raw_response": None,
                    "parsed_answer": None,
                    "error": str(exc),
                }

            # Write to JSONL
            jsonl_fh.write(json.dumps(result, default=str) + "\n")
            jsonl_fh.flush()

            # Record to DB
            record_answer(eval_db, eval_type, session_id, sample, result)

            if result["error"] is not None:
                errors += 1
            completed += 1

            if completed % 10 == 0 or completed == len(samples):
                logger.info("  %s: %d/%d done, %d errors", eval_type, completed, len(samples), errors)

    return {"completed": completed, "errors": errors}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run automated LLM-based food similarity evaluation",
    )
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name (default: gpt-5.4)")
    parser.add_argument("--base-url", default=None, help="API base URL (overrides .env LLM_API_BASE)")
    parser.add_argument("--api-key", default=None, help="API key (overrides .env LLM_API_KEY)")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per eval type (default: 50)")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel API workers (default: 8)")
    parser.add_argument(
        "--eval-types",
        nargs="+",
        choices=ALL_EVAL_TYPES,
        default=list(ALL_EVAL_TYPES),
        help="Eval types to run (default: all)",
    )
    parser.add_argument("--username", default=None, help="Session username (default: model name)")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort level (default: medium, 'none' to disable)",
    )
    parser.add_argument("--db", default="data/eval/automated_eval.db", help="Database path (default: data/eval/automated_eval.db)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load .env for API credentials
    dotenv = _load_dotenv(Path(".env"))
    api_key = args.api_key or dotenv.get("LLM_API_KEY") or dotenv.get("OPENAI_API_KEY") or ""
    base_url = args.base_url or dotenv.get("LLM_API_BASE") or None
    if not api_key:
        logger.error("No API key found. Set LLM_API_KEY in .env or pass --api-key")
        return

    username = args.username or args.model
    reasoning = args.reasoning_effort if args.reasoning_effort != "none" else None

    # Init OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Init data stores
    logger.info("Loading precomputed data...")
    store = PrecomputedStore(Path("data"))
    store.load_all()

    logger.info("Initializing eval database at %s", args.db)
    eval_db = EvalStore(Path(args.db), store)

    run_id = uuid4().hex[:8]
    session_id = eval_db.create_or_get_session(username, f"auto-{args.model}-{run_id}")

    # Prepare output directory and JSONL file
    out_dir = Path("data/eval/automated")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    jsonl_path = out_dir / f"{safe_model}_{ts}.jsonl"

    logger.info(
        "Config: model=%s, base_url=%s, samples=%d, workers=%d, types=%s, username=%s, reasoning=%s",
        args.model,
        base_url or "(default)",
        args.num_samples,
        args.max_workers,
        args.eval_types,
        username,
        reasoning or "disabled",
    )
    logger.info("Session: %s, JSONL: %s", session_id, jsonl_path)

    # Run evaluations
    summary: dict[str, dict[str, int]] = {}
    with open(jsonl_path, "a") as jsonl_fh:
        for eval_type in args.eval_types:
            stats = run_eval_type(
                eval_db=eval_db,
                client=client,
                model=args.model,
                eval_type=eval_type,
                session_id=session_id,
                num_samples=args.num_samples,
                max_workers=args.max_workers,
                jsonl_fh=jsonl_fh,
                reasoning_effort=reasoning,
            )
            summary[eval_type] = stats

    # Print summary
    logger.info("=" * 50)
    logger.info("Run complete. Summary:")
    total_done = 0
    total_err = 0
    for eval_type, stats in summary.items():
        logger.info("  %s: %d completed, %d errors", eval_type, stats["completed"], stats["errors"])
        total_done += stats["completed"]
        total_err += stats["errors"]
    logger.info("  Total: %d completed, %d errors", total_done, total_err)
    logger.info("  Results DB: %s", args.db)
    logger.info("  Raw logs: %s", jsonl_path)

    eval_db.close()


if __name__ == "__main__":
    main()
