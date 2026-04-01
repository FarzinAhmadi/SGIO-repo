from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from food_similarity.config import LlmScoringConfig, resolve_device

logger = logging.getLogger(__name__)

_RATING_RE = re.compile(r"\b(10|[1-9])\b")

_SYSTEM_PROMPT = (
    "Rate how similar the product is to the query on a scale "
    "of 1 to 10. Respond with ONLY a single integer."
)


def _build_messages(query: str, product_name: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\nProduct: {product_name}"},
    ]


def _parse_rating(text: str) -> int | None:
    match = _RATING_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _normalize_rating(rating: int) -> float:
    return (rating - 1) / 9.0


class LlmBackend(ABC):
    @abstractmethod
    def score(self, query: str, product_names: list[str]) -> list[float]: ...


class LocalLlmBackend(LlmBackend):
    def __init__(self, config: LlmScoringConfig, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = resolve_device(device)

        # Try flash_attention_2, fall back to sdpa
        attn_impl = "sdpa"
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                pass

        logger.info("Loading LLM model %s on %s", config.model, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, product_names: list[str]) -> list[float]:
        import torch

        scores: list[float] = []
        batch_size = self.config.batch_size

        for i in range(0, len(product_names), batch_size):
            batch_names = product_names[i : i + batch_size]
            batch_texts = []
            for name in batch_names:
                messages = _build_messages(query, name)
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=8, do_sample=False
                )

            # Decode only the generated tokens (after the input)
            input_len = inputs["input_ids"].shape[1]
            for j in range(len(batch_names)):
                generated_ids = outputs[j, input_len:]
                text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                rating = _parse_rating(text)
                if rating is not None:
                    scores.append(_normalize_rating(rating))
                else:
                    logger.warning(
                        "Could not parse LLM rating from: %r", text
                    )
                    scores.append(0.0)

        return scores


class ApiLlmBackend(LlmBackend):
    def __init__(self, config: LlmScoringConfig) -> None:
        import httpx

        self.config = config
        api_key = (
            config.api_key
            or os.environ.get("LLM_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        api_base = config.api_base or os.environ.get("LLM_API_BASE", "")
        self._url = f"{api_base.rstrip('/')}/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=60.0, headers=self._headers)
        logger.info(
            "Using API LLM backend: model=%s, base=%s",
            config.model,
            config.api_base,
        )

    def _score_one(self, query: str, product_name: str) -> float:
        import httpx

        messages = _build_messages(query, product_name)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": 8,
            # "temperature": 0,
        }
        try:
            response = self._client.post(self._url, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.warning(
                "API LLM request failed for product %r: %d %s — %s",
                product_name,
                e.response.status_code,
                e.response.reason_phrase,
                e.response.text,
            )
            return 0.0
        except Exception:
            logger.warning(
                "API LLM request failed for product %r",
                product_name,
                exc_info=True,
            )
            return 0.0

        rating = _parse_rating(text)
        if rating is not None:
            return _normalize_rating(rating)
        logger.warning("Could not parse LLM rating from: %r", text)
        return 0.0

    def score(self, query: str, product_names: list[str]) -> list[float]:
        max_workers = max(1, self.config.batch_size)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._score_one, query, name): i
                for i, name in enumerate(product_names)
            }
            results = [0.0] * len(product_names)
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results


def create_llm_backend(
    config: LlmScoringConfig, device: str = "auto"
) -> LlmBackend:
    if config.backend == "api":
        return ApiLlmBackend(config)
    return LocalLlmBackend(config, device=device)
