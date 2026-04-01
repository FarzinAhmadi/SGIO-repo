from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from food_similarity.config import RerankerScoringConfig, resolve_device

logger = logging.getLogger(__name__)


class RerankerModel:
    def __init__(
        self, config: RerankerScoringConfig, device: str = "auto"
    ) -> None:
        self.config = config
        self.device = resolve_device(device)

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Try flash_attention_2, fall back to sdpa
        attn_impl = "sdpa"
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                pass

        logger.info("Loading reranker model %s on %s", config.model, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute token IDs for "yes" and "no"
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.config.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _build_messages(self, query: str, document: str) -> list[dict]:
        pair_text = self._format_pair(query, document)
        return [
            {
                "role": "system",
                "content": (
                    "Judge whether the document is relevant to the query. "
                    'Answer only "yes" or "no".'
                ),
            },
            {"role": "user", "content": pair_text},
        ]

    def score_pairs(
        self, pairs: list[tuple[str, str]], batch_size: int | None = None
    ) -> list[float]:
        """Score a flat list of (query, document) pairs.

        Unlike :meth:`score`, each pair may have a different query string.
        This enables efficient batching across many query items at once.
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            texts = [
                self.tokenizer.apply_chat_template(
                    self._build_messages(query, doc),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for query, doc in batch
            ]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # With left-padding, all actual tokens end at the final position.
            last_pos = outputs.logits.shape[1] - 1
            for j in range(len(batch)):
                logits = outputs.logits[j, last_pos]
                yes_no = torch.tensor(
                    [logits[self.yes_token_id], logits[self.no_token_id]],
                    device=self.device,
                )
                scores.append(torch.softmax(yes_no, dim=0)[0].item())
        return scores

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = []
        batch_size = self.config.batch_size

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_texts = []
            for doc in batch_docs:
                messages = self._build_messages(query, doc)
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
                outputs = self.model(**inputs)

            # With left-padding, all actual tokens end at the final position.
            last_pos = outputs.logits.shape[1] - 1
            for j in range(len(batch_docs)):
                logits = outputs.logits[j, last_pos]
                yes_no_logits = torch.tensor(
                    [logits[self.yes_token_id], logits[self.no_token_id]],
                    device=self.device,
                )
                probs = torch.softmax(yes_no_logits, dim=0)
                scores.append(probs[0].item())

        return scores
