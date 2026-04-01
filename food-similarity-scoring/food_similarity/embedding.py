from __future__ import annotations

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from food_similarity.config import EmbeddingScoringConfig, resolve_device

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, config: EmbeddingScoringConfig, device: str = "auto") -> None:
        self.config = config
        self.device = resolve_device(device)

        model_kwargs: dict = {}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            # Try flash_attention_2, fall back to sdpa
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                model_kwargs["attn_implementation"] = "sdpa"
                logger.info("Using SDPA attention (flash-attn not available)")
        else:
            model_kwargs["torch_dtype"] = torch.float32

        logger.info(
            "Loading embedding model %s on %s", config.model, self.device
        )
        self.model = SentenceTransformer(
            config.model,
            device=self.device,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )
        self.model.max_seq_length = 512

    def encode_query(self, query: str) -> np.ndarray:
        prompt_template = (
            f"Instruct: {self.config.instruction}\nQuery: {{query}}"
        )
        embedding = self.model.encode(
            [query],
            prompt=prompt_template.format(query=""),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0]

    def encode_documents(
        self, documents: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        if batch_size is None:
            batch_size = 256
        return self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
