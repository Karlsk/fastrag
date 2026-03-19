from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

from fastrag.config import FastRAGSettings, get_settings
from fastrag.llm.base import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI-compatible embedding client."""

    def __init__(
        self,
        settings: FastRAGSettings | None = None,
        model: str | None = None,
    ):
        s = settings or get_settings()
        self._model = model or s.EMBEDDING_MODEL
        self._client = AsyncOpenAI(
            api_key=s.OPENAI_API_KEY,
            base_url=s.OPENAI_BASE_URL,
            timeout=s.LLM_TIMEOUT,
            max_retries=s.LLM_MAX_RETRIES,
        )
        logger.debug("Embedding init: model=%s", self._model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.monotonic()
        try:
            resp = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            result = [d.embedding for d in sorted_data]
            elapsed = time.monotonic() - t0
            dim = len(result[0]) if result else 0
            logger.info(
                "Embedding: batch=%d, dim=%d, elapsed=%.3fs",
                len(texts), dim, elapsed,
            )
            return result
        except Exception:
            logger.error(
                "Embedding failed: elapsed=%.3fs",
                time.monotonic() - t0, exc_info=True,
            )
            raise
