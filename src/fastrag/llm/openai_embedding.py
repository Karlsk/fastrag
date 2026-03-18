from __future__ import annotations

from openai import AsyncOpenAI

from fastrag.config import FastRAGSettings, get_settings
from fastrag.llm.base import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI-compatible embedding client."""

    def __init__(self, settings: FastRAGSettings | None = None, model: str | None = None):
        s = settings or get_settings()
        self._model = model or s.EMBEDDING_MODEL
        self._client = AsyncOpenAI(
            api_key=s.OPENAI_API_KEY,
            base_url=s.OPENAI_BASE_URL,
            timeout=s.LLM_TIMEOUT,
            max_retries=s.LLM_MAX_RETRIES,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]
