from __future__ import annotations

import logging
import time

import httpx
import numpy as np

from fastrag.config import FastRAGSettings, get_settings
from fastrag.models.search import ScoredDocument
from fastrag.rerank.base import BaseReranker

logger = logging.getLogger(__name__)


class OpenAIReranker(BaseReranker):
    """Reranker using OpenAI-compatible /rerank endpoint.

    Works with Jina, Cohere, and other providers that expose an OpenAI-compatible rerank API.
    """

    def __init__(self, settings: FastRAGSettings | None = None, model: str | None = None):
        s = settings or get_settings()
        self._model = model or s.RERANK_MODEL
        self._base_url = s.OPENAI_BASE_URL.rstrip("/")
        self._api_key = s.OPENAI_API_KEY
        self._timeout = s.LLM_TIMEOUT

    def rerank(self, query: str, documents: list[ScoredDocument], top_k: int | None = None) -> list[ScoredDocument]:
        if not documents:
            return []

        texts = [doc.chunk.content for doc in documents]
        t0 = time.monotonic()
        scores = self._call_api(query, texts, top_k)

        result = []
        for i, doc in enumerate(documents):
            result.append(
                ScoredDocument(
                    chunk=doc.chunk,
                    score=scores[i],
                    vector_score=doc.vector_score,
                    keyword_score=doc.keyword_score,
                )
            )

        result.sort(key=lambda x: x.score, reverse=True)
        if top_k:
            result = result[:top_k]
        logger.info(
            "OpenAIReranker: in=%d, out=%d, elapsed=%.3fs",
            len(documents), len(result),
            time.monotonic() - t0,
        )
        return result

    def _call_api(self, query: str, texts: list[str], top_k: int | None) -> np.ndarray:
        payload = {
            "model": self._model,
            "query": query,
            "documents": texts,
        }
        if top_k:
            payload["top_n"] = top_k

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self._base_url}/rerank"

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        scores = np.zeros(len(texts))
        for item in data.get("results", []):
            idx = item["index"]
            scores[idx] = item.get("relevance_score", 0.0)

        return self._normalize_scores(scores)
