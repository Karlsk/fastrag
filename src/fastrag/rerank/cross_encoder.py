from __future__ import annotations

import numpy as np

from fastrag.models.search import ScoredDocument
from fastrag.rerank.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Reranker using a local cross-encoder model from sentence-transformers.

    Requires the `sentence-transformers` package.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install fastrag[rerank-local]")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[ScoredDocument], top_k: int | None = None) -> list[ScoredDocument]:
        if not documents:
            return []

        pairs = [[query, doc.chunk.content] for doc in documents]
        raw_scores = self._model.predict(pairs)
        scores = self._normalize_scores(np.array(raw_scores))

        result = []
        for i, doc in enumerate(documents):
            result.append(
                ScoredDocument(
                    chunk=doc.chunk,
                    score=float(scores[i]),
                    vector_score=doc.vector_score,
                    keyword_score=doc.keyword_score,
                )
            )

        result.sort(key=lambda x: x.score, reverse=True)
        if top_k:
            result = result[:top_k]
        return result
