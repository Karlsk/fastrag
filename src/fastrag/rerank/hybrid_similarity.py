from __future__ import annotations

from collections import Counter

import numpy as np

from fastrag.models.search import ScoredDocument
from fastrag.rerank.base import BaseReranker


class HybridSimilarityReranker(BaseReranker):
    """Reranker combining token similarity and vector similarity.

    Similar to RAGFlow's Dealer.rerank() method that uses weighted fusion
    of keyword-based and vector-based similarity scores.
    """

    def __init__(self, token_weight: float = 0.3, vector_weight: float = 0.7):
        self._tw = token_weight
        self._vw = vector_weight

    def rerank(self, query: str, documents: list[ScoredDocument], top_k: int | None = None) -> list[ScoredDocument]:
        if not documents:
            return []

        query_tokens = self._tokenize(query)

        token_sims = np.array([self._token_similarity(query_tokens, doc.chunk.content) for doc in documents])
        vector_sims = np.array([doc.vector_score if doc.vector_score is not None else 0.0 for doc in documents])

        if token_sims.max() > 0:
            token_sims = self._normalize_scores(token_sims)
        if vector_sims.max() > 0:
            vector_sims = self._normalize_scores(vector_sims)

        combined = self._tw * token_sims + self._vw * vector_sims

        result = []
        for i, doc in enumerate(documents):
            result.append(
                ScoredDocument(
                    chunk=doc.chunk,
                    score=float(combined[i]),
                    vector_score=float(vector_sims[i]),
                    keyword_score=float(token_sims[i]),
                )
            )

        result.sort(key=lambda x: x.score, reverse=True)
        if top_k:
            result = result[:top_k]
        return result

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    @staticmethod
    def _token_similarity(query_tokens: list[str], doc_text: str) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = doc_text.lower().split()
        if not doc_tokens:
            return 0.0
        query_counter = Counter(query_tokens)
        doc_counter = Counter(doc_tokens)
        overlap = sum((query_counter & doc_counter).values())
        return overlap / len(query_tokens)
