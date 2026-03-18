from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fastrag.models.search import ScoredDocument


class BaseReranker(ABC):
    """Abstract base class for rerankers.

    Subclasses must implement `rerank()`.
    """

    @abstractmethod
    def rerank(self, query: str, documents: list[ScoredDocument], top_k: int | None = None) -> list[ScoredDocument]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query.
            documents: Candidate documents to rerank.
            top_k: If set, return only the top-k results.

        Returns:
            Reranked list of ScoredDocument, sorted by score descending.
        """
        ...

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s < 1e-9:
            return np.ones_like(scores) if max_s > 0 else np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
