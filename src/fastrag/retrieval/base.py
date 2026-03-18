from __future__ import annotations

from abc import ABC, abstractmethod

from fastrag.models.search import ScoredDocument


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[ScoredDocument]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.

        Returns:
            A list of ScoredDocument sorted by relevance (descending).
        """
        ...
