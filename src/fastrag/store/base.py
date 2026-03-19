from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fastrag.models.document import Chunk
from fastrag.models.search import ScoredDocument


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends.

    Defines the interface for vector storage operations. Implement this to
    integrate with different vector databases (Milvus, Qdrant, Weaviate, etc.).
    """

    @abstractmethod
    def create_collection(self, name: str, dimension: int, **kwargs: Any) -> None:
        """Create a new collection/index."""
        ...

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Drop an existing collection."""
        ...

    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """Check if a collection exists."""
        ...

    @abstractmethod
    def insert(self, collection: str, chunks: list[Chunk]) -> list[str]:
        """Insert chunks (with embeddings) into a collection. Returns inserted IDs."""
        ...

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        """Search by vector similarity. Returns scored results."""
        ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None:
        """Delete chunks by IDs."""
        ...

    @abstractmethod
    def count(self, collection: str) -> int:
        """Return the number of entries in a collection."""
        ...

    def list_chunks(self, collection: str, limit: int = 10000) -> list[Chunk]:
        """Return all chunks (without embeddings) from a collection.

        Used to rebuild in-memory indexes (e.g. BM25) from persisted data.
        Subclasses may override for efficiency; the default returns an empty list.
        """
        return []
