from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class BaseLLM(ABC):
    """Abstract base class for LLM chat models."""

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send messages and return the complete response text."""
        ...

    @abstractmethod
    async def stream_chat(self, messages: list[dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Send messages and yield response chunks as they arrive."""
        ...
        yield ""  # pragma: no cover


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning a list of embedding vectors."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        result = await self.embed([text])
        return result[0]
