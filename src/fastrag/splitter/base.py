from __future__ import annotations

from abc import ABC, abstractmethod

from fastrag.models.document import Chunk, Document
from fastrag.utils.tokenizer import count_tokens


class BaseSplitter(ABC):
    """Abstract base class for text splitters.

    Subclasses must implement `split()`.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        """
        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of overlapping tokens between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...

    @staticmethod
    def _count_tokens(text: str) -> int:
        return count_tokens(text)

    def _make_chunks(self, texts: list[str], document: Document) -> list[Chunk]:
        """Helper: convert a list of text segments into Chunk objects."""
        chunks = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    content=text,
                    document_id=document.id,
                    index=i,
                    metadata=document.metadata.model_copy(),
                )
            )
        return chunks
