from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from fastrag.models.document import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders.

    Subclasses must define `supported_extensions` and implement `load()`.
    """

    supported_extensions: ClassVar[list[str]]

    @abstractmethod
    def load(self, source: str | Path | bytes, **kwargs) -> list[Document]:
        """Load a document from a file path or raw bytes.

        Args:
            source: File path (str/Path) or raw file content (bytes).

        Returns:
            A list of Document objects extracted from the source.
        """
        ...

    @staticmethod
    def _read_bytes(source: str | Path | bytes) -> bytes:
        if isinstance(source, bytes):
            return source
        return Path(source).read_bytes()

    @staticmethod
    def _resolve_source_name(source: str | Path | bytes) -> str:
        if isinstance(source, bytes):
            return "<bytes>"
        return str(source)
