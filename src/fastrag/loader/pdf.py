from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar

from fastrag.loader.base import BaseLoader
from fastrag.models.document import Document, Metadata


class PDFLoader(BaseLoader):
    """PDF loader using pdfplumber."""

    supported_extensions: ClassVar[list[str]] = [".pdf"]

    def load(self, source: str | Path | bytes, **kwargs) -> list[Document]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required for PDF loading. Install with: pip install fastrag[pdf]")

        raw = self._read_bytes(source)
        source_name = self._resolve_source_name(source)
        documents: list[Document] = []

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                documents.append(
                    Document(
                        content=text,
                        metadata=Metadata(
                            source=source_name,
                            file_type="pdf",
                            page_number=i + 1,
                        ),
                    )
                )

        return documents
