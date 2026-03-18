from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from fastrag.loader.base import BaseLoader
from fastrag.models.document import Document, Metadata


class MarkdownLoader(BaseLoader):
    supported_extensions: ClassVar[list[str]] = [".md", ".markdown"]

    def load(self, source: str | Path | bytes, **kwargs) -> list[Document]:
        raw = self._read_bytes(source)
        text = raw.decode("utf-8", errors="replace")
        source_name = self._resolve_source_name(source)

        headers = self._extract_headers(text)
        return [
            Document(
                content=text,
                metadata=Metadata(
                    source=source_name,
                    file_type="markdown",
                    extra={"headers": headers},
                ),
            )
        ]

    @staticmethod
    def _extract_headers(text: str) -> list[dict[str, str | int]]:
        headers = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
            headers.append({"level": len(match.group(1)), "text": match.group(2).strip()})
        return headers
