from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from fastrag.loader.base import BaseLoader
from fastrag.models.document import Document, Metadata


class TxtLoader(BaseLoader):
    supported_extensions: ClassVar[list[str]] = [".txt"]

    def __init__(self, encoding: str | None = None):
        self._encoding = encoding

    def load(self, source: str | Path | bytes, **kwargs) -> list[Document]:
        raw = self._read_bytes(source)
        encoding = self._encoding or self._detect_encoding(raw)
        text = raw.decode(encoding, errors="replace")
        return [
            Document(
                content=text,
                metadata=Metadata(
                    source=self._resolve_source_name(source),
                    file_type="txt",
                ),
            )
        ]

    @staticmethod
    def _detect_encoding(raw: bytes) -> str:
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"):
            try:
                raw.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue
        return "utf-8"
