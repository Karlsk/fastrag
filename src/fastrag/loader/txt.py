from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from fastrag.loader.base import BaseLoader
from fastrag.models.document import Document, Metadata

logger = logging.getLogger(__name__)


class TxtLoader(BaseLoader):
    supported_extensions: ClassVar[list[str]] = [".txt"]

    def __init__(self, encoding: str | None = None):
        self._encoding = encoding

    def load(self, source: str | Path | bytes, **kwargs) -> list[Document]:
        raw = self._read_bytes(source)
        encoding = self._encoding or self._detect_encoding(raw)
        text = raw.decode(encoding, errors="replace")
        src_name = self._resolve_source_name(source)
        logger.info(
            "Loaded TXT: source=%s, encoding=%s, len=%d",
            src_name, encoding, len(text),
        )
        return [
            Document(
                content=text,
                metadata=Metadata(
                    source=src_name,
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
