from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from fastrag.loader.base import BaseLoader
from fastrag.models.document import Document, Metadata


class HTMLLoader(BaseLoader):
    """HTML loader using BeautifulSoup."""

    supported_extensions: ClassVar[list[str]] = [".html", ".htm"]

    def load(self, source: str | Path | bytes, **kwargs) -> list[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML loading. Install with: pip install fastrag[html]")

        raw = self._read_bytes(source)
        html = raw.decode("utf-8", errors="replace")
        source_name = self._resolve_source_name(source)

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        return [
            Document(
                content=text,
                metadata=Metadata(
                    source=source_name,
                    file_type="html",
                    extra={"title": title} if title else {},
                ),
            )
        ]
