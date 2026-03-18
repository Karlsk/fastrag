from __future__ import annotations

import re

from fastrag.models.document import Chunk, Document
from fastrag.splitter.base import BaseSplitter
from fastrag.utils.tokenizer import count_tokens


class MarkdownSplitter(BaseSplitter):
    """Split markdown documents by header levels.

    Splits on markdown headers (# to ######), grouping content under each header
    into a chunk. If a section exceeds chunk_size, it falls back to line-based splitting.
    """

    _HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        super().__init__(chunk_size, chunk_overlap)

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sections = self._split_by_headers(text)
        merged = self._merge_sections(sections)
        return self._make_chunks(merged, document)

    def _split_by_headers(self, text: str) -> list[str]:
        matches = list(self._HEADER_RE.finditer(text))
        if not matches:
            return [text]

        sections: list[str] = []
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(preamble)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)

        return sections

    def _merge_sections(self, sections: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for section in sections:
            sec_tokens = count_tokens(section)

            if sec_tokens > self.chunk_size:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                # Split oversized section by lines
                lines = section.split("\n")
                buf: list[str] = []
                buf_tokens = 0
                for line in lines:
                    lt = count_tokens(line)
                    if buf_tokens + lt > self.chunk_size and buf:
                        chunks.append("\n".join(buf))
                        buf = []
                        buf_tokens = 0
                    buf.append(line)
                    buf_tokens += lt
                if buf:
                    chunks.append("\n".join(buf))
                continue

            if current_tokens + sec_tokens > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            current.append(section)
            current_tokens += sec_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks
