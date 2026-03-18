from __future__ import annotations

from fastrag.models.document import Chunk, Document
from fastrag.splitter.base import BaseSplitter
from fastrag.utils.tokenizer import count_tokens, truncate_by_tokens


class TokenSplitter(BaseSplitter):
    """Split text by token count with optional overlap.

    Similar to RAGFlow's naive_merge: splits text into segments, then merges
    segments into chunks that fit within chunk_size tokens.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0, separator: str = "\n"):
        super().__init__(chunk_size, chunk_overlap)
        self._separator = separator

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        segments = text.split(self._separator) if self._separator else [text]
        segments = [s for s in segments if s.strip()]

        merged = self._merge_segments(segments)
        return self._make_chunks(merged, document)

    def _merge_segments(self, segments: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = count_tokens(seg)

            if seg_tokens > self.chunk_size:
                if current:
                    chunks.append(self._separator.join(current))
                    current = self._get_overlap_segments(current)
                    current_tokens = sum(count_tokens(s) for s in current)
                chunks.append(truncate_by_tokens(seg, self.chunk_size))
                continue

            if current_tokens + seg_tokens > self.chunk_size and current:
                chunks.append(self._separator.join(current))
                current = self._get_overlap_segments(current)
                current_tokens = sum(count_tokens(s) for s in current)

            current.append(seg)
            current_tokens += seg_tokens

        if current:
            chunks.append(self._separator.join(current))

        return chunks

    def _get_overlap_segments(self, segments: list[str]) -> list[str]:
        if self.chunk_overlap <= 0:
            return []
        overlap: list[str] = []
        tokens = 0
        for seg in reversed(segments):
            seg_tokens = count_tokens(seg)
            if tokens + seg_tokens > self.chunk_overlap:
                break
            overlap.insert(0, seg)
            tokens += seg_tokens
        return overlap
