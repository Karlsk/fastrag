from __future__ import annotations

import re

from fastrag.models.document import Chunk, Document
from fastrag.splitter.base import BaseSplitter
from fastrag.utils.tokenizer import count_tokens


class SentenceSplitter(BaseSplitter):
    """Split text by sentence boundaries, then merge into chunks.

    Supports both Chinese and English sentence-ending punctuation.
    """

    _SENTENCE_RE = re.compile(r"(?<=[。！？!?\.\n])\s*")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        super().__init__(chunk_size, chunk_overlap)

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sentences = [s for s in self._SENTENCE_RE.split(text) if s.strip()]
        merged = self._merge_sentences(sentences)
        return self._make_chunks(merged, document)

    def _merge_sentences(self, sentences: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = count_tokens(sent)
            if current_tokens + sent_tokens > self.chunk_size and current:
                chunks.append(" ".join(current))
                overlap = self._collect_overlap(current)
                current = overlap
                current_tokens = sum(count_tokens(s) for s in current)
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _collect_overlap(self, segments: list[str]) -> list[str]:
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
