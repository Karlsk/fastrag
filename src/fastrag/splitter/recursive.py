from __future__ import annotations

from fastrag.models.document import Chunk, Document
from fastrag.splitter.base import BaseSplitter
from fastrag.utils.tokenizer import count_tokens


class RecursiveSplitter(BaseSplitter):
    """Recursively split text using a hierarchy of separators.

    Tries each separator in order. If a chunk is still too large after splitting
    with one separator, it falls through to the next.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ".", "；", ";", "！", "!", "？", "?", " "]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        separators: list[str] | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self._separators = separators or self.DEFAULT_SEPARATORS

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []
        pieces = self._recursive_split(text, 0)
        merged = self._merge_small_pieces(pieces)
        return self._make_chunks(merged, document)

    def _recursive_split(self, text: str, depth: int) -> list[str]:
        if count_tokens(text) <= self.chunk_size:
            return [text]

        if depth >= len(self._separators):
            return [text]

        sep = self._separators[depth]
        parts = text.split(sep)

        result: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            if count_tokens(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._recursive_split(part, depth + 1))
        return result

    def _merge_small_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = count_tokens(piece)
            if current_tokens + piece_tokens > self.chunk_size and current:
                chunks.append("\n".join(current))
                overlap = self._collect_overlap(current)
                current = overlap
                current_tokens = sum(count_tokens(s) for s in current)
            current.append(piece)
            current_tokens += piece_tokens

        if current:
            chunks.append("\n".join(current))
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
