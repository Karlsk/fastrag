from __future__ import annotations

import math
from collections import defaultdict

from fastrag.models.document import Chunk
from fastrag.models.search import ScoredDocument
from fastrag.retrieval.base import BaseRetriever


class KeywordRetriever(BaseRetriever):
    """Simple BM25-based keyword retriever operating on in-memory chunks.

    For production use, consider integrating with Elasticsearch or similar.
    """

    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75):
        self._chunks: list[Chunk] = chunks or []
        self._k1 = k1
        self._b = b
        self._avg_dl: float = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._doc_term_freqs: list[dict[str, int]] = []
        if self._chunks:
            self._build_index()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._build_index()

    def _build_index(self) -> None:
        self._doc_freqs = defaultdict(int)
        self._doc_lens = []
        self._doc_term_freqs = []

        for chunk in self._chunks:
            terms = self._tokenize(chunk.content)
            self._doc_lens.append(len(terms))
            tf: dict[str, int] = defaultdict(int)
            seen: set[str] = set()
            for t in terms:
                tf[t] += 1
                if t not in seen:
                    self._doc_freqs[t] += 1
                    seen.add(t)
            self._doc_term_freqs.append(dict(tf))

        total = sum(self._doc_lens) if self._doc_lens else 1
        self._avg_dl = total / max(len(self._doc_lens), 1)

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[ScoredDocument]:
        query_terms = self._tokenize(query)
        n = len(self._chunks)
        scores: list[float] = []

        for i in range(n):
            score = 0.0
            dl = self._doc_lens[i]
            for term in query_terms:
                if term not in self._doc_term_freqs[i]:
                    continue
                tf = self._doc_term_freqs[i][term]
                df = self._doc_freqs.get(term, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                score += idf * numerator / denominator
            scores.append(score)

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in indexed:
            if score <= 0:
                break
            results.append(
                ScoredDocument(
                    chunk=self._chunks[idx],
                    score=score,
                    keyword_score=score,
                )
            )
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
