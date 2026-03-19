from __future__ import annotations

import logging
import time

from fastrag.models.search import ScoredDocument
from fastrag.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Combine vector and keyword retrieval with weighted fusion.

    Uses Reciprocal Rank Fusion (RRF) or weighted sum to merge results,
    similar to RAGFlow's FusionExpr("weighted_sum").
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        fusion_method: str = "weighted_sum",
    ):
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._vw = vector_weight
        self._kw = keyword_weight
        self._fusion = fusion_method

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[ScoredDocument]:
        fetch_k = top_k * 3

        t0 = time.monotonic()
        vector_results = await self._vector.retrieve(query, top_k=fetch_k, **kwargs)
        keyword_results = await self._keyword.retrieve(query, top_k=fetch_k, **kwargs)

        if self._fusion == "rrf":
            merged = self._rrf_fusion(vector_results, keyword_results, top_k)
        else:
            merged = self._weighted_sum(vector_results, keyword_results, top_k)
        logger.info(
            "HybridRetriever: merged=%d, elapsed=%.3fs",
            len(merged), time.monotonic() - t0,
        )
        return merged

    def _weighted_sum(
        self,
        vector_results: list[ScoredDocument],
        keyword_results: list[ScoredDocument],
        top_k: int,
    ) -> list[ScoredDocument]:
        merged: dict[str, ScoredDocument] = {}

        v_max = max((r.score for r in vector_results), default=1.0) or 1.0
        k_max = max((r.score for r in keyword_results), default=1.0) or 1.0

        for r in vector_results:
            norm_score = r.score / v_max
            doc = ScoredDocument(
                chunk=r.chunk,
                score=self._vw * norm_score,
                vector_score=norm_score,
            )
            merged[r.chunk.id] = doc

        for r in keyword_results:
            norm_score = r.score / k_max
            if r.chunk.id in merged:
                existing = merged[r.chunk.id]
                merged[r.chunk.id] = ScoredDocument(
                    chunk=existing.chunk,
                    score=existing.score + self._kw * norm_score,
                    vector_score=existing.vector_score,
                    keyword_score=norm_score,
                )
            else:
                merged[r.chunk.id] = ScoredDocument(
                    chunk=r.chunk,
                    score=self._kw * norm_score,
                    keyword_score=norm_score,
                )

        sorted_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]

    def _rrf_fusion(
        self,
        vector_results: list[ScoredDocument],
        keyword_results: list[ScoredDocument],
        top_k: int,
        k: int = 60,
    ) -> list[ScoredDocument]:
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, ScoredDocument] = {}

        for rank, r in enumerate(vector_results):
            rrf_scores[r.chunk.id] = rrf_scores.get(r.chunk.id, 0.0) + self._vw / (k + rank + 1)
            chunk_map[r.chunk.id] = r

        for rank, r in enumerate(keyword_results):
            rrf_scores[r.chunk.id] = rrf_scores.get(r.chunk.id, 0.0) + self._kw / (k + rank + 1)
            if r.chunk.id not in chunk_map:
                chunk_map[r.chunk.id] = r

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        results = []
        for cid in sorted_ids:
            orig = chunk_map[cid]
            results.append(
                ScoredDocument(
                    chunk=orig.chunk,
                    score=rrf_scores[cid],
                    vector_score=orig.vector_score,
                    keyword_score=orig.keyword_score,
                )
            )
        return results
