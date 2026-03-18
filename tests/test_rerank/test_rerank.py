import numpy as np
import pytest

from fastrag.models.document import Chunk, Metadata
from fastrag.models.search import ScoredDocument
from fastrag.rerank.hybrid_similarity import HybridSimilarityReranker


@pytest.fixture
def scored_documents():
    return [
        ScoredDocument(
            chunk=Chunk(id="1", content="Python is great for data science", document_id="doc1", metadata=Metadata()),
            score=0.8,
            vector_score=0.8,
        ),
        ScoredDocument(
            chunk=Chunk(id="2", content="Java is used in enterprise", document_id="doc1", metadata=Metadata()),
            score=0.6,
            vector_score=0.6,
        ),
        ScoredDocument(
            chunk=Chunk(id="3", content="Python machine learning libraries", document_id="doc2", metadata=Metadata()),
            score=0.7,
            vector_score=0.7,
        ),
    ]


class TestHybridSimilarityReranker:
    def test_rerank(self, scored_documents):
        reranker = HybridSimilarityReranker(token_weight=0.3, vector_weight=0.7)
        results = reranker.rerank("Python data science", scored_documents)
        assert len(results) == 3
        assert all(isinstance(r, ScoredDocument) for r in results)
        # Should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_rerank_top_k(self, scored_documents):
        reranker = HybridSimilarityReranker()
        results = reranker.rerank("Python", scored_documents, top_k=2)
        assert len(results) == 2

    def test_rerank_empty(self):
        reranker = HybridSimilarityReranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_normalize_scores(self):
        from fastrag.rerank.base import BaseReranker
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = BaseReranker._normalize_scores(scores)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)
