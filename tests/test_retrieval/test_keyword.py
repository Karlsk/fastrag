import pytest

from fastrag.models.document import Chunk, Metadata
from fastrag.models.search import ScoredDocument
from fastrag.retrieval.keyword import KeywordRetriever


@pytest.fixture
def sample_chunks():
    return [
        Chunk(id="1", content="Python is a programming language", document_id="doc1", metadata=Metadata()),
        Chunk(id="2", content="Java is also a programming language", document_id="doc1", metadata=Metadata()),
        Chunk(id="3", content="RAG stands for retrieval augmented generation", document_id="doc2", metadata=Metadata()),
        Chunk(id="4", content="Vector databases store embeddings", document_id="doc2", metadata=Metadata()),
        Chunk(id="5", content="Python is great for machine learning", document_id="doc3", metadata=Metadata()),
    ]


class TestKeywordRetriever:
    @pytest.mark.asyncio
    async def test_basic_retrieval(self, sample_chunks):
        retriever = KeywordRetriever(chunks=sample_chunks)
        results = await retriever.retrieve("Python programming", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, ScoredDocument) for r in results)
        # Python-related chunks should rank higher
        assert "python" in results[0].chunk.content.lower() or "programming" in results[0].chunk.content.lower()

    @pytest.mark.asyncio
    async def test_top_k(self, sample_chunks):
        retriever = KeywordRetriever(chunks=sample_chunks)
        results = await retriever.retrieve("programming language", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_no_match(self, sample_chunks):
        retriever = KeywordRetriever(chunks=sample_chunks)
        results = await retriever.retrieve("quantum physics", top_k=3)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_add_chunks(self):
        retriever = KeywordRetriever()
        retriever.add_chunks([
            Chunk(id="1", content="Hello world", document_id="doc1", metadata=Metadata()),
        ])
        results = await retriever.retrieve("hello", top_k=1)
        assert len(results) == 1
