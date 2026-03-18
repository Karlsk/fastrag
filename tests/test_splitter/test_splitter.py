import pytest

from fastrag.models.document import Document, Metadata
from fastrag.splitter import (
    BaseSplitter,
    MarkdownSplitter,
    RecursiveSplitter,
    SentenceSplitter,
    TokenSplitter,
)


@pytest.fixture
def long_document():
    text = "\n".join([f"This is sentence number {i}." for i in range(100)])
    return Document(content=text, metadata=Metadata(source="test.txt", file_type="txt"))


@pytest.fixture
def markdown_document():
    return Document(
        content="""# Introduction
This is the introduction section with some text.

## Chapter 1
This is chapter 1 content. It has multiple paragraphs.
More content here.

## Chapter 2
This is chapter 2 content.

### Section 2.1
Sub-section content here.
""",
        metadata=Metadata(source="test.md", file_type="markdown"),
    )


class TestTokenSplitter:
    def test_basic_split(self, long_document):
        splitter = TokenSplitter(chunk_size=50)
        chunks = splitter.split(long_document)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_id == long_document.id
            assert chunk.content.strip()

    def test_overlap(self, long_document):
        splitter = TokenSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split(long_document)
        assert len(chunks) > 1

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            TokenSplitter(chunk_size=50, chunk_overlap=50)

    def test_empty_document(self):
        doc = Document(content="", metadata=Metadata())
        splitter = TokenSplitter()
        assert splitter.split(doc) == []


class TestRecursiveSplitter:
    def test_basic_split(self, long_document):
        splitter = RecursiveSplitter(chunk_size=50)
        chunks = splitter.split(long_document)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_id == long_document.id

    def test_custom_separators(self, long_document):
        splitter = RecursiveSplitter(chunk_size=50, separators=["\n", ".", " "])
        chunks = splitter.split(long_document)
        assert len(chunks) > 1


class TestMarkdownSplitter:
    def test_split_by_headers(self, markdown_document):
        splitter = MarkdownSplitter(chunk_size=200)
        chunks = splitter.split(markdown_document)
        assert len(chunks) >= 1
        assert any("#" in c.content for c in chunks)

    def test_empty(self):
        doc = Document(content="  ", metadata=Metadata())
        splitter = MarkdownSplitter()
        assert splitter.split(doc) == []


class TestSentenceSplitter:
    def test_basic_split(self, long_document):
        splitter = SentenceSplitter(chunk_size=50)
        chunks = splitter.split(long_document)
        assert len(chunks) > 1

    def test_chunk_index(self, long_document):
        splitter = SentenceSplitter(chunk_size=50)
        chunks = splitter.split(long_document)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
