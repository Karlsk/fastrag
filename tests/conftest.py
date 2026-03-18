import pytest


@pytest.fixture
def sample_text():
    return """FastRAG is a modular RAG engine.
It supports document parsing, text cleaning, splitting, retrieval, reranking, and pipelines.

## Features

- Document Loader: Parse PDF, DOCX, TXT, Markdown, HTML, CSV, Excel
- Text Cleaner: HTML cleaning, whitespace normalization, Unicode normalization
- Text Splitter: Token-based, recursive, markdown, sentence splitting
- Retrieval: Vector search, BM25 keyword search, hybrid fusion
- Reranker: OpenAI API, cross-encoder, hybrid similarity
- Pipeline: JSON DSL, graph execution, conditional branching

## Architecture

The project uses abstract base classes (ABC) for each module.
Custom implementations can be added by inheriting the base class.
"""


@pytest.fixture
def sample_txt_file(tmp_path, sample_text):
    p = tmp_path / "sample.txt"
    p.write_text(sample_text, encoding="utf-8")
    return p


@pytest.fixture
def sample_md_file(tmp_path, sample_text):
    p = tmp_path / "sample.md"
    p.write_text(sample_text, encoding="utf-8")
    return p
