# FastRAG

A modular RAG (Retrieval-Augmented Generation) engine with extensible document processing, retrieval, and low-code pipeline.

[中文文档](README_zh.md)

## Features

FastRAG covers 6 core modules, each with an abstract base class (ABC) interface for easy extension:

| Module | Base Class | Built-in Implementations |
|--------|-----------|------------------------|
| **Document Loader** | `BaseLoader` | TXT, PDF, DOCX, Markdown, HTML, CSV, Excel |
| **Text Cleaner** | `BaseCleaner` | HTML tag removal, whitespace normalization, Unicode normalization, control char removal, regex |
| **Text Splitter** | `BaseSplitter` | Token-based, recursive, Markdown header, sentence boundary |
| **Retrieval** | `BaseRetriever` / `BaseVectorStore` | Vector search, BM25 keyword search, hybrid fusion (weighted sum / RRF) |
| **Rerank** | `BaseReranker` | OpenAI-compatible API, local cross-encoder, hybrid similarity |
| **Pipeline** | `ComponentBase` / `Graph` | JSON DSL-driven graph engine with conditional branching |

## Quick Start

### Installation

```bash
# Basic install
uv sync

# With all optional dependencies
uv sync --all-extras

# Or install specific extras
uv sync --extra pdf --extra milvus
```

### Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key environment variables (prefixed with `FASTRAG_`):

```
FASTRAG_OPENAI_API_KEY=sk-xxx
FASTRAG_OPENAI_BASE_URL=https://api.openai.com/v1
FASTRAG_LLM_MODEL=gpt-4o-mini
FASTRAG_EMBEDDING_MODEL=text-embedding-3-small
FASTRAG_MILVUS_URI=http://localhost:19530
```

### Usage Examples

#### Document Loading & Processing

```python
from fastrag.loader import get_loader
from fastrag.cleaner import HTMLCleaner, WhitespaceCleaner
from fastrag.splitter import TokenSplitter

# Load a document
loader = get_loader("paper.pdf")
docs = loader.load("paper.pdf")

# Clean text
cleaner = HTMLCleaner() + WhitespaceCleaner()
for doc in docs:
    doc.content = cleaner.clean(doc.content)

# Split into chunks
splitter = TokenSplitter(chunk_size=512, chunk_overlap=50)
chunks = []
for doc in docs:
    chunks.extend(splitter.split(doc))
```

#### Retrieval & Rerank

```python
from fastrag.retrieval import KeywordRetriever, HybridRetriever, VectorRetriever
from fastrag.rerank import HybridSimilarityReranker
from fastrag.llm import OpenAIEmbedding
from fastrag.store import MilvusVectorStore

# Set up vector retrieval
store = MilvusVectorStore()
embedding = OpenAIEmbedding()
vector_retriever = VectorRetriever(store, embedding, collection="my_docs")

# Set up hybrid retrieval
keyword_retriever = KeywordRetriever(chunks=chunks)
hybrid = HybridRetriever(vector_retriever, keyword_retriever, vector_weight=0.7, keyword_weight=0.3)

# Retrieve and rerank
results = await hybrid.retrieve("What is RAG?", top_k=10)
reranker = HybridSimilarityReranker(token_weight=0.3, vector_weight=0.7)
reranked = reranker.rerank("What is RAG?", results, top_k=5)
```

#### Low-Code Pipeline (JSON DSL)

```python
import asyncio
from fastrag.pipeline import Graph

dsl = {
    "components": {
        "begin": {
            "obj": {"component_name": "Begin", "params": {}},
            "upstream": [],
            "downstream": ["retrieval_0"],
        },
        "retrieval_0": {
            "obj": {"component_name": "Retrieval", "params": {"collection": "my_docs", "top_k": 5}},
            "upstream": ["begin"],
            "downstream": ["llm_0"],
        },
        "llm_0": {
            "obj": {
                "component_name": "LLM",
                "params": {
                    "prompt_template": "Based on the context:\n{retrieval_0@content}\n\nAnswer: {sys.query}",
                },
            },
            "upstream": ["retrieval_0"],
            "downstream": ["msg_0"],
        },
        "msg_0": {
            "obj": {"component_name": "Message", "params": {"content": "{llm_0@content}"}},
            "upstream": ["llm_0"],
            "downstream": [],
        },
    },
    "globals": {"sys.query": ""},
    "path": ["begin"],
}

async def main():
    graph = Graph(dsl)
    result = await graph.run_to_completion(query="What is RAG?")
    print(result["message"])

asyncio.run(main())
```

### Custom Extensions

Every module supports custom implementations by inheriting the base class:

```python
from fastrag.loader.base import BaseLoader
from fastrag.models.document import Document, Metadata

class MyCustomLoader(BaseLoader):
    supported_extensions = [".custom"]

    def load(self, source, **kwargs):
        raw = self._read_bytes(source)
        text = raw.decode("utf-8")
        return [Document(content=text, metadata=Metadata(source=str(source), file_type="custom"))]
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint & format
uv run ruff check src/
uv run ruff format src/
```

## Architecture

```
              Pipeline (JSON DSL Graph Engine)
  ┌──────┐  ┌───────────┐  ┌─────┐  ┌─────────┐  ┌──────┐
  │Begin │─▶│ Retrieval  │─▶│ LLM │─▶│ Message │  │Switch│
  └──────┘  └─────┬─────┘  └──┬──┘  └─────────┘  └──────┘
              uses │       uses │
  ┌────────────────▼──┐  ┌────▼──────────┐
  │  HybridRetriever  │  │  OpenAIChat   │
  └────────┬──────────┘  └──────────────┘
           │
  ┌────────▼──────────┐
  │ MilvusVectorStore  │
  └───────────────────┘

         Document Processing Chain
  ┌────────┐  ┌─────────┐  ┌──────────┐
  │ Loader │─▶│ Cleaner │─▶│ Splitter │─▶ Chunks
  └────────┘  │Pipeline │  └──────────┘
              └─────────┘
```

## License

MIT
