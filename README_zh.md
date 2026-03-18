# FastRAG

一个模块化的 RAG（检索增强生成）引擎，具备可扩展的文档处理、检索和低代码 Pipeline 能力。

[English Documentation](README.md)

## 特性

FastRAG 包含 6 大核心模块，每个模块都通过抽象基类（ABC）定义接口，方便自定义扩展：

| 模块 | 抽象基类 | 内置实现 |
|------|---------|---------|
| **文档解析** | `BaseLoader` | TXT, PDF, DOCX, Markdown, HTML, CSV, Excel |
| **文本清洗** | `BaseCleaner` | HTML标签清除, 空白规范化, Unicode规范化, 控制字符移除, 正则替换 |
| **分割策略** | `BaseSplitter` | 按Token分割, 递归分割, Markdown标题分割, 句子边界分割 |
| **检索策略** | `BaseRetriever` / `BaseVectorStore` | 向量检索, BM25关键词检索, 混合融合检索（加权求和/RRF） |
| **重排序** | `BaseReranker` | OpenAI兼容API, 本地Cross-Encoder, 混合相似度 |
| **Pipeline** | `ComponentBase` / `Graph` | JSON DSL 驱动的图执行引擎，支持条件分支 |

## 快速开始

### 安装

```bash
# 基础安装
uv sync

# 安装所有可选依赖
uv sync --all-extras

# 或安装特定依赖组
uv sync --extra pdf --extra milvus
```

### 配置

复制 `.env.example` 到 `.env` 并修改配置：

```bash
cp .env.example .env
```

主要环境变量（以 `FASTRAG_` 为前缀）：

```
FASTRAG_OPENAI_API_KEY=sk-xxx
FASTRAG_OPENAI_BASE_URL=https://api.openai.com/v1
FASTRAG_LLM_MODEL=gpt-4o-mini
FASTRAG_EMBEDDING_MODEL=text-embedding-3-small
FASTRAG_MILVUS_URI=http://localhost:19530
```

### 使用示例

#### 文档加载与处理

```python
from fastrag.loader import get_loader
from fastrag.cleaner import HTMLCleaner, WhitespaceCleaner
from fastrag.splitter import TokenSplitter

# 加载文档
loader = get_loader("paper.pdf")
docs = loader.load("paper.pdf")

# 文本清洗（支持 + 运算符链式组合）
cleaner = HTMLCleaner() + WhitespaceCleaner()
for doc in docs:
    doc.content = cleaner.clean(doc.content)

# 分割为 chunks
splitter = TokenSplitter(chunk_size=512, chunk_overlap=50)
chunks = []
for doc in docs:
    chunks.extend(splitter.split(doc))
```

#### 检索与重排

```python
from fastrag.retrieval import KeywordRetriever, HybridRetriever, VectorRetriever
from fastrag.rerank import HybridSimilarityReranker
from fastrag.llm import OpenAIEmbedding
from fastrag.store import MilvusVectorStore

# 设置向量检索
store = MilvusVectorStore()
embedding = OpenAIEmbedding()
vector_retriever = VectorRetriever(store, embedding, collection="my_docs")

# 设置混合检索
keyword_retriever = KeywordRetriever(chunks=chunks)
hybrid = HybridRetriever(vector_retriever, keyword_retriever, vector_weight=0.7, keyword_weight=0.3)

# 检索并重排
results = await hybrid.retrieve("什么是RAG？", top_k=10)
reranker = HybridSimilarityReranker(token_weight=0.3, vector_weight=0.7)
reranked = reranker.rerank("什么是RAG？", results, top_k=5)
```

#### 低代码 Pipeline（JSON DSL）

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
                    "prompt_template": "根据以下上下文:\n{retrieval_0@content}\n\n回答问题: {sys.query}",
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
    result = await graph.run_to_completion(query="什么是RAG？")
    print(result["message"])

asyncio.run(main())
```

### 自定义扩展

每个模块都支持通过继承基类来添加自定义实现：

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

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest tests/ -v

# 代码检查与格式化
uv run ruff check src/
uv run ruff format src/
```

## 架构

```
              Pipeline (JSON DSL 图执行引擎)
  ┌──────┐  ┌───────────┐  ┌─────┐  ┌─────────┐  ┌──────┐
  │Begin │─▶│ Retrieval  │─▶│ LLM │─▶│ Message │  │Switch│
  └──────┘  └─────┬─────┘  └──┬──┘  └─────────┘  └──────┘
              使用 │       使用 │
  ┌────────────────▼──┐  ┌────▼──────────┐
  │  HybridRetriever  │  │  OpenAIChat   │
  └────────┬──────────┘  └──────────────┘
           │
  ┌────────▼──────────┐
  │ MilvusVectorStore  │
  └───────────────────┘

         文档处理链
  ┌────────┐  ┌─────────┐  ┌──────────┐
  │ Loader │─▶│ Cleaner │─▶│ Splitter │─▶ Chunks
  └────────┘  │Pipeline │  └──────────┘
              └─────────┘
```

## 许可证

MIT
