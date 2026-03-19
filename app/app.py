"""FastRAG - Streamlit main entry point."""

from __future__ import annotations

import streamlit as st
from utils import init_sidebar

st.set_page_config(page_title="FastRAG", page_icon="⚡", layout="wide")

init_sidebar()

st.title("FastRAG")
st.markdown("**A modular RAG engine with extensible document processing, retrieval, and low-code pipeline.**")

st.divider()

st.subheader("Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
**Loader**
Upload and parse documents (TXT, PDF, DOCX, Markdown, HTML, CSV, Excel).

**Cleaner**
Chain multiple text cleaners: HTML removal, whitespace normalization, unicode, control chars, regex.
"""
    )

with col2:
    st.markdown(
        """
**Splitter**
Split documents into chunks: Token, Recursive, Markdown-aware, Sentence boundary.

**Retrieval**
Vector (Milvus), Keyword (BM25), Hybrid (weighted / RRF fusion).
"""
    )

with col3:
    st.markdown(
        """
**Rerank**
Re-rank retrieved chunks: OpenAI API, CrossEncoder, HybridSimilarity.

**Pipeline**
Low-code JSON DSL graph engine with Begin, LLM, Retrieval, Message, Categorize, Switch components.
"""
    )

st.divider()

st.subheader("Quick Start")
st.markdown(
    """
1. Configure **API Key**, **Model**, and **Milvus URI** in the left sidebar.
2. Create a **Collection** (set the correct embedding dimension).
3. Go to **Document Processing** to upload, parse, clean, split, and store documents.
4. Go to **RAG Chat** to ask questions over your documents.
5. Use **Retrieval Test** to compare different retrieval strategies.
6. Use **Pipeline Editor** to design and run custom JSON DSL pipelines.
"""
)
