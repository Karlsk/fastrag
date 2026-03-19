"""Page 2 - RAG Chat: Retrieve -> Rerank -> LLM answer."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils import get_collection_name, get_embedding, get_llm, get_settings, get_store, init_sidebar, run_async

logger = logging.getLogger("fastrag.app.rag_chat")

init_sidebar()

st.title("RAG Chat")

# ----- Retrieval config -----
with st.expander("Retrieval Settings", expanded=False):
    col1, col2 = st.columns(2)
    strategy = col1.selectbox("Retrieval Strategy", ["Vector", "Keyword", "Hybrid"])
    top_k = col2.slider("Top K", 1, 20, 5)

    if strategy == "Hybrid":
        col3, col4 = st.columns(2)
        vector_weight = col3.slider("Vector Weight", 0.0, 1.0, 0.7, step=0.1)
        keyword_weight = col4.slider("Keyword Weight", 0.0, 1.0, 0.3, step=0.1)
    else:
        vector_weight, keyword_weight = 0.7, 0.3

    enable_rerank = st.checkbox("Enable Rerank (HybridSimilarity)", value=False)

# ----- Chat history -----
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----- Chat input -----
# Persist query in session_state so it survives Streamlit reruns.
query = st.chat_input("Ask a question about your documents...")
if query:
    st.session_state["_pending_query"] = query
    st.session_state["chat_history"].append({"role": "user", "content": query})

# ----- Process pending query (rerun-safe) -----
if "_pending_query" in st.session_state:
    pending_query = st.session_state.pop("_pending_query")
    logger.info("=== Processing query: %s ===", pending_query[:80])

    with st.chat_message("user"):
        st.markdown(pending_query)

    settings = get_settings()
    collection = get_collection_name()

    try:
        # -- Retrieve --
        logger.info("Step 1: Retrieve (strategy=%s, top_k=%d)", strategy, top_k)
        from fastrag.retrieval.vector import VectorRetriever

        store = get_store(settings)
        embedding = get_embedding(settings)
        results = []

        if strategy == "Vector":
            retriever = VectorRetriever(store=store, embedding=embedding, collection=collection)
            results = run_async(retriever.retrieve(pending_query, top_k=top_k))

        elif strategy == "Keyword":
            from fastrag.retrieval.keyword import KeywordRetriever

            chunks = st.session_state.get("chunks", [])
            if not chunks and store.has_collection(collection):
                chunks = store.list_chunks(collection)
                st.session_state["chunks"] = chunks
            if not chunks:
                st.warning("No chunks found. Please process documents first or use Vector retrieval.")
                st.stop()
            retriever = KeywordRetriever(chunks=chunks)
            results = run_async(retriever.retrieve(pending_query, top_k=top_k))

        else:  # Hybrid
            from fastrag.retrieval.hybrid import HybridRetriever
            from fastrag.retrieval.keyword import KeywordRetriever

            vec_retriever = VectorRetriever(store=store, embedding=embedding, collection=collection)
            chunks = st.session_state.get("chunks", [])
            if not chunks and store.has_collection(collection):
                chunks = store.list_chunks(collection)
                st.session_state["chunks"] = chunks
            if not chunks:
                st.warning("No chunks for keyword retrieval. Using vector-only.")
                results = run_async(vec_retriever.retrieve(pending_query, top_k=top_k))
            else:
                kw_retriever = KeywordRetriever(chunks=chunks)
                hybrid = HybridRetriever(
                    vector_retriever=vec_retriever,
                    keyword_retriever=kw_retriever,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                )
                results = run_async(hybrid.retrieve(pending_query, top_k=top_k))

        logger.info("Retrieve done: %d results", len(results))

        # -- Rerank --
        if enable_rerank and results:
            from fastrag.rerank.hybrid_similarity import HybridSimilarityReranker

            reranker = HybridSimilarityReranker()
            results = reranker.rerank(pending_query, results, top_k=top_k)

        # -- LLM Generate --
        context = "\n\n---\n\n".join([r.chunk.content for r in results])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question based on the provided context. "
                    "If the context doesn't contain relevant information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {pending_query}",
            },
        ]

        logger.info("Step 2: LLM call (model=%s)", settings.LLM_MODEL)
        llm = get_llm(settings)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            full_response = ""
            for token in llm.sync_stream_chat(messages):
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response or "_No response from model_")
        logger.info("LLM done: len=%d", len(full_response))
        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})

        # -- Show retrieved chunks in sidebar --
        with st.sidebar:
            st.divider()
            st.subheader(f"Retrieved Chunks ({len(results)})")
            for i, r in enumerate(results):
                score_str = f"score={r.score:.4f}"
                if r.vector_score is not None:
                    score_str += f" | vec={r.vector_score:.4f}"
                if r.keyword_score is not None:
                    score_str += f" | kw={r.keyword_score:.4f}"
                with st.expander(f"Chunk {i + 1} ({score_str})", expanded=False):
                    st.text(r.chunk.content[:500])

    except Exception as e:
        logger.error("Processing failed: %s", e, exc_info=True)
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")

    logger.info("=== Query processing complete ===")
