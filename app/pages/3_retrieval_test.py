"""Page 3 - Retrieval Test: Compare Vector / Keyword / Hybrid side by side."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils import get_collection_name, get_embedding, get_settings, get_store, init_sidebar, run_async

init_sidebar()

st.title("Retrieval Test")
st.markdown("Compare **Vector**, **Keyword (BM25)**, and **Hybrid** retrieval results side by side.")

# ----- Query input -----
col_q, col_k = st.columns([3, 1])
query = col_q.text_input("Query", placeholder="Enter your search query...")
top_k = col_k.number_input("Top K", value=5, min_value=1, max_value=50)

# ----- Hybrid params -----
with st.expander("Hybrid Parameters", expanded=False):
    col1, col2, col3 = st.columns(3)
    vector_weight = col1.slider("Vector Weight", 0.0, 1.0, 0.7, step=0.1, key="rt_vw")
    keyword_weight = col2.slider("Keyword Weight", 0.0, 1.0, 0.3, step=0.1, key="rt_kw")
    fusion_method = col3.selectbox("Fusion Method", ["weighted_sum", "rrf"])

enable_rerank = st.checkbox("Apply Rerank (HybridSimilarity) to all results", value=False)


def _display_results(results: list, prefix: str) -> None:
    """Display a list of ScoredDocument results."""
    if not results:
        st.info("No results")
        return

    for i, r in enumerate(results):
        scores = [f"**score**: {r.score:.4f}"]
        if r.vector_score is not None:
            scores.append(f"vec: {r.vector_score:.4f}")
        if r.keyword_score is not None:
            scores.append(f"kw: {r.keyword_score:.4f}")
        score_line = " | ".join(scores)

        with st.container(border=True):
            st.markdown(f"**#{i + 1}** {score_line}")
            content = r.chunk.content
            st.text(content[:300] + ("..." if len(content) > 300 else ""))


# ----- Run -----
if st.button("Search", type="primary") and query:
    settings = get_settings()
    collection = get_collection_name()
    store = get_store(settings)
    embedding = get_embedding(settings)

    from fastrag.retrieval.hybrid import HybridRetriever
    from fastrag.retrieval.keyword import KeywordRetriever
    from fastrag.retrieval.vector import VectorRetriever

    vec_retriever = VectorRetriever(store=store, embedding=embedding, collection=collection)

    # Try to get chunks for keyword retrieval
    chunks = st.session_state.get("chunks", [])
    if not chunks and store.has_collection(collection):
        with st.spinner("Loading chunks from Milvus for BM25 index..."):
            chunks = store.list_chunks(collection)
            st.session_state["chunks"] = chunks
    has_keyword = bool(chunks)
    kw_retriever = KeywordRetriever(chunks=chunks) if has_keyword else None

    col_v, col_kw, col_h = st.columns(3)

    # -- Vector results --
    with col_v:
        st.subheader("Vector")
        with st.spinner("Searching..."):
            try:
                vec_results = run_async(vec_retriever.retrieve(query, top_k=top_k))
                _display_results(vec_results, "vec")
            except Exception as e:
                st.error(str(e))
                vec_results = []

    # -- Keyword results --
    with col_kw:
        st.subheader("Keyword (BM25)")
        if not has_keyword:
            st.warning("No chunks in session. Process documents first.")
            kw_results = []
        else:
            with st.spinner("Searching..."):
                try:
                    kw_results = run_async(kw_retriever.retrieve(query, top_k=top_k))
                    _display_results(kw_results, "kw")
                except Exception as e:
                    st.error(str(e))
                    kw_results = []

    # -- Hybrid results --
    with col_h:
        st.subheader("Hybrid")
        if not has_keyword:
            st.warning("Keyword index unavailable, showing vector-only.")
            hybrid_results = vec_results
            _display_results(hybrid_results, "hyb")
        else:
            with st.spinner("Searching..."):
                try:
                    hybrid = HybridRetriever(
                        vector_retriever=vec_retriever,
                        keyword_retriever=kw_retriever,
                        vector_weight=vector_weight,
                        keyword_weight=keyword_weight,
                        fusion_method=fusion_method,
                    )
                    hybrid_results = run_async(hybrid.retrieve(query, top_k=top_k))
                    _display_results(hybrid_results, "hyb")
                except Exception as e:
                    st.error(str(e))
                    hybrid_results = []

    # -- Rerank comparison --
    if enable_rerank:
        st.divider()
        st.subheader("After Rerank (HybridSimilarity)")
        from fastrag.rerank.hybrid_similarity import HybridSimilarityReranker

        reranker = HybridSimilarityReranker()

        col_rv, col_rkw, col_rh = st.columns(3)

        with col_rv:
            st.markdown("**Vector (reranked)**")
            if vec_results:
                reranked = reranker.rerank(query, vec_results, top_k=top_k)
                _display_results(reranked, "rvec")

        with col_rkw:
            st.markdown("**Keyword (reranked)**")
            if kw_results:
                reranked = reranker.rerank(query, kw_results, top_k=top_k)
                _display_results(reranked, "rkw")

        with col_rh:
            st.markdown("**Hybrid (reranked)**")
            if hybrid_results:
                reranked = reranker.rerank(query, hybrid_results, top_k=top_k)
                _display_results(reranked, "rhyb")
