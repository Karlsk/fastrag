"""Page 1 - Document Processing: Upload -> Parse -> Clean -> Split -> Embed -> Store."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from app/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils import get_collection_name, get_embedding, get_settings, get_store, init_sidebar, run_async

init_sidebar()

st.title("Document Processing")
st.markdown(
    "Upload a file and walk through the full processing pipeline: **Parse -> Clean -> Split -> Embed -> Store**"
)

# ----- Step 1: Upload & Parse -----
st.subheader("Step 1: Upload & Parse")

uploaded = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx", "md", "html", "csv", "xlsx"],
)

if uploaded is not None:
    from fastrag.loader import get_loader

    with st.spinner("Parsing document..."):
        try:
            loader = get_loader(uploaded.name)
            documents = loader.load(uploaded.read())
            st.session_state["parsed_docs"] = documents
            st.success(f"Parsed **{len(documents)}** document(s)")
        except Exception as e:
            st.error(f"Parse failed: {e}")
            documents = []

    if documents:
        with st.expander("Preview parsed documents", expanded=False):
            for i, doc in enumerate(documents[:20]):
                st.markdown(f"**Document {i + 1}** (source: `{doc.metadata.source}`)")
                st.text(doc.content[:500] + ("..." if len(doc.content) > 500 else ""))
                st.divider()

# ----- Step 2: Clean -----
if "parsed_docs" in st.session_state and st.session_state["parsed_docs"]:
    st.subheader("Step 2: Clean")

    from fastrag.cleaner import (
        ControlCharRemover,
        HTMLCleaner,
        UnicodeNormalizer,
        WhitespaceCleaner,
    )

    cleaner_options = {
        "HTMLCleaner": HTMLCleaner,
        "WhitespaceCleaner": WhitespaceCleaner,
        "UnicodeNormalizer": UnicodeNormalizer,
        "ControlCharRemover": ControlCharRemover,
    }

    selected = st.multiselect(
        "Select cleaners (applied in order)",
        options=list(cleaner_options.keys()),
        default=["WhitespaceCleaner", "UnicodeNormalizer"],
    )

    if st.button("Run Cleaners"):
        if not selected:
            st.warning("Select at least one cleaner")
        else:
            instances = [cleaner_options[name]() for name in selected]
            pipeline = instances[0]
            for inst in instances[1:]:
                pipeline = pipeline + inst

            docs = st.session_state["parsed_docs"]
            from fastrag.models.document import Document

            cleaned_docs = []
            for doc in docs:
                cleaned_text = pipeline.clean(doc.content)
                cleaned_docs.append(Document(content=cleaned_text, metadata=doc.metadata))
            st.session_state["cleaned_docs"] = cleaned_docs
            st.success(f"Cleaned **{len(cleaned_docs)}** document(s)")

            with st.expander("Before / After comparison", expanded=False):
                for i, (orig, cleaned) in enumerate(zip(docs[:5], cleaned_docs[:5])):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Original {i + 1}**")
                        st.text(orig.content[:300])
                    with col2:
                        st.markdown(f"**Cleaned {i + 1}**")
                        st.text(cleaned.content[:300])
                    st.divider()

# ----- Step 3: Split -----
source_key = "cleaned_docs" if "cleaned_docs" in st.session_state else "parsed_docs"
if source_key in st.session_state and st.session_state[source_key]:
    st.subheader("Step 3: Split")

    from fastrag.splitter import (
        MarkdownSplitter,
        RecursiveSplitter,
        SentenceSplitter,
        TokenSplitter,
    )

    splitter_options = {
        "TokenSplitter": TokenSplitter,
        "RecursiveSplitter": RecursiveSplitter,
        "MarkdownSplitter": MarkdownSplitter,
        "SentenceSplitter": SentenceSplitter,
    }

    col1, col2, col3 = st.columns(3)
    splitter_name = col1.selectbox("Splitter", list(splitter_options.keys()))
    chunk_size = col2.slider("Chunk size (tokens)", 64, 2048, 512, step=64)
    chunk_overlap = col3.slider("Chunk overlap (tokens)", 0, 256, 32, step=16)

    if st.button("Run Splitter"):
        splitter_cls = splitter_options[splitter_name]
        splitter = splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = st.session_state[source_key]

        all_chunks = []
        for doc in docs:
            chunks = splitter.split(doc)
            all_chunks.extend(chunks)

        st.session_state["chunks"] = all_chunks
        st.success(f"Split into **{len(all_chunks)}** chunk(s)")

        from fastrag.utils.tokenizer import count_tokens

        with st.expander("Chunk preview", expanded=False):
            for i, chunk in enumerate(all_chunks[:30]):
                tokens = count_tokens(chunk.content)
                st.markdown(f"**Chunk {i + 1}** | {tokens} tokens | doc: `{chunk.document_id[:8]}...`")
                st.text(chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""))
                st.divider()

# ----- Step 4: Embed & Store -----
if "chunks" in st.session_state and st.session_state["chunks"]:
    st.subheader("Step 4: Embed & Store to Milvus")

    collection = get_collection_name()
    st.info(f"Target collection: **{collection}**")

    if st.button("Embed & Store"):
        chunks = st.session_state["chunks"]
        settings = get_settings()

        with st.spinner("Computing embeddings..."):
            try:
                embedding = get_embedding(settings)
                batch_size = 64
                all_embeddings: list[list[float]] = []
                progress = st.progress(0.0)

                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start : start + batch_size]
                    texts = [c.content for c in batch]
                    embs = run_async(embedding.embed(texts))
                    all_embeddings.extend(embs)
                    progress.progress(min(1.0, (start + batch_size) / len(chunks)))

                for chunk, emb in zip(chunks, all_embeddings):
                    chunk.embedding = emb

                st.success(f"Computed **{len(all_embeddings)}** embeddings (dim={len(all_embeddings[0])})")
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                st.stop()

        with st.spinner("Inserting into Milvus..."):
            try:
                store = get_store(settings)
                if not store.has_collection(collection):
                    dim = len(all_embeddings[0])
                    store.create_collection(collection, dimension=dim)
                    st.info(f"Created collection '{collection}' (dim={dim})")

                ids = store.insert(collection, chunks)
                st.success(f"Stored **{len(ids)}** chunks into '{collection}'")

                cnt = store.count(collection)
                st.info(f"Collection '{collection}' now has **{cnt}** total chunks")
            except Exception as e:
                st.error(f"Store failed: {e}")
