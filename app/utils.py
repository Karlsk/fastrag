"""Shared utilities for FastRAG Streamlit app."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import streamlit as st

from fastrag.config import FastRAGSettings
from fastrag.llm.openai_embedding import OpenAIEmbedding
from fastrag.llm.openai_llm import OpenAIChat
from fastrag.store.milvus import MilvusVectorStore


def run_async(coro: Any) -> Any:
    """Run an async coroutine in Streamlit's synchronous context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


_ENV_MAPPING: dict[str, str] = {
    "api_key": "FASTRAG_OPENAI_API_KEY",
    "base_url": "FASTRAG_OPENAI_BASE_URL",
    "llm_model": "FASTRAG_LLM_MODEL",
    "embedding_model": "FASTRAG_EMBEDDING_MODEL",
    "milvus_uri": "FASTRAG_MILVUS_URI",
}


def apply_settings_to_env() -> None:
    """Write session_state settings to os.environ with FASTRAG_ prefix."""
    for key, env_key in _ENV_MAPPING.items():
        val = st.session_state.get(key, "")
        if val:
            os.environ[env_key] = val


def get_settings() -> FastRAGSettings:
    """Build FastRAGSettings from current session_state."""
    apply_settings_to_env()
    return FastRAGSettings()


def get_embedding(settings: FastRAGSettings | None = None) -> OpenAIEmbedding:
    """Return an OpenAIEmbedding instance."""
    return OpenAIEmbedding(settings=settings or get_settings())


def get_llm(settings: FastRAGSettings | None = None) -> OpenAIChat:
    """Return an OpenAIChat instance."""
    return OpenAIChat(settings=settings or get_settings())


def get_store(settings: FastRAGSettings | None = None) -> MilvusVectorStore:
    """Return a MilvusVectorStore instance."""
    return MilvusVectorStore(settings=settings or get_settings())


def get_collection_name() -> str:
    """Return current collection name from session_state."""
    return st.session_state.get("collection_name", "default")


def init_sidebar() -> None:
    """Render the shared sidebar with settings and collection management."""
    with st.sidebar:
        st.title("Settings")

        st.text_input(
            "OpenAI API Key",
            type="password",
            key="api_key",
            value=st.session_state.get("api_key", os.environ.get("FASTRAG_OPENAI_API_KEY", "")),
        )
        st.text_input(
            "Base URL",
            key="base_url",
            value=st.session_state.get(
                "base_url", os.environ.get("FASTRAG_OPENAI_BASE_URL", "https://api.openai.com/v1")
            ),
        )
        st.text_input(
            "LLM Model",
            key="llm_model",
            value=st.session_state.get("llm_model", os.environ.get("FASTRAG_LLM_MODEL", "gpt-4o-mini")),
        )
        st.text_input(
            "Embedding Model",
            key="embedding_model",
            value=st.session_state.get(
                "embedding_model", os.environ.get("FASTRAG_EMBEDDING_MODEL", "text-embedding-3-small")
            ),
        )
        st.text_input(
            "Milvus URI",
            key="milvus_uri",
            value=st.session_state.get("milvus_uri", os.environ.get("FASTRAG_MILVUS_URI", "http://localhost:19530")),
        )
        st.text_input(
            "Collection Name",
            key="collection_name",
            value=st.session_state.get("collection_name", "default"),
        )

        st.divider()
        st.subheader("Collection")

        col1, col2 = st.columns(2)
        dimension = col1.number_input("Dimension", value=1536, min_value=1, max_value=4096, step=1)

        if col1.button("Create", use_container_width=True):
            try:
                store = get_store()
                store.create_collection(get_collection_name(), dimension=int(dimension))
                st.success(f"Collection '{get_collection_name()}' created")
            except Exception as e:
                st.error(str(e))

        if col2.button("Drop", use_container_width=True):
            try:
                store = get_store()
                store.drop_collection(get_collection_name())
                st.success(f"Collection '{get_collection_name()}' dropped")
            except Exception as e:
                st.error(str(e))

        if st.button("Count", use_container_width=True):
            try:
                store = get_store()
                if store.has_collection(get_collection_name()):
                    cnt = store.count(get_collection_name())
                    st.info(f"Collection '{get_collection_name()}': {cnt} chunks")
                else:
                    st.warning(f"Collection '{get_collection_name()}' does not exist")
            except Exception as e:
                st.error(str(e))
