from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class FastRAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FASTRAG_", env_file=".env", extra="ignore")

    # LLM / Embedding (OpenAI-compatible)
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    RERANK_MODEL: str = ""

    # Milvus
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_TOKEN: str = ""

    # Timeouts
    LLM_TIMEOUT: int = 600
    LLM_MAX_RETRIES: int = 3

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = ""


def get_settings() -> FastRAGSettings:
    return FastRAGSettings()
