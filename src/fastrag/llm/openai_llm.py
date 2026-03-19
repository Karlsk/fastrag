from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Generator

from openai import AsyncOpenAI, OpenAI

from fastrag.config import FastRAGSettings, get_settings
from fastrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAIChat(BaseLLM):
    """OpenAI-compatible chat LLM client."""

    def __init__(
        self,
        settings: FastRAGSettings | None = None,
        model: str | None = None,
    ):
        s = settings or get_settings()
        self._model = model or s.LLM_MODEL
        self._client = AsyncOpenAI(
            api_key=s.OPENAI_API_KEY,
            base_url=s.OPENAI_BASE_URL,
            timeout=s.LLM_TIMEOUT,
            max_retries=s.LLM_MAX_RETRIES,
        )
        self._sync_client = OpenAI(
            api_key=s.OPENAI_API_KEY,
            base_url=s.OPENAI_BASE_URL,
            timeout=s.LLM_TIMEOUT,
            max_retries=s.LLM_MAX_RETRIES,
        )
        logger.debug(
            "OpenAIChat init: model=%s, base_url=%s",
            self._model, s.OPENAI_BASE_URL,
        )

    async def chat(
        self, messages: list[dict[str, str]], **kwargs,
    ) -> str:
        t0 = time.monotonic()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False,
                **kwargs,
            )
            content = resp.choices[0].message.content or ""
            elapsed = time.monotonic() - t0
            logger.info(
                "LLM chat: model=%s, len=%d, elapsed=%.3fs",
                self._model, len(content), elapsed,
            )
            return content
        except Exception:
            elapsed = time.monotonic() - t0
            logger.error(
                "LLM chat failed: model=%s, elapsed=%.3fs",
                self._model, elapsed, exc_info=True,
            )
            raise

    async def stream_chat(
        self, messages: list[dict[str, str]], **kwargs,
    ) -> AsyncGenerator[str, None]:
        t0 = time.monotonic()
        total_len = 0
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    total_len += len(delta.content)
                    yield delta.content
            elapsed = time.monotonic() - t0
            logger.info(
                "LLM stream: model=%s, len=%d, elapsed=%.3fs",
                self._model, total_len, elapsed,
            )
        except Exception:
            elapsed = time.monotonic() - t0
            logger.error(
                "LLM stream failed: model=%s, elapsed=%.3fs",
                self._model, elapsed, exc_info=True,
            )
            raise

    # --- Sync methods for Streamlit / non-async contexts ---

    def sync_chat(
        self, messages: list[dict[str, str]], **kwargs,
    ) -> str:
        """Synchronous version of ``chat()``."""
        t0 = time.monotonic()
        try:
            resp = self._sync_client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False,
                **kwargs,
            )
            content = resp.choices[0].message.content or ""
            elapsed = time.monotonic() - t0
            logger.info(
                "LLM sync_chat: model=%s, len=%d, elapsed=%.3fs",
                self._model, len(content), elapsed,
            )
            return content
        except Exception:
            elapsed = time.monotonic() - t0
            logger.error(
                "LLM sync_chat failed: model=%s, elapsed=%.3fs",
                self._model, elapsed, exc_info=True,
            )
            raise

    def sync_stream_chat(
        self, messages: list[dict[str, str]], **kwargs,
    ) -> Generator[str, None, None]:
        """Synchronous streaming — yields tokens one by one.

        Suitable for ``st.write_stream()`` in Streamlit.
        Handles thinking/reasoning models (e.g. Qwen3.5) that send
        ``reasoning_content`` before the actual ``content``.
        """
        t0 = time.monotonic()
        total_len = 0
        thinking = False
        try:
            stream = self._sync_client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                # Detect thinking/reasoning phase (Qwen3.5, DeepSeek, etc.)
                rc = getattr(delta, "reasoning_content", None)
                if rc and not delta.content:
                    if not thinking:
                        thinking = True
                        logger.debug("LLM thinking phase started")
                    continue
                if delta.content:
                    if thinking:
                        thinking = False
                        logger.debug("LLM thinking phase ended, content starting")
                    total_len += len(delta.content)
                    yield delta.content
            elapsed = time.monotonic() - t0
            logger.info(
                "LLM sync_stream: model=%s, len=%d, elapsed=%.3fs",
                self._model, total_len, elapsed,
            )
        except Exception:
            elapsed = time.monotonic() - t0
            logger.error(
                "LLM sync_stream failed: model=%s, elapsed=%.3fs",
                self._model, elapsed, exc_info=True,
            )
            raise
