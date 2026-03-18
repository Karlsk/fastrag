from __future__ import annotations

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from fastrag.config import FastRAGSettings, get_settings
from fastrag.llm.base import BaseLLM


class OpenAIChat(BaseLLM):
    """OpenAI-compatible chat LLM client."""

    def __init__(self, settings: FastRAGSettings | None = None, model: str | None = None):
        s = settings or get_settings()
        self._model = model or s.LLM_MODEL
        self._client = AsyncOpenAI(
            api_key=s.OPENAI_API_KEY,
            base_url=s.OPENAI_BASE_URL,
            timeout=s.LLM_TIMEOUT,
            max_retries=s.LLM_MAX_RETRIES,
        )

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
