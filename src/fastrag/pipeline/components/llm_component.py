from __future__ import annotations

import logging
from typing import Any

from fastrag.config import get_settings
from fastrag.llm.openai_llm import OpenAIChat
from fastrag.pipeline.base import ComponentBase, ComponentParam
from fastrag.pipeline.registry import register_component, register_param

logger = logging.getLogger(__name__)


class LLMParam(ComponentParam):
    def __init__(self) -> None:
        super().__init__()
        self.llm_id: str = ""
        self.sys_prompt: str = ""
        self.prompt_template: str = "{sys.query}"
        self.max_tokens: int = 0
        self.temperature: float = 0.7

    def check(self) -> None:
        if not self.prompt_template:
            raise ValueError("LLM component requires a prompt_template")


register_param("LLM")(LLMParam)


@register_component("LLM")
class LLMComponent(ComponentBase):
    """LLM chat component. Sends a prompt (with variable resolution) to an LLM."""

    component_name = "LLM"

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        param: LLMParam = self._param  # type: ignore[assignment]

        prompt = self._resolve_template(param.prompt_template)

        messages: list[dict[str, str]] = []
        if param.sys_prompt:
            sys_prompt = self._resolve_template(param.sys_prompt)
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        settings = get_settings()
        model = param.llm_id or settings.LLM_MODEL
        llm = OpenAIChat(settings=settings, model=model)

        llm_kwargs: dict[str, Any] = {}
        if param.max_tokens > 0:
            llm_kwargs["max_tokens"] = param.max_tokens
        if param.temperature >= 0:
            llm_kwargs["temperature"] = param.temperature

        try:
            content = await llm.chat(messages, **llm_kwargs)
        except Exception as e:
            self._error = str(e)
            content = ""
            logger.error("LLM invocation failed for %s: %s", self._id, e)

        self.set_output("content", content)
        self._context.set_component_output(self._id, self._outputs)
        return self._outputs
