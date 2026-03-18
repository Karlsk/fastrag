from __future__ import annotations

import json
import logging
from typing import Any

from fastrag.config import get_settings
from fastrag.llm.openai_llm import OpenAIChat
from fastrag.pipeline.base import ComponentBase, ComponentParam
from fastrag.pipeline.registry import register_component, register_param

logger = logging.getLogger(__name__)


class CategorizeParam(ComponentParam):
    def __init__(self) -> None:
        super().__init__()
        self.llm_id: str = ""
        self.categories: dict[str, dict[str, Any]] = {}
        # categories format:
        # {
        #   "category_name": {
        #       "description": "when to choose this",
        #       "examples": ["example1"],
        #       "to": ["next_component_id"]
        #   }
        # }

    def check(self) -> None:
        if not self.categories:
            raise ValueError("Categorize component requires at least one category")


register_param("Categorize")(CategorizeParam)


@register_component("Categorize")
class CategorizeComponent(ComponentBase):
    """Classify input into categories and route to different downstream components.

    Uses an LLM to determine which category the input belongs to,
    then sets _next to the appropriate downstream component(s).
    """

    component_name = "Categorize"

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        param: CategorizeParam = self._param  # type: ignore[assignment]

        query = self._context.get_global("sys.query", "")

        categories_desc = []
        for name, info in param.categories.items():
            desc = info.get("description", "")
            examples = info.get("examples", [])
            ex_str = f" (examples: {', '.join(examples)})" if examples else ""
            categories_desc.append(f"- {name}: {desc}{ex_str}")

        prompt = (
            f"Classify the following user query into exactly one of these categories:\n"
            f"{chr(10).join(categories_desc)}\n\n"
            f"User query: {query}\n\n"
            f'Respond with ONLY a JSON object: {{"category": "category_name"}}'
        )

        settings = get_settings()
        model = param.llm_id or settings.LLM_MODEL
        llm = OpenAIChat(settings=settings, model=model)

        try:
            resp = await llm.chat([{"role": "user", "content": prompt}], temperature=0)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(resp)
            category = result.get("category", "")
        except Exception as e:
            self._error = str(e)
            category = ""
            logger.error("Categorize failed for %s: %s", self._id, e)

        # Determine next component(s) based on category
        next_components: list[str] = []
        if category in param.categories:
            next_components = param.categories[category].get("to", [])

        self.set_output("category", category)
        self.set_output("_next", next_components)
        self._context.set_component_output(self._id, self._outputs)
        return self._outputs
