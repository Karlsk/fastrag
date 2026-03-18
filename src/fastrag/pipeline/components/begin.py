from __future__ import annotations

from typing import Any

from fastrag.pipeline.base import ComponentBase, ComponentParam
from fastrag.pipeline.registry import register_component, register_param


class BeginParam(ComponentParam):
    def __init__(self) -> None:
        super().__init__()
        self.prologue: str = ""


register_param("Begin")(BeginParam)


@register_component("Begin")
class BeginComponent(ComponentBase):
    """Entry point of a pipeline. Receives initial query and sets sys.query."""

    component_name = "Begin"

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        query = self._context.get_global("sys.query", "")
        self.set_output("query", query)
        self.set_output("prologue", self._param.prologue if hasattr(self._param, "prologue") else "")
        self._context.set_component_output(self._id, self._outputs)
        return self._outputs
