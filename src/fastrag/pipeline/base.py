from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from fastrag.pipeline.context import PipelineContext


class ComponentParam(ABC):
    """Base class for component parameters.

    Subclasses define specific parameters and validation logic.
    """

    def __init__(self) -> None:
        self.description: str = ""

    def update(self, conf: dict[str, Any]) -> None:
        """Update parameters from a dict (DSL params)."""
        for k, v in conf.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def check(self) -> None:
        """Validate parameters. Override in subclasses."""
        pass

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class ComponentBase(ABC):
    """Abstract base class for pipeline components.

    Similar to RAGFlow's ComponentBase in agent/component/base.py.
    """

    component_name: ClassVar[str] = ""

    def __init__(self, context: PipelineContext, component_id: str, param: ComponentParam) -> None:
        self._context = context
        self._id = component_id
        self._param = param
        self._outputs: dict[str, Any] = {}
        self._error: str | None = None

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        """Execute this component. Must set outputs via set_output()."""
        ...

    def output(self, key: str | None = None) -> Any:
        """Get output value(s). If key is None, return all outputs."""
        if key is None:
            return self._outputs
        return self._outputs.get(key)

    def set_output(self, key: str, value: Any) -> None:
        self._outputs[key] = value

    def error(self) -> str | None:
        return self._error

    def reset(self) -> None:
        self._outputs = {}
        self._error = None

    def _resolve_template(self, template: str) -> str:
        """Resolve variable references in a template string."""
        return self._context.resolve_template(template)
