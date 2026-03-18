from __future__ import annotations

import re
from typing import Any


class PipelineContext:
    """Variable system for pipeline execution.

    Manages global variables (sys.*) and component outputs (component_id@key).
    Supports template resolution with {variable} syntax.
    """

    VARIABLE_PATTERN = re.compile(r"\{([a-zA-Z0-9_:]+@[a-zA-Z0-9_.]+|sys\.[a-zA-Z0-9_.]+|env\.[a-zA-Z0-9_.]+)\}")

    def __init__(self, globals: dict[str, Any] | None = None):
        self._globals: dict[str, Any] = globals or {}
        self._component_outputs: dict[str, dict[str, Any]] = {}

    @property
    def globals(self) -> dict[str, Any]:
        return self._globals

    def set_global(self, key: str, value: Any) -> None:
        self._globals[key] = value

    def get_global(self, key: str, default: Any = None) -> Any:
        return self._globals.get(key, default)

    def set_component_output(self, component_id: str, outputs: dict[str, Any]) -> None:
        self._component_outputs[component_id] = outputs

    def get_component_output(self, component_id: str) -> dict[str, Any]:
        return self._component_outputs.get(component_id, {})

    def get_variable(self, expr: str) -> Any:
        """Resolve a variable expression.

        Supports:
        - sys.query -> globals["sys.query"]
        - env.XXX -> globals["env.XXX"]
        - component_id@key -> component outputs
        - component_id@key.nested.path -> nested access
        """
        if expr.startswith("sys.") or expr.startswith("env."):
            return self._globals.get(expr)

        if "@" not in expr:
            return self._globals.get(expr)

        cpn_id, var_path = expr.split("@", 1)
        outputs = self._component_outputs.get(cpn_id, {})

        parts = var_path.split(".")
        root_key = parts[0]
        value = outputs.get(root_key)

        for part in parts[1:]:
            if value is None:
                break
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, (list, tuple)):
                try:
                    value = value[int(part)]
                except (ValueError, IndexError):
                    value = None
            else:
                value = getattr(value, part, None)

        return value

    def set_variable(self, expr: str, value: Any) -> None:
        if expr.startswith("sys.") or expr.startswith("env."):
            self._globals[expr] = value
            return

        if "@" in expr:
            cpn_id, key = expr.split("@", 1)
            if cpn_id not in self._component_outputs:
                self._component_outputs[cpn_id] = {}
            self._component_outputs[cpn_id][key] = value

    def resolve_template(self, template: str) -> str:
        """Replace all {variable} references in a template string with actual values."""

        def _replacer(match: re.Match) -> str:
            expr = match.group(1)
            val = self.get_variable(expr)
            if val is None:
                return match.group(0)
            return str(val)

        return self.VARIABLE_PATTERN.sub(_replacer, template)

    def reset_component(self, component_id: str) -> None:
        self._component_outputs.pop(component_id, None)

    def reset_all_components(self) -> None:
        self._component_outputs.clear()
