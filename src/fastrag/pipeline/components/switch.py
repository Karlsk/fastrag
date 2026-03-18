from __future__ import annotations

import logging
import operator
from typing import Any

from fastrag.pipeline.base import ComponentBase, ComponentParam
from fastrag.pipeline.registry import register_component, register_param

logger = logging.getLogger(__name__)

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "contains": lambda a, b: b in str(a),
    "not_contains": lambda a, b: b not in str(a),
}


class SwitchParam(ComponentParam):
    def __init__(self) -> None:
        super().__init__()
        self.conditions: list[dict[str, Any]] = []
        # conditions format:
        # [
        #   {
        #       "variable": "{component_id@key}",
        #       "operator": "==",
        #       "value": "some_value",
        #       "to": ["next_component_id"]
        #   },
        #   ...
        # ]
        self.default_to: list[str] = []

    def check(self) -> None:
        if not self.conditions and not self.default_to:
            raise ValueError("Switch component requires at least one condition or a default_to")


register_param("Switch")(SwitchParam)


@register_component("Switch")
class SwitchComponent(ComponentBase):
    """Conditional branching based on variable values.

    Evaluates conditions in order; the first match determines the next component(s).
    Falls back to default_to if no condition matches.
    """

    component_name = "Switch"

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        param: SwitchParam = self._param  # type: ignore[assignment]

        next_components: list[str] = []

        for cond in param.conditions:
            var_expr = cond.get("variable", "")
            resolved = self._resolve_template(var_expr) if var_expr.startswith("{") else var_expr
            op_name = cond.get("operator", "==")
            expected = cond.get("value")
            op_fn = _OPS.get(op_name, operator.eq)

            try:
                if op_fn(resolved, expected):
                    next_components = cond.get("to", [])
                    break
            except Exception as e:
                logger.warning("Switch condition evaluation failed: %s", e)

        if not next_components:
            next_components = param.default_to

        self.set_output("_next", next_components)
        self._context.set_component_output(self._id, self._outputs)
        return self._outputs
