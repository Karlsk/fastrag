from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastrag.models.pipeline import PipelineDSL
from fastrag.pipeline.base import ComponentBase
from fastrag.pipeline.context import PipelineContext
from fastrag.pipeline.registry import get_component_class, get_param_class

logger = logging.getLogger(__name__)


class Graph:
    """Graph-based pipeline execution engine.

    Parses a DSL JSON definition, instantiates components, and executes them
    in topological order following the path. Supports conditional branching
    (Categorize/Switch) that dynamically modifies the execution path.

    Similar to RAGFlow's Graph/Canvas classes in agent/canvas.py.
    """

    def __init__(self, dsl: PipelineDSL | dict | str) -> None:
        if isinstance(dsl, str):
            dsl = PipelineDSL.model_validate(json.loads(dsl))
        elif isinstance(dsl, dict):
            dsl = PipelineDSL.model_validate(dsl)

        self._dsl = dsl
        self._context = PipelineContext(globals=dict(dsl.globals))
        self._components: dict[str, dict[str, Any]] = {}
        self._component_objs: dict[str, ComponentBase] = {}
        self._path: list[str] = list(dsl.path)

    @property
    def context(self) -> PipelineContext:
        return self._context

    def load(self) -> None:
        """Parse DSL and instantiate all components."""
        for cpn_id, node_def in self._dsl.components.items():
            comp_name = node_def.obj.component_name
            params_dict = node_def.obj.params

            param_cls = get_param_class(comp_name)
            param = param_cls()
            param.update(params_dict)
            param.check()

            comp_cls = get_component_class(comp_name)
            comp_obj = comp_cls(context=self._context, component_id=cpn_id, param=param)

            self._components[cpn_id] = {
                "def": node_def,
                "downstream": node_def.downstream,
                "upstream": node_def.upstream,
            }
            self._component_objs[cpn_id] = comp_obj

    def get_component(self, cpn_id: str) -> dict[str, Any]:
        return self._components[cpn_id]

    def get_component_obj(self, cpn_id: str) -> ComponentBase:
        return self._component_objs[cpn_id]

    async def run(self, query: str = "", **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Execute the pipeline and yield events.

        Events:
        - {"event": "workflow_started", "data": {...}}
        - {"event": "node_started", "data": {"node_id": ...}}
        - {"event": "node_finished", "data": {"node_id": ..., "outputs": ..., "elapsed": ...}}
        - {"event": "message", "data": {"content": ...}}
        - {"event": "workflow_finished", "data": {"elapsed": ...}}
        """
        self.load()

        if query:
            self._context.set_global("sys.query", query)
        for k, v in kwargs.items():
            self._context.set_global(k, v)

        for cpn_obj in self._component_objs.values():
            cpn_obj.reset()
        self._context.reset_all_components()

        workflow_start = time.monotonic()
        yield {"event": "workflow_started", "data": {"query": query, "globals": dict(self._context.globals)}}

        idx = 0
        while idx < len(self._path):
            cpn_id = self._path[idx]

            if cpn_id not in self._component_objs:
                logger.warning("Component %s not found, skipping", cpn_id)
                idx += 1
                continue

            cpn_obj = self._component_objs[cpn_id]
            cpn_def = self._components[cpn_id]

            yield {"event": "node_started", "data": {"node_id": cpn_id, "component": cpn_obj.component_name}}

            node_start = time.monotonic()
            try:
                await cpn_obj.invoke()
            except Exception as e:
                cpn_obj._error = str(e)
                logger.error("Component %s failed: %s", cpn_id, e)

            elapsed = time.monotonic() - node_start

            yield {
                "event": "node_finished",
                "data": {
                    "node_id": cpn_id,
                    "component": cpn_obj.component_name,
                    "outputs": cpn_obj.output(),
                    "error": cpn_obj.error(),
                    "elapsed": round(elapsed, 3),
                },
            }

            # Determine next step
            if cpn_obj.component_name.lower() in ("categorize", "switch"):
                next_ids = cpn_obj.output("_next") or []
                self._path.extend(next_ids)
            elif cpn_obj.error():
                logger.warning("Component %s had error, following downstream anyway", cpn_id)
                self._path.extend(cpn_def["downstream"])
            else:
                self._path.extend(cpn_def["downstream"])

            # Emit message events for Message components
            if cpn_obj.component_name == "Message":
                content = cpn_obj.output("content")
                if content:
                    yield {"event": "message", "data": {"content": content, "node_id": cpn_id}}

            idx += 1

        total_elapsed = time.monotonic() - workflow_start
        yield {"event": "workflow_finished", "data": {"elapsed": round(total_elapsed, 3)}}

    async def run_to_completion(self, query: str = "", **kwargs: Any) -> dict[str, Any]:
        """Run the pipeline and return the final message content."""
        last_message = ""
        events: list[dict[str, Any]] = []
        async for event in self.run(query, **kwargs):
            events.append(event)
            if event["event"] == "message":
                last_message = event["data"]["content"]

        return {"message": last_message, "events": events}
