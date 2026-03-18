from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ComponentDef(BaseModel):
    component_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class NodeDef(BaseModel):
    obj: ComponentDef
    upstream: list[str] = Field(default_factory=list)
    downstream: list[str] = Field(default_factory=list)


class PipelineDSL(BaseModel):
    components: dict[str, NodeDef]
    globals: dict[str, Any] = Field(default_factory=dict)
    path: list[str] = Field(default_factory=lambda: ["begin"])
