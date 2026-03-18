from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastrag.pipeline.base import ComponentBase, ComponentParam

logger = logging.getLogger(__name__)

_component_registry: dict[str, type[ComponentBase]] = {}
_param_registry: dict[str, type[ComponentParam]] = {}


def register_component(name: str):
    """Class decorator to register a component.

    Usage:
        @register_component("LLM")
        class LLMComponent(ComponentBase):
            component_name = "LLM"
            ...
    """

    def decorator(cls):
        _component_registry[name] = cls
        return cls

    return decorator


def register_param(name: str):
    """Class decorator to register a component param class."""

    def decorator(cls):
        _param_registry[name] = cls
        return cls

    return decorator


def get_component_class(name: str) -> type[ComponentBase]:
    """Look up a component class by name."""
    if name not in _component_registry:
        _auto_discover()
    if name not in _component_registry:
        raise KeyError(f"Unknown component: {name}. Available: {sorted(_component_registry.keys())}")
    return _component_registry[name]


def get_param_class(name: str) -> type[ComponentParam]:
    """Look up a component param class by name."""
    if name not in _param_registry:
        _auto_discover()
    if name not in _param_registry:
        from fastrag.pipeline.base import ComponentParam

        return ComponentParam  # type: ignore[return-value]
    return _param_registry[name]


def list_components() -> list[str]:
    _auto_discover()
    return sorted(_component_registry.keys())


_discovered = False


def _auto_discover() -> None:
    """Auto-discover and import all component modules."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    components_dir = os.path.join(os.path.dirname(__file__), "components")
    if not os.path.isdir(components_dir):
        return

    for filename in os.listdir(components_dir):
        if filename.startswith("_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        try:
            importlib.import_module(f"fastrag.pipeline.components.{module_name}")
        except Exception as e:
            logger.warning("Failed to import component module %s: %s", module_name, e)
