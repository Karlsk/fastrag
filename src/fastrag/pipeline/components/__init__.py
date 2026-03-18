"""Built-in pipeline components. Auto-discovered by the registry."""

# Import all components so they self-register via decorators
from fastrag.pipeline.components.begin import BeginComponent
from fastrag.pipeline.components.categorize import CategorizeComponent
from fastrag.pipeline.components.llm_component import LLMComponent
from fastrag.pipeline.components.message import MessageComponent
from fastrag.pipeline.components.retrieval import RetrievalComponent
from fastrag.pipeline.components.switch import SwitchComponent

__all__ = [
    "BeginComponent",
    "LLMComponent",
    "RetrievalComponent",
    "MessageComponent",
    "CategorizeComponent",
    "SwitchComponent",
]
