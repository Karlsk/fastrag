from __future__ import annotations

import logging
from typing import Any

from fastrag.config import get_settings
from fastrag.llm.openai_embedding import OpenAIEmbedding
from fastrag.pipeline.base import ComponentBase, ComponentParam
from fastrag.pipeline.registry import register_component, register_param
from fastrag.retrieval.vector import VectorRetriever
from fastrag.store.milvus import MilvusVectorStore

logger = logging.getLogger(__name__)


class RetrievalParam(ComponentParam):
    def __init__(self) -> None:
        super().__init__()
        self.collection: str = ""
        self.top_k: int = 5

    def check(self) -> None:
        if not self.collection:
            raise ValueError("Retrieval component requires a collection name")


register_param("Retrieval")(RetrievalParam)


@register_component("Retrieval")
class RetrievalComponent(ComponentBase):
    """Retrieve relevant chunks from a vector store."""

    component_name = "Retrieval"

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        param: RetrievalParam = self._param  # type: ignore[assignment]

        query = self._context.get_global("sys.query", "")

        try:
            settings = get_settings()
            store = MilvusVectorStore(settings=settings)
            embedding = OpenAIEmbedding(settings=settings)
            retriever = VectorRetriever(store=store, embedding=embedding, collection=param.collection)

            results = await retriever.retrieve(query, top_k=param.top_k)
            chunks_text = "\n\n".join(r.chunk.content for r in results)
        except Exception as e:
            self._error = str(e)
            chunks_text = ""
            results = []
            logger.error("Retrieval failed for %s: %s", self._id, e)

        self.set_output("content", chunks_text)
        self.set_output("chunks", [r.chunk.model_dump() for r in results])
        self.set_output("count", len(results))
        self._context.set_component_output(self._id, self._outputs)
        return self._outputs
