from __future__ import annotations

import logging
from typing import Any

from fastrag.config import FastRAGSettings, get_settings
from fastrag.models.document import Chunk, Metadata
from fastrag.models.search import ScoredDocument
from fastrag.store.base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation using pymilvus."""

    def __init__(self, settings: FastRAGSettings | None = None):
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("pymilvus is required. Install with: pip install fastrag[milvus]")

        s = settings or get_settings()
        self._client = MilvusClient(uri=s.MILVUS_URI, token=s.MILVUS_TOKEN or None)

    def create_collection(self, name: str, dimension: int, **kwargs: Any) -> None:
        from pymilvus import CollectionSchema, DataType, FieldSchema

        if self._client.has_collection(name):
            logger.info("Collection %s already exists, skipping creation", name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields=fields, description=kwargs.get("description", ""))
        self._client.create_collection(collection_name=name, schema=schema)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 256},
        )
        self._client.create_index(collection_name=name, index_params=index_params)

    def drop_collection(self, name: str) -> None:
        self._client.drop_collection(collection_name=name)

    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(collection_name=name)

    def insert(self, collection: str, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []

        data = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding. Embed before inserting.")
            data.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "metadata_json": chunk.metadata.model_dump_json(),
                    "embedding": chunk.embedding,
                }
            )

        self._client.insert(collection_name=collection, data=data)
        return [chunk.id for chunk in chunks]

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        filter_expr = ""
        if filters:
            parts = []
            for k, v in filters.items():
                if isinstance(v, str):
                    parts.append(f'{k} == "{v}"')
                else:
                    parts.append(f"{k} == {v}")
            filter_expr = " and ".join(parts)

        results = self._client.search(
            collection_name=collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["id", "content", "document_id", "metadata_json"],
            filter=filter_expr or None,
            anns_field="embedding",
        )

        scored: list[ScoredDocument] = []
        if results:
            for hit in results[0]:
                entity = hit.get("entity", {})
                metadata = Metadata.model_validate_json(entity.get("metadata_json", "{}"))
                chunk = Chunk(
                    id=entity.get("id", hit.get("id", "")),
                    content=entity.get("content", ""),
                    document_id=entity.get("document_id", ""),
                    metadata=metadata,
                )
                distance = hit.get("distance", 0.0)
                scored.append(ScoredDocument(chunk=chunk, score=distance, vector_score=distance))

        return scored

    def delete(self, collection: str, ids: list[str]) -> None:
        if ids:
            self._client.delete(collection_name=collection, ids=ids)

    def count(self, collection: str) -> int:
        stats = self._client.get_collection_stats(collection_name=collection)
        return stats.get("row_count", 0)
