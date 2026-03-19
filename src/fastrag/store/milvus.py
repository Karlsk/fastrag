from __future__ import annotations

import logging
import time
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
            raise ImportError(
                "pymilvus is required. "
                "Install with: pip install fastrag[milvus]"
            )

        s = settings or get_settings()
        logger.info("Connecting to Milvus: uri=%s", s.MILVUS_URI)
        try:
            self._client = MilvusClient(
                uri=s.MILVUS_URI,
                token=s.MILVUS_TOKEN or None,
            )
        except Exception:
            logger.exception("Failed to connect to Milvus: uri=%s", s.MILVUS_URI)
            raise
        logger.info("MilvusVectorStore ready: uri=%s", s.MILVUS_URI)

    def _ensure_loaded(self, collection: str) -> None:
        """Load a collection into memory if not already loaded."""
        load_state = self._client.get_load_state(collection_name=collection)
        # load_state can be a dict or an enum depending on pymilvus version
        state = load_state.get("state", load_state) if isinstance(load_state, dict) else load_state
        state_str = str(state)
        if "Loaded" not in state_str:
            logger.info("Loading collection '%s' (state=%s)...", collection, state_str)
            self._client.load_collection(collection_name=collection)
            logger.info("Collection '%s' loaded", collection)

    def create_collection(
        self, name: str, dimension: int, **kwargs: Any,
    ) -> None:
        from pymilvus import CollectionSchema, DataType, FieldSchema

        if self._client.has_collection(name):
            logger.info("Collection '%s' already exists, ensuring loaded", name)
            self._ensure_loaded(name)
            return

        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR,
                is_primary=True, max_length=64,
            ),
            FieldSchema(
                name="content", dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="document_id", dtype=DataType.VARCHAR,
                max_length=64,
            ),
            FieldSchema(
                name="metadata_json", dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
            ),
        ]
        schema = CollectionSchema(
            fields=fields,
            description=kwargs.get("description", ""),
        )
        self._client.create_collection(
            collection_name=name, schema=schema,
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 256},
        )
        self._client.create_index(
            collection_name=name, index_params=index_params,
        )
        self._client.load_collection(collection_name=name)
        logger.info(
            "Created and loaded collection '%s': dim=%d", name, dimension,
        )

    def drop_collection(self, name: str) -> None:
        self._client.drop_collection(collection_name=name)
        logger.info("Dropped collection '%s'", name)

    def has_collection(self, name: str) -> bool:
        return self._client.has_collection(collection_name=name)

    def insert(
        self, collection: str, chunks: list[Chunk],
    ) -> list[str]:
        if not chunks:
            return []

        data = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk {chunk.id} has no embedding."
                )
            data.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "metadata_json": chunk.metadata.model_dump_json(),
                    "embedding": chunk.embedding,
                }
            )

        logger.info(
            "Inserting %d chunks into '%s'...", len(chunks), collection,
        )
        t0 = time.monotonic()
        self._client.insert(
            collection_name=collection, data=data,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "Inserted %d chunks into '%s', elapsed=%.3fs",
            len(chunks), collection, elapsed,
        )
        return [chunk.id for chunk in chunks]

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        self._ensure_loaded(collection)

        filter_expr = ""
        if filters:
            parts = []
            for k, v in filters.items():
                if isinstance(v, str):
                    parts.append(f'{k} == "{v}"')
                else:
                    parts.append(f"{k} == {v}")
            filter_expr = " and ".join(parts)

        t0 = time.monotonic()
        results = self._client.search(
            collection_name=collection,
            data=[query_vector],
            limit=top_k,
            output_fields=[
                "id", "content",
                "document_id", "metadata_json",
            ],
            filter=filter_expr or None,
            anns_field="embedding",
        )

        scored: list[ScoredDocument] = []
        if results:
            for hit in results[0]:
                entity = hit.get("entity", {})
                meta_json = entity.get("metadata_json", "{}")
                metadata = Metadata.model_validate_json(meta_json)
                chunk = Chunk(
                    id=entity.get("id", hit.get("id", "")),
                    content=entity.get("content", ""),
                    document_id=entity.get("document_id", ""),
                    metadata=metadata,
                )
                dist = hit.get("distance", 0.0)
                scored.append(ScoredDocument(
                    chunk=chunk, score=dist,
                    vector_score=dist,
                ))

        elapsed = time.monotonic() - t0
        logger.info(
            "Search '%s': top_k=%d, results=%d, elapsed=%.3fs",
            collection, top_k, len(scored), elapsed,
        )
        return scored

    def delete(self, collection: str, ids: list[str]) -> None:
        if ids:
            self._client.delete(
                collection_name=collection, ids=ids,
            )
            logger.info(
                "Deleted %d from '%s'", len(ids), collection,
            )

    def count(self, collection: str) -> int:
        stats = self._client.get_collection_stats(
            collection_name=collection,
        )
        cnt = stats.get("row_count", 0)
        logger.debug("Collection '%s' count: %d", collection, cnt)
        return cnt

    def list_chunks(self, collection: str, limit: int = 10000) -> list[Chunk]:
        """Fetch all chunks (without embeddings) from a Milvus collection."""
        self._ensure_loaded(collection)
        t0 = time.monotonic()
        rows = self._client.query(
            collection_name=collection,
            filter="",
            output_fields=["id", "content", "document_id", "metadata_json"],
            limit=limit,
        )
        chunks: list[Chunk] = []
        for row in rows:
            metadata = Metadata.model_validate_json(
                row.get("metadata_json", "{}"),
            )
            chunks.append(Chunk(
                id=row.get("id", ""),
                content=row.get("content", ""),
                document_id=row.get("document_id", ""),
                metadata=metadata,
            ))
        elapsed = time.monotonic() - t0
        logger.info(
            "Listed %d chunks from '%s', elapsed=%.3fs",
            len(chunks), collection, elapsed,
        )
        return chunks
