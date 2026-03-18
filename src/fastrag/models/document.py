from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    source: str = ""
    file_type: str = ""
    page_number: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: Metadata = Field(default_factory=Metadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    document_id: str
    index: int = 0
    metadata: Metadata = Field(default_factory=Metadata)
    embedding: list[float] | None = None
