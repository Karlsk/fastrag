from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from fastrag.models.document import Chunk


class ScoredDocument(BaseModel):
    chunk: Chunk
    score: float = 0.0
    vector_score: float | None = None
    keyword_score: float | None = None


class SearchResult(BaseModel):
    query: str
    results: list[ScoredDocument] = Field(default_factory=list)
    total: int = 0


@dataclass
class MatchTextExpr:
    fields: list[str]
    text: str
    topn: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchDenseExpr:
    vector_column: str
    embedding: list[float]
    topn: int = 10
    distance_type: str = "cosine"


@dataclass
class FusionExpr:
    method: str = "weighted_sum"
    topn: int = 10
    weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
