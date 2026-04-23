from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    image_id: str
    image_path: str
    source_url: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    explanation: str
    labels: list[str] = Field(default_factory=list)
    dominant_colors: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    took_ms: float
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    indexed_images: int
    embedder_backend: str
