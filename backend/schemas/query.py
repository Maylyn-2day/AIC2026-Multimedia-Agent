"""
Hybrid Query Schemas.

Request and response models for ``POST /v1/db/query`` — the core
hybrid search endpoint combining Qdrant vector search and
Elasticsearch BM25 with RRF fusion.
"""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, Field, model_validator


class QueryFilters(BaseModel):
    """
    Structured metadata filters for hybrid search.

    Maps to the ``filters`` field in the API contract request schema.
    All fields are optional — omitting a filter disables it.
    """

    objects: list[str] | None = Field(
        default=None,
        description="Object labels to filter by (e.g. ['person', 'laptop'])",
    )
    ocr: str | None = Field(
        default=None,
        description="OCR text to match (e.g. 'HTV9')",
    )
    asr: str | None = Field(
        default=None,
        description="ASR transcript keyword to match",
    )
    timestamp_range: tuple[AwareDatetime, AwareDatetime] | None = Field(
        default=None,
        description=(
            "Inclusive ISO 8601 datetime range as [start, end] (e.g. ['2026-03-14T10:00:00Z', '2026-03-14T12:00:00Z'])"
        ),
    )
    video_ids: list[str] | None = Field(
        default=None,
        description="Restrict search to specific video IDs",
    )
    channel: str | None = Field(
        default=None,
        description="YouTube channel name filter",
    )

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> QueryFilters:
        """Reject timestamp ranges whose start is later than their end."""
        if self.timestamp_range is not None:
            start, end = self.timestamp_range
            if start > end:
                raise ValueError("timestamp_range start must not be later than end")
        return self


class HybridQueryRequest(BaseModel):
    """
    Request body for ``POST /v1/db/query``.

    Matches the contract schema from ``docs/api_contract.md`` Section 3.2.
    """

    raw_query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language search query",
    )
    filters: QueryFilters = Field(
        default_factory=QueryFilters,
        description="Optional metadata filters",
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of results to return",
    )
    use_hippo_rag: bool = Field(
        default=False,
        description="Enable HippoRAG-style retrieval augmentation",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversational context tracking",
    )


class QueryResultData(BaseModel):
    """Structured data payload for the query response."""

    results: list[dict] = Field(default_factory=list, description="List of keyframe results")
    total: int = Field(default=0, description="Total matching results")
    cluster_id: str | None = Field(default=None, description="SOM cluster identifier")
    som_coords: list[list[float]] | None = Field(default=None, description="SOM 2D coordinates for visualization")
