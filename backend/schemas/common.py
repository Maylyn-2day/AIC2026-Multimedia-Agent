"""
Common Pydantic Models.

Defines the standardized API response envelope used by all endpoints,
matching the contract in ``docs/api_contract.md`` Section 1.3.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Structured error detail for 4xx/5xx responses."""

    code: int = Field(..., description="HTTP status code")
    error: str = Field(..., description="Error type name")
    detail: str = Field(..., description="Human-readable error description")


class BaseResponse(BaseModel):
    """
    Standardized API response envelope.

    Every endpoint wraps its payload in this structure to ensure
    consistent client-side parsing across the Streamlit frontend.

    Matches the contract::

        {
            "status": "success | error",
            "data": {},
            "message": "...",
            "execution_time": "0.35s",
            "agent_reasoning": "..."
        }
    """

    status: str = Field(
        default="success",
        description="Response status: 'success' or 'error'",
    )
    data: Any = Field(
        default=None,
        description="Response payload (endpoint-specific)",
    )
    message: str = Field(
        default="",
        description="Human-readable status or error message",
    )
    execution_time: str = Field(
        default="0.00s",
        description="Server-side processing time",
    )
    agent_reasoning: str | None = Field(
        default=None,
        description="System 2 Chain-of-Thought trace, if applicable",
    )


class KeyframeResult(BaseModel):
    """A single keyframe search result returned by query endpoints."""

    video_id: str = Field(..., description="Video identifier (e.g. 'L01_V001')")
    frame_id: int = Field(..., description="Frame index within the video")
    score: float = Field(..., description="Relevance score (RRF or model confidence)")
    thumbnail_url: str | None = Field(default=None, description="URL or path to keyframe thumbnail")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata (OCR, objects, etc.)")


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., description="Total number of results")
    returned: int = Field(..., description="Number of results in this page")
    offset: int = Field(default=0, description="Offset from start")
