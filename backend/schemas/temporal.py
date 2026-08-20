"""
Temporal Alignment (TRAKE) Schemas.

Request and response models for ``POST /v1/temporal/align`` — the
multi-stage temporal alignment engine that decomposes queries into
Q_past, Q_current, Q_future and enforces temporal ordering constraints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TemporalAlignRequest(BaseModel):
    """
    Request body for ``POST /v1/temporal/align``.

    The raw query is decomposed by the System 2 Agent into a temporal
    triple. If ``auto_decompose`` is True, the backend will call the
    Agent to perform the decomposition automatically.
    """

    raw_query: str = Field(
        ...,
        description="Full natural language TRAKE query",
    )
    q_past: str | None = Field(
        default=None,
        description="Decomposed past-event query (Q_previous)",
    )
    q_current: str | None = Field(
        default=None,
        description="Decomposed current-event query (Q_current)",
    )
    q_future: str | None = Field(
        default=None,
        description="Decomposed future-event query (Q_next)",
    )
    auto_decompose: bool = Field(
        default=True,
        description="Auto-decompose raw_query into temporal triple via Agent",
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of temporal sequences to return",
    )


class TemporalFrame(BaseModel):
    """A single frame within a temporal sequence."""

    video_id: str
    frame_id: int
    score: float
    phase: str = Field(..., description="Temporal phase: 'past', 'current', or 'future'")


class TemporalSequence(BaseModel):
    """
    A complete temporal sequence satisfying the ordering constraint:
    ``index(r_past) < index(r_current) < index(r_future)``
    within the same video.
    """

    video_id: str = Field(..., description="Video containing the full temporal sequence")
    final_score: float = Field(
        ...,
        description="S_final = w_c·Score(r_c) + w_p·Score(r_p) + w_n·Score(r_n)",
    )
    frames: list[TemporalFrame] = Field(
        ...,
        description="Ordered frames [past, current, future]",
    )


class TemporalAlignData(BaseModel):
    """Structured data payload for temporal alignment response."""

    sequences: list[TemporalSequence] = Field(default_factory=list)
    decomposition: dict[str, str] = Field(
        default_factory=dict,
        description="Agent's temporal decomposition: {past, current, future}",
    )
