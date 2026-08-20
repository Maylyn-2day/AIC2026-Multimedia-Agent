"""
Reranking / Early-Fusion Schemas.

Request and response models for ``POST /v1/rerank/early-fusion`` —
the Qwen2.5-VL visual verification and VQA answer extraction endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RerankCandidate(BaseModel):
    """A single candidate keyframe to be reranked by the VLM.

    The ``rrf_score`` field carries the fused ranking score computed by
    :func:`backend.services.fusion.weighted_rrf` through the API boundary.
    When present it is used as the base for rerank scoring; ``score`` is
    kept as the pre-fusion retrieval score for comparison and display.
    """

    video_id: str = Field(..., description="Video identifier")
    frame_id: int = Field(..., description="Frame index within the video")
    score: float = Field(..., description="Initial retrieval score (pre-rerank)")
    rrf_score: float | None = Field(
        default=None,
        description=(
            "RRF fusion score from weighted_rrf(). When provided, the reranker "
            "uses this as the base multiplied by grounding_confidence."
        ),
    )
    thumbnail_path: str | None = Field(default=None, description="Path to keyframe image")


class RerankRequest(BaseModel):
    """
    Request body for ``POST /v1/rerank/early-fusion``.

    Sends the top-N candidates from hybrid search for deep
    visual verification via Qwen2.5-VL.
    """

    query: str = Field(..., description="Original natural language query")
    candidates: list[RerankCandidate] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Candidate keyframes to rerank (max 50)",
    )
    task_type: str = Field(
        default="KIS",
        description="Task type: 'KIS', 'VQA', or 'TRAKE'",
    )
    extract_answer: bool = Field(
        default=False,
        description="Whether to extract a VQA text answer",
    )


class GroundingResult(BaseModel):
    """Bounding box grounding result from Grounding DINO / OWL-ViT."""

    label: str = Field(..., description="Detected object label")
    confidence: float = Field(..., description="Detection confidence score")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] normalized to [0, 1]",
    )


class RerankResultItem(BaseModel):
    """A single reranked result with optional VQA answer and grounding."""

    video_id: str
    frame_id: int
    original_score: float
    rerank_score: float
    vqa_answer: str | None = Field(default=None, description="VQA text answer if extracted")
    grounding: list[GroundingResult] = Field(default_factory=list)
    reasoning_trace: str | None = Field(
        default=None,
        description="System 2 Chain-of-Thought trace from VLM reasoning (Phase 3+)",
    )


class VQAResponse(BaseModel):
    """Structured data payload for rerank responses."""

    results: list[RerankResultItem] = Field(default_factory=list)
    vqa_answer: str | None = Field(default=None, description="Primary VQA answer string")
