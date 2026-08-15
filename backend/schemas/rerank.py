"""
Reranking / Early-Fusion Schemas.

Request and response models for ``POST /v1/rerank/early-fusion`` —
the Qwen2.5-VL visual verification and VQA answer extraction endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RerankCandidate(BaseModel):
    """A single candidate keyframe to be reranked by the VLM."""

    video_id: str = Field(..., description="Video identifier")
    frame_id: int = Field(..., description="Frame index within the video")
    score: float = Field(..., description="Initial retrieval score (pre-rerank)")
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
        description="Bounding box [x1, y1, x2, y2] normalized",
    )


class RerankResultItem(BaseModel):
    """A single reranked result with optional VQA answer and grounding."""

    video_id: str
    frame_id: int
    original_score: float
    rerank_score: float
    vqa_answer: str | None = Field(default=None, description="VQA text answer if extracted")
    grounding: list[GroundingResult] = Field(default_factory=list)


class VQAResponse(BaseModel):
    """Structured data payload for rerank responses."""

    results: list[RerankResultItem] = Field(default_factory=list)
    vqa_answer: str | None = Field(default=None, description="Primary VQA answer string")
