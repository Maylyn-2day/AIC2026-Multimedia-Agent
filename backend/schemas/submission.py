"""
Submission Schemas.

Request and response models for ``POST /v1/submission/submit`` —
packages results for the AIC 2026 competition scoring server.
Must strictly follow the BTC-mandated format.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Competition task types as defined by AIC 2026 rules."""

    KIS = "KIS"
    VQA = "VQA"
    TRAKE = "TRAKE"


class SubmissionItem(BaseModel):
    """
    A single result entry in the submission payload.

    For KIS: ``video_id`` + ``frame_id`` required.
    For VQA: ``video_id`` + ``frame_id`` + ``answer`` required.
    For TRAKE: ``video_id`` + ``frame_id`` (multiple per sequence).
    """

    video_id: str = Field(..., description="Video identifier (e.g. 'L01_V001')")
    frame_id: int = Field(..., ge=0, description="Frame index within the video")
    answer: str | None = Field(
        default=None,
        description="Text answer for VQA tasks (e.g. 'màu đỏ')",
    )


class SubmissionPayload(BaseModel):
    """
    Request body for ``POST /v1/submission/submit``.

    Matches the BTC-mandated format from ``docs/api_contract.md`` Section 3.7.
    The system enforces the "Fill 100" strategy: always submit 100 ranked results.
    """

    task_type: TaskType = Field(..., description="Task type: KIS, VQA, or TRAKE")
    question_id: str = Field(
        default="",
        description="Competition question identifier",
    )
    results: list[SubmissionItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Ranked results (up to 100, sorted by score descending)",
    )


class SubmissionResult(BaseModel):
    """Response data after a successful submission."""

    submitted: bool = Field(..., description="Whether submission was accepted")
    task_type: str = Field(..., description="Task type that was submitted")
    result_count: int = Field(..., description="Number of results submitted")
    question_id: str = Field(default="", description="Question ID submitted for")
