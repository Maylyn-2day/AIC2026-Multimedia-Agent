"""Strict schemas for local validation and packaging of submissions."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

StrictFrameId = Annotated[int, Field(strict=True, ge=0)]


class TaskType(StrEnum):
    """Canonical AIC 2026 task types; Q&A is represented as VQA."""

    KIS = "KIS"
    VQA = "VQA"
    TRAKE = "TRAKE"


class SubmissionItem(BaseModel):
    """One caller-ranked video frame, optionally with a VQA answer."""

    model_config = ConfigDict(extra="forbid")
    video_id: str = Field(min_length=1, max_length=200)
    frame_id: StrictFrameId
    answer: str | None = Field(default=None, max_length=2000)

    @field_validator("video_id")
    @classmethod
    def strip_video_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("video_id must not be blank")
        return stripped

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class SubmissionPayload(BaseModel):
    """Submission candidate list containing at most 100 caller-ranked items."""

    model_config = ConfigDict(extra="forbid")
    task_type: TaskType
    question_id: str = Field(default="", max_length=200)
    results: list[SubmissionItem]

    @field_validator("question_id")
    @classmethod
    def strip_question_id(cls, value: str) -> str:
        return value.strip()


class PackagedSubmission(BaseModel):
    """Locally validated package; it has not been sent to the BTC server."""

    model_config = ConfigDict(extra="forbid")
    task_type: TaskType
    question_id: str
    results: list[SubmissionItem]
    result_count: int
    validated: bool = True
    submitted: bool = False


class SubmissionResult(PackagedSubmission):
    """Backward-compatible name for the validate-only result schema."""
