"""Pure local validation and packaging for AIC submissions."""

from __future__ import annotations

from copy import deepcopy

from backend.core.exceptions import SubmissionFormatError
from backend.schemas.submission import PackagedSubmission, SubmissionPayload, TaskType


def validate_and_package_submission(payload: SubmissionPayload) -> PackagedSubmission:
    """Validate task rules without mutation, sorting, or network submission."""
    copied = payload.model_copy(deep=True)
    if not 1 <= len(copied.results) <= 100:
        raise SubmissionFormatError("results must contain between 1 and 100 items")
    identities = [(item.video_id, item.frame_id) for item in copied.results]
    if copied.task_type in {TaskType.KIS, TaskType.VQA} and len(identities) != len(set(identities)):
        raise SubmissionFormatError("duplicate (video_id, frame_id) is not allowed")
    if copied.task_type is TaskType.VQA:
        if any(not item.answer for item in copied.results):
            raise SubmissionFormatError("every VQA result requires a non-blank answer")
    elif any(item.answer is not None for item in copied.results):
        raise SubmissionFormatError(f"{copied.task_type.value} results must not contain an answer")
    if copied.task_type is TaskType.TRAKE:
        if len(copied.results) < 2:
            raise SubmissionFormatError("TRAKE requires at least 2 frames")
        if len({item.video_id for item in copied.results}) != 1:
            raise SubmissionFormatError("all TRAKE frames must belong to the same video_id")
        frame_ids = [item.frame_id for item in copied.results]
        if any(current >= following for current, following in zip(frame_ids, frame_ids[1:], strict=False)):
            raise SubmissionFormatError("TRAKE frame_id values must be strictly increasing")
    return PackagedSubmission(
        task_type=copied.task_type,
        question_id=copied.question_id,
        results=deepcopy(copied.results),
        result_count=len(copied.results),
        validated=True,
        submitted=False,
    )
