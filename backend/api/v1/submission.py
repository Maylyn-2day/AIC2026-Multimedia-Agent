"""Validate and package submissions locally without network delivery."""

from __future__ import annotations

import time

from fastapi import APIRouter

from backend.schemas.common import BaseResponse
from backend.schemas.submission import SubmissionPayload
from backend.services.submission_service import validate_and_package_submission

router = APIRouter()


@router.post("/submission/submit", response_model=BaseResponse, summary="Validate Submission Locally")
async def submit_results(payload: SubmissionPayload) -> BaseResponse:
    """Validate and package results; no competition request is performed."""
    start = time.perf_counter()
    packaged = validate_and_package_submission(payload)
    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data=packaged.model_dump(mode="json"),
        message="Submission validated locally; not sent to competition server",
        execution_time=f"{elapsed:.3f}s",
        agent_reasoning=None,
    )
