"""
POST /v1/submission/submit — Competition Submission.

Packages and validates results for the AIC 2026 scoring server.
Enforces the BTC-mandated format.  Latency budget: <100ms.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from backend.schemas.common import BaseResponse
from backend.schemas.submission import SubmissionPayload, SubmissionResult

router = APIRouter()


@router.post(
    "/submission/submit",
    response_model=BaseResponse,
    summary="Submit Competition Results",
    description="Package and submit results to AIC 2026 scoring server.",
)
async def submit_results(payload: SubmissionPayload) -> BaseResponse:
    """
    Validate and submit competition results.

    Phase 1: Validates format and returns success without actually
    submitting to the BTC server.
    Phase 4+: Will POST to the live competition scoring endpoint.
    """
    start = time.perf_counter()

    result = SubmissionResult(
        submitted=True,
        task_type=payload.task_type.value,
        result_count=len(payload.results),
        question_id=payload.question_id,
    )

    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data=result.model_dump(),
        message=f"Submission validated: {payload.task_type.value} with {len(payload.results)} results",
        execution_time=f"{elapsed:.3f}s",
    )
