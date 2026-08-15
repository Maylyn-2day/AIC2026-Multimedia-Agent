"""
POST /v1/query/image-example — Image-to-Image Search.

Click-to-refine similarity search using SigLIP 2 re-query.
Latency budget: <100ms.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from backend.schemas.common import BaseResponse
from backend.schemas.image import ImageQueryRequest

router = APIRouter()


@router.post(
    "/query/image-example",
    response_model=BaseResponse,
    summary="Image-to-Image Search",
    description="Find visually similar keyframes by clicking a reference frame.",
)
async def image_query(request: ImageQueryRequest) -> BaseResponse:
    """
    Execute image-to-image similarity search.

    Phase 1: Returns placeholder response.
    Phase 3+: Will encode the reference image with SigLIP 2 and
    search Qdrant for nearest neighbors.
    """
    start = time.perf_counter()
    elapsed = time.perf_counter() - start

    return BaseResponse(
        status="success",
        data={
            "results": [],
            "total": 0,
            "source": {
                "video_id": request.video_id,
                "frame_id": request.frame_id,
            },
        },
        message="Image-to-image search placeholder (Phase 1)",
        execution_time=f"{elapsed:.3f}s",
    )
