"""
POST /v1/query/sketch — Sketch-to-Image Search.

Receives sketch from Canvas, processes via ControlNet, then
performs similarity search.  Latency budget: <300ms.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from backend.schemas.common import BaseResponse
from backend.schemas.image import SketchQueryRequest

router = APIRouter()


@router.post(
    "/query/sketch",
    response_model=BaseResponse,
    summary="Sketch-to-Image Search",
    description="Convert a hand-drawn sketch to feature vector and search.",
)
async def sketch_query(request: SketchQueryRequest) -> BaseResponse:
    """
    Execute sketch-to-image search.

    Phase 1: Returns placeholder response.
    Phase 3+: Will process sketch through ControlNet + SDXL-Turbo
    pipeline and search Qdrant.
    """
    start = time.perf_counter()
    elapsed = time.perf_counter() - start

    return BaseResponse(
        status="success",
        data={"results": [], "total": 0},
        message="Sketch search placeholder (Phase 1)",
        execution_time=f"{elapsed:.3f}s",
    )
