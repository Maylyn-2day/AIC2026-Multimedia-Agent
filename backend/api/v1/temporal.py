"""
POST /v1/temporal/align — TRAKE Temporal Alignment.

Multi-stage temporal alignment engine that decomposes queries
into Q_past, Q_current, Q_future.  Latency budget: <200ms.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter

from backend.schemas.common import BaseResponse
from backend.schemas.temporal import TemporalAlignRequest

router = APIRouter()

_MOCK_FILE = Path(__file__).resolve().parents[3] / "data" / "mock" / "temporal_response.json"


@router.post(
    "/temporal/align",
    response_model=BaseResponse,
    summary="TRAKE Temporal Alignment",
    description="Decompose query into temporal phases and find ordered frame sequences.",
)
async def temporal_align(request: TemporalAlignRequest) -> BaseResponse:
    """
    Execute TRAKE temporal alignment.

    Phase 1: Returns mock data.
    Phase 3+: Will decompose query via Agent and run multi-stage
    temporal sequence matching.
    """
    start = time.perf_counter()

    if _MOCK_FILE.exists():
        mock_data = json.loads(_MOCK_FILE.read_text(encoding="utf-8"))
        elapsed = time.perf_counter() - start
        return BaseResponse(
            status="success",
            data=mock_data.get("data", {}),
            message=f"Temporal alignment for: '{request.raw_query}'",
            execution_time=f"{elapsed:.3f}s",
            agent_reasoning=mock_data.get("agent_reasoning"),
        )

    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data={"sequences": [], "decomposition": {}},
        message="Temporal engine not connected yet",
        execution_time=f"{elapsed:.3f}s",
    )
