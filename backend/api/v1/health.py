"""
GET /v1/health — System Heartbeat.

Checks connectivity to Qdrant, Elasticsearch, and reports
the loaded/unloaded status of AI models and VRAM usage.
Latency budget: <50ms.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter

from backend.core.config import get_settings
from backend.schemas.common import BaseResponse

router = APIRouter()

# Path to mock response for Phase 1
_MOCK_FILE = Path(__file__).resolve().parents[3] / "data" / "mock" / "health_response.json"


@router.get(
    "/health",
    response_model=BaseResponse,
    summary="System Heartbeat",
    description="Check Qdrant, Elasticsearch connectivity and model status.",
)
async def health_check() -> BaseResponse:
    """
    Return system health status.

    In Phase 1 (mock mode), returns pre-built mock data.
    In production, this will probe live DB connections and GPU state.
    """
    start = time.perf_counter()
    settings = get_settings()

    # Phase 1: Return mock data
    if _MOCK_FILE.exists():
        mock_data = json.loads(_MOCK_FILE.read_text(encoding="utf-8"))
        elapsed = time.perf_counter() - start
        return BaseResponse(
            status="success",
            data=mock_data.get("data", {}),
            message=f"All systems operational ({settings.app_env})",
            execution_time=f"{elapsed:.3f}s",
        )

    # Production: probe live services (Phase 2+)
    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data={
            "qdrant": "not_checked",
            "elasticsearch": "not_checked",
            "models": {},
        },
        message="Health check placeholder",
        execution_time=f"{elapsed:.3f}s",
    )
