"""
POST /v1/db/query — Hybrid Search (Vector + BM25 + RRF).

Core search endpoint combining Qdrant dense retrieval and
Elasticsearch sparse retrieval with Reciprocal Rank Fusion.
Latency budget: <400ms.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter

from backend.schemas.common import BaseResponse
from backend.schemas.query import HybridQueryRequest

router = APIRouter()

_MOCK_FILE = Path(__file__).resolve().parents[3] / "data" / "mock" / "query_response.json"


@router.post(
    "/db/query",
    response_model=BaseResponse,
    summary="Hybrid Search",
    description="Vector + BM25 hybrid search with RRF fusion and metadata filters.",
)
async def hybrid_query(request: HybridQueryRequest) -> BaseResponse:
    """
    Execute a hybrid search query.

    Phase 1: Returns mock data.
    Phase 2+: Will execute real Qdrant + ES search with RRF fusion.
    """
    start = time.perf_counter()

    # Phase 1: Return mock data
    if _MOCK_FILE.exists():
        mock_data = json.loads(_MOCK_FILE.read_text(encoding="utf-8"))
        elapsed = time.perf_counter() - start
        return BaseResponse(
            status="success",
            data=mock_data.get("data", {}),
            message=f"Hybrid search completed for: '{request.raw_query}'",
            execution_time=f"{elapsed:.3f}s",
            agent_reasoning=mock_data.get("agent_reasoning"),
        )

    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data={"results": [], "total": 0},
        message="No search backend connected yet",
        execution_time=f"{elapsed:.3f}s",
    )
