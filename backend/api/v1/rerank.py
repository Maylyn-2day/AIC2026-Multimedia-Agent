"""
POST /v1/rerank/early-fusion — VLM Reranking & VQA.

Qwen2.5-VL visual verification and VQA answer extraction.
Latency budget: <600ms.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter

from backend.schemas.common import BaseResponse
from backend.schemas.rerank import RerankRequest

router = APIRouter()

_MOCK_FILE = Path(__file__).resolve().parents[3] / "data" / "mock" / "rerank_response.json"


@router.post(
    "/rerank/early-fusion",
    response_model=BaseResponse,
    summary="VLM Reranking & VQA",
    description="Rerank candidates via Qwen2.5-VL with optional VQA answer extraction.",
)
async def rerank_early_fusion(request: RerankRequest) -> BaseResponse:
    """
    Rerank candidate keyframes using visual language model.

    Phase 1: Returns mock data.
    Phase 3+: Will invoke Qwen2.5-VL for deep visual reasoning.
    """
    start = time.perf_counter()

    if _MOCK_FILE.exists():
        mock_data = json.loads(_MOCK_FILE.read_text(encoding="utf-8"))
        elapsed = time.perf_counter() - start
        return BaseResponse(
            status="success",
            data=mock_data.get("data", {}),
            message=f"Reranking completed for {len(request.candidates)} candidates",
            execution_time=f"{elapsed:.3f}s",
            agent_reasoning=mock_data.get("agent_reasoning"),
        )

    elapsed = time.perf_counter() - start
    return BaseResponse(
        status="success",
        data={"results": [], "vqa_answer": None},
        message="VLM reranker not connected yet",
        execution_time=f"{elapsed:.3f}s",
    )
