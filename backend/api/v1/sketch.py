"""
POST /v1/query/sketch — Sketch-to-Image Search.

Receives a Base64-encoded sketch from the Streamlit canvas, converts it to
an embedding vector via ``SketchService``, and performs Qdrant similarity
search (or returns deterministic mock results when Qdrant is unavailable).

Latency budget: <300ms on GPU; <2s in mock/fallback mode.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Header

from backend.core.config import Settings, get_settings
from backend.core.logging import setup_logger
from backend.schemas.common import BaseResponse
from backend.schemas.image import SketchQueryRequest
from backend.services.sketch_service import SketchService

logger = setup_logger("sketch_route")
router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level service singleton (overridable in tests via dependency_overrides)
# ---------------------------------------------------------------------------

_sketch_service: SketchService = SketchService(
    encoder=None,         # M4's SigLIP 2 encoder injected once it ships
    vector_dim=get_settings().qdrant_vector_size,
    apply_edge_detection=True,
    qdrant_client=None,   # Phase 2: inject real Qdrant client
    collection_name=get_settings().qdrant_collection_name,
)


def get_sketch_service() -> SketchService:
    """Provide the shared SketchService singleton."""
    return _sketch_service


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post(
    "/query/sketch",
    response_model=BaseResponse,
    summary="Sketch-to-Image Search",
    description=(
        "Convert a hand-drawn canvas sketch to an embedding vector via "
        "SketchService (ControlNet edge + SigLIP 2 encoder or fallback), "
        "then search Qdrant for visually similar keyframes."
    ),
)
async def sketch_query(
    request: SketchQueryRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    sketch_service: SketchService = Depends(get_sketch_service),
) -> BaseResponse:
    """Execute sketch-to-image similarity search.

    Falls back to deterministic mock results when Qdrant is not connected,
    so the frontend remains functional during development.
    """
    start = time.perf_counter()

    session_tag = f"[{x_session_id}] " if x_session_id else ""
    logger.info(
        "%sSketch query: top_k=%d, has_prompt=%s",
        session_tag, request.top_k, bool(request.prompt),
    )

    results = sketch_service.search_by_sketch(
        sketch_base64=request.sketch_base64,
        prompt=request.prompt,
        top_k=request.top_k,
    )

    elapsed = time.perf_counter() - start
    mode = "qdrant" if sketch_service.has_qdrant else "mock"
    logger.info(
        "%sSketch search complete: %d results [%s] %.3fs",
        session_tag, len(results), mode, elapsed,
    )

    return BaseResponse(
        status="success",
        data={"results": results, "total": len(results)},
        message=f"Sketch search returned {len(results)} results ({mode} mode)",
        execution_time=f"{elapsed:.3f}s",
    )
