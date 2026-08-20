"""
POST /v1/rerank/early-fusion — Grounding DINO + Qwen2.5-VL Reranking & VQA.

Three-stage pipeline (Phase 3):
    1. Grounding DINO bounding-box verification (Top-50 candidates).
    2. Optional Qwen2.5-VL deep reasoning         (Top-5 candidates).
    3. Optional VQA answer extraction              (Top-1 candidate).

Latency budget: <600ms (GPU); graceful degradation on CPU/mock mode.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException

from backend.core.config import Settings, get_settings
from backend.core.logging import setup_logger
from backend.schemas.common import BaseResponse
from backend.schemas.rerank import RerankRequest, RerankResultItem, VQAResponse
from backend.services.reranker import GroundingReranker
from backend.services.vlm_service import VLMService

logger = setup_logger("rerank_route")
router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level service singletons.
#
# In production these would be initialised in the FastAPI lifespan hook and
# stored in app.state.  For Phase 3 we keep them as module-level singletons
# so the route can import them directly without requiring app-state threading.
# ---------------------------------------------------------------------------

_settings: Settings = get_settings()

_reranker: GroundingReranker = GroundingReranker(
    model_id=_settings.grounding_dino_model_id,
    device="cuda",
    max_vram_gb=_settings.max_vram_gb,
    alpha=0.5,
)

_vlm: VLMService = VLMService(
    model_id=_settings.qwen_vl_model_id,
    device="cuda",
    use_4bit=True,
    max_vram_gb=_settings.max_vram_gb,
)


# ---------------------------------------------------------------------------
# FastAPI dependency providers (overridable in tests via app.dependency_overrides)
# ---------------------------------------------------------------------------


def get_reranker() -> GroundingReranker:
    """Provide the shared GroundingReranker singleton."""
    return _reranker


def get_vlm() -> VLMService:
    """Provide the shared VLMService singleton."""
    return _vlm


def get_keyframes_dir() -> Path:
    """Resolve the keyframes root directory from Settings."""
    return Path(get_settings().keyframes_dir)


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post(
    "/rerank/early-fusion",
    response_model=BaseResponse,
    summary="VLM Reranking & VQA",
    description=(
        "Three-stage cascading pipeline: Grounding DINO bbox verification "
        "(Top-50) → optional Qwen2.5-VL deep reasoning (Top-5) → optional "
        "VQA answer extraction (Top-1)."
    ),
)
async def rerank_early_fusion(
    request: RerankRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    reranker: GroundingReranker = Depends(get_reranker),
    vlm: VLMService = Depends(get_vlm),
    keyframes_dir: Path = Depends(get_keyframes_dir),
) -> BaseResponse:
    """Rerank candidates via Grounding DINO + Qwen2.5-VL.

    **Stage 1 — Grounding DINO (always runs):**
    Processes all ``request.candidates`` (max 50).  When the model is not
    loaded, returns the original RRF ordering unchanged (pass-through mode).

    **Stage 2 — Qwen2.5-VL deep reasoning (opt-in):**
    Runs only when ``task_type`` is ``"VQA"`` or ``"TRAKE"``, or when the
    VLM is already loaded.  Operates on the top
    ``settings.deep_reasoning_top_k`` (default 5) candidates only.

    **Stage 3 — VQA answer extraction (opt-in):**
    Runs when ``request.extract_answer == True`` and ``task_type == "VQA"``.
    Calls the VLM on the top-1 reranked keyframe.

    Returns a :class:`~backend.schemas.common.BaseResponse` whose ``data``
    field is a serialised :class:`~backend.schemas.rerank.VQAResponse`.
    """
    start = time.perf_counter()
    settings = get_settings()

    session_tag = f"[{x_session_id}] " if x_session_id else ""
    logger.info(
        "%sRerank request: %d candidates, task=%s, extract_answer=%s",
        session_tag,
        len(request.candidates),
        request.task_type,
        request.extract_answer,
    )

    # ── Stage 1: Grounding DINO reranking ───────────────────────────
    try:
        reranked: list[RerankResultItem] = await reranker.rerank(
            query=request.query,
            candidates=request.candidates,
            keyframes_dir=keyframes_dir,
        )
    except ValueError as exc:
        # e.g. candidate count exceeded the 50-item limit
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Stage 2: VLM deep reasoning (top-5, opt-in) ─────────────────
    should_deep_reason = (
        vlm.is_loaded
        or request.task_type in ("VQA", "TRAKE")
    )
    if should_deep_reason and reranked:
        reranked = await vlm.deep_reason(
            query=request.query,
            candidates=reranked,
            keyframes_dir=keyframes_dir,
            top_k=settings.deep_reasoning_top_k,
        )

    # ── Stage 3: VQA answer extraction (top-1, opt-in) ──────────────
    vqa_answer: str | None = None
    if request.extract_answer and request.task_type == "VQA" and reranked:
        top_candidate = reranked[0]
        image_path = keyframes_dir / top_candidate.video_id / f"{top_candidate.frame_id:06d}.jpg"
        vqa_answer = await vlm.extract_vqa_answer(
            query=request.query,
            image_path=image_path,
        ) or None  # coerce "" to None

    elapsed = time.perf_counter() - start
    logger.info(
        "%sRerank complete: %d results, answer=%r, elapsed=%.3fs",
        session_tag,
        len(reranked),
        vqa_answer,
        elapsed,
    )

    payload = VQAResponse(results=reranked, vqa_answer=vqa_answer)

    return BaseResponse(
        status="success",
        data=payload.model_dump(),
        message=(
            f"Reranked {len(reranked)} candidates"
            + (f" | VQA: {vqa_answer}" if vqa_answer else "")
        ),
        execution_time=f"{elapsed:.3f}s",
    )
