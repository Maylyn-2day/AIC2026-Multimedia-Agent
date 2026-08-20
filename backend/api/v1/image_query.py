"""
POST /v1/query/image-example — Image-to-Image Similarity Search.

Two input modes:
  A) ``image_base64`` — Upload / paste a raw image; extract embedding via
     SketchService (same encoder pipeline as sketch search).
  B) ``video_id`` + ``frame_id`` — Reference an existing keyframe by
     identity; resolve its path on disk and encode it.

Both modes return a ranked list of visually similar keyframes.
Latency budget: <100ms on GPU; <2s in mock/fallback mode.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException

from backend.core.config import get_settings
from backend.core.exceptions import InvalidQueryError
from backend.core.logging import setup_logger
from backend.schemas.common import BaseResponse, KeyframeResult
from backend.schemas.image import ImageQueryRequest
from backend.services.sketch_service import SketchService
from frontend.utils.image_utils import base64_encode

logger = setup_logger("image_query_route")
router = APIRouter()

# ---------------------------------------------------------------------------
# Shared service singleton — same SketchService as sketch route
# (encoder + Qdrant client injected once, shared across both routes)
# ---------------------------------------------------------------------------

_settings = get_settings()

_sketch_service: SketchService = SketchService(
    encoder=None,
    vector_dim=_settings.qdrant_vector_size,
    apply_edge_detection=False,   # no edge-detect for photo queries
    qdrant_client=None,
    collection_name=_settings.qdrant_collection_name,
)


def get_sketch_service() -> SketchService:
    """Provide the shared SketchService singleton for image-query route."""
    return _sketch_service


def get_keyframes_dir() -> Path:
    """Resolve keyframes root directory from Settings."""
    return Path(get_settings().keyframes_dir)


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post(
    "/query/image-example",
    response_model=BaseResponse,
    summary="Image-to-Image Search",
    description=(
        "Find visually similar keyframes using a reference image. "
        "Accepts either a Base64-encoded image (Mode A) or a video_id + "
        "frame_id pair that resolves to a keyframe on disk (Mode B)."
    ),
)
async def image_query(
    request: ImageQueryRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
    sketch_service: SketchService = Depends(get_sketch_service),
    keyframes_dir: Path = Depends(get_keyframes_dir),
) -> BaseResponse:
    """Execute image-to-image similarity search.

    **Mode A** — ``image_base64`` is provided:
    Decodes the Base64 payload to a PIL image, re-encodes it as PNG,
    and passes it through SketchService for embedding + search.

    **Mode B** — ``video_id`` + ``frame_id`` are provided:
    Resolves the path ``{keyframes_dir}/{video_id}/{frame_id:06d}.jpg``,
    loads the JPEG, converts it to Base64, then follows the same path.

    Falls back to mock results when Qdrant is not connected.
    """
    start = time.perf_counter()
    session_tag = f"[{x_session_id}] " if x_session_id else ""

    # ── Validate: at least one input mode must be provided ───────────
    if request.image_base64 is None and (
        request.video_id is None or request.frame_id is None
    ):
        raise InvalidQueryError(
            "Provide either 'image_base64' or both 'video_id' and 'frame_id'."
        )

    # ── Resolve sketch_base64 from whichever mode was provided ───────
    sketch_base64: str

    if request.image_base64 is not None:
        # Mode A: use the supplied base64 directly
        sketch_base64 = request.image_base64
        logger.info("%sImage query [Mode A]: base64 input, top_k=%d", session_tag, request.top_k)

    else:
        # Mode B: resolve keyframe path and encode to base64
        assert request.video_id is not None
        assert request.frame_id is not None

        image_path = keyframes_dir / request.video_id / f"{request.frame_id:06d}.jpg"

        if not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Keyframe not found: {image_path}",
            )

        try:
            from PIL import Image
            pil_img = Image.open(image_path).convert("RGB")
            sketch_base64 = base64_encode(pil_img, fmt="PNG")
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot read keyframe image: {exc}",
            ) from exc

        logger.info(
            "%sImage query [Mode B]: %s/%06d, top_k=%d",
            session_tag, request.video_id, request.frame_id, request.top_k,
        )

    # ── Run vector search ─────────────────────────────────────────────
    raw_results = sketch_service.search_by_sketch(
        sketch_base64=sketch_base64,
        top_k=request.top_k,
    )

    # Normalise to KeyframeResult schema
    results = [
        KeyframeResult(
            video_id=r["video_id"],
            frame_id=r["frame_id"],
            score=float(r["score"]),
            thumbnail_url=r.get("thumbnail_url"),   # type: ignore[arg-type]
            metadata=r.get("metadata", {}),         # type: ignore[arg-type]
        ).model_dump()
        for r in raw_results
    ]

    elapsed = time.perf_counter() - start
    mode = "qdrant" if sketch_service.has_qdrant else "mock"
    logger.info(
        "%sImage query complete: %d results [%s] %.3fs",
        session_tag, len(results), mode, elapsed,
    )

    return BaseResponse(
        status="success",
        data={
            "results": results,
            "total": len(results),
            "source": {
                "video_id": request.video_id,
                "frame_id": request.frame_id,
                "mode": "base64" if request.image_base64 is not None else "keyframe_id",
            },
        },
        message=f"Image search returned {len(results)} results ({mode} mode)",
        execution_time=f"{elapsed:.3f}s",
    )
