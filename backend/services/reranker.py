"""
Grounding DINO Visual Grounding Reranker Service.

Implements Stage 2 of the cascading retrieval pipeline:

    Dense+Sparse (Qdrant + ES)
    → RRF Fusion (fusion.py)
    → [THIS MODULE] Grounding DINO — bbox verification, Top-50
    → VLM deep reasoning (vlm_service.py)

The reranker receives the top-N RRF candidates and verifies each
keyframe image using Grounding DINO to detect whether the query
object(s) are actually present.  Candidates are re-scored by
combining the RRF score with the grounding confidence:

    final_score = rrf_score * grounding_confidence

If the model is not loaded (no GPU / VRAM budget exceeded), the
service transparently falls back to pass-through mode, preserving
the original RRF ordering with a ``grounding_confidence`` of 1.0.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from backend.core.exceptions import ModelNotLoadedError, VRAMOverflowError
from backend.core.logging import setup_logger
from backend.schemas.rerank import GroundingResult, RerankCandidate, RerankResultItem

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from PIL import Image

logger = setup_logger("reranker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BOX_THRESHOLD: float = 0.35
_DEFAULT_TEXT_THRESHOLD: float = 0.25
_MAX_INPUT_CANDIDATES: int = 50  # API contract: Stage 2 receives ≤ 50 frames

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_keyframe_path(
    video_id: str,
    frame_id: int,
    keyframes_dir: Path | str,
) -> Path:
    """Return the canonical path for a stored keyframe image.

    Convention: ``{keyframes_dir}/{video_id}/{frame_id:06d}.jpg``

    Args:
        video_id: Video identifier, e.g. ``"L01_V001"``.
        frame_id: Zero-based or BTC-assigned frame index.
        keyframes_dir: Root directory containing per-video subdirectories.

    Returns:
        Resolved :class:`pathlib.Path` for the keyframe JPEG.
    """
    return Path(keyframes_dir) / video_id / f"{frame_id:06d}.jpg"


# ---------------------------------------------------------------------------
# Grounding DINO model wrapper
# ---------------------------------------------------------------------------


class GroundingReranker:
    """Grounding DINO-based visual grounding service for candidate reranking.

    Implements lazy model loading so the service can be instantiated at
    application startup without immediately consuming VRAM.  Call
    :meth:`load_model` during the FastAPI lifespan startup hook (or on
    the first request) when GPU resources are confirmed available.

    Example::

        reranker = GroundingReranker(
            model_id="IDEA-Research/grounding-dino-base",
            device="cuda",
        )
        await reranker.load_model()
        results = await reranker.rerank(
            query="woman in red shirt",
            candidates=rrf_output[:50],
            keyframes_dir=settings.keyframes_dir,
        )
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        box_threshold: float = _DEFAULT_BOX_THRESHOLD,
        text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
        max_vram_gb: float = 8.0,
        alpha: float = 0.5,
    ) -> None:
        """Initialise the reranker without loading model weights.

        Args:
            model_id: HuggingFace model identifier for Grounding DINO.
            device: PyTorch device string (``"cuda"`` or ``"cpu"``).
                Falls back to ``"cpu"`` automatically if CUDA is unavailable.
            box_threshold: Minimum confidence for predicted bounding boxes.
            text_threshold: Minimum confidence for text–box alignment.
            max_vram_gb: VRAM safety ceiling; raises
                :class:`VRAMOverflowError` if loading would exceed it.
            alpha: Grounding boost weight in the scoring formula
                ``rrf_score * (1 + alpha * grounding_confidence)``.
                Higher values reward confirmed detections more aggressively.
                Must be >= 0.  Defaults to ``0.5``.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        self._model_id = model_id
        self._device = device
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._max_vram_gb = max_vram_gb
        self._alpha = alpha

        self._processor: object | None = None
        self._model: object | None = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the Grounding DINO model weights are resident in memory."""
        return self._is_loaded

    @property
    def device(self) -> str:
        """Effective compute device (may differ from requested if CUDA absent)."""
        return self._device

    @property
    def alpha(self) -> float:
        """Grounding boost weight used in ``rrf_score * (1 + alpha * confidence)``."""
        return self._alpha

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load_model(self) -> None:
        """Lazy-load Grounding DINO processor and model weights.

        Verifies CUDA availability and estimated VRAM usage before loading.
        Sets the model to ``eval()`` mode and disables gradient computation
        context for all subsequent inference calls.

        Raises:
            VRAMOverflowError: If the estimated model size exceeds
                ``max_vram_gb``.
            ModelNotLoadedError: If the HuggingFace download or
                ``torch`` import fails.
        """
        if self._is_loaded:
            logger.info("[Reranker] Model already loaded — skipping.")
            return

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ModelNotLoadedError(self._model_id) from exc

        # ── CUDA availability check ──────────────────────────────────
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "[Reranker] CUDA requested but unavailable — falling back to CPU."
            )
            self._device = "cpu"

        # ── VRAM guard (Grounding DINO base ≈ 1.8 GB on GPU) ────────
        if self._device == "cuda":
            available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            estimated_gb = 1.8  # approximate for grounding-dino-base
            if estimated_gb > self._max_vram_gb or estimated_gb > available_gb:
                raise VRAMOverflowError(
                    required_gb=estimated_gb,
                    available_gb=min(available_gb, self._max_vram_gb),
                )

        logger.info("[Reranker] Loading %s on %s …", self._model_id, self._device)
        try:
            processor = AutoProcessor.from_pretrained(self._model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self._model_id)
            model = model.to(self._device)
            model.eval()
        except Exception as exc:
            raise ModelNotLoadedError(self._model_id) from exc

        self._processor = processor
        self._model = model
        self._is_loaded = True
        logger.info("[Reranker] Grounding DINO loaded successfully on %s.", self._device)

    def unload_model(self) -> None:
        """Release model weights and free associated GPU/CPU memory.

        Safe to call even when the model is not loaded.
        """
        if not self._is_loaded:
            return

        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._is_loaded = False

        try:
            import torch

            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch wasn't importable; no VRAM to free

        logger.info("[Reranker] Model unloaded and VRAM released.")

    # ------------------------------------------------------------------
    # Core reranking logic
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        keyframes_dir: Path | str,
    ) -> list[RerankResultItem]:
        """Rerank candidates using visual grounding verification.

        For each candidate keyframe:
        1. Resolve the image path via :func:`resolve_keyframe_path`.
        2. Run Grounding DINO (or fallback) to obtain bounding boxes.
        3. Compute ``final_score = rrf_score * max_grounding_confidence``.
        4. Sort descending by ``final_score`` and assign sequential ranks.

        If a keyframe image cannot be read (missing file, corrupt JPEG),
        the candidate is assigned ``grounding_confidence = 0.0`` and a
        synthetic fallback score, but is **not** silently dropped — the
        caller must decide whether to exclude zero-score results.

        Args:
            query: Natural-language text prompt describing the target object
                or scene (e.g. ``"người phụ nữ mặc áo đỏ"``).
            candidates: Validated :class:`RerankCandidate` list from the API
                boundary (max :data:`_MAX_INPUT_CANDIDATES` items).
            keyframes_dir: Root directory for keyframe images.

        Returns:
            List of :class:`RerankResultItem` sorted by ``rerank_score``
            descending.  Length equals ``len(candidates)``.

        Raises:
            ValueError: If ``len(candidates) > _MAX_INPUT_CANDIDATES``.
        """
        if len(candidates) > _MAX_INPUT_CANDIDATES:
            raise ValueError(
                f"Reranker input exceeds maximum {_MAX_INPUT_CANDIDATES} candidates "
                f"(received {len(candidates)}). Filter upstream with RRF top_k."
            )

        if not candidates:
            return []

        if not self._is_loaded:
            logger.warning(
                "[Reranker] Model not loaded — using pass-through scoring for %d candidates.",
                len(candidates),
            )
            return self._passthrough_results(candidates)

        results: list[RerankResultItem] = []
        for candidate in candidates:
            image_path = resolve_keyframe_path(
                candidate.video_id, candidate.frame_id, keyframes_dir
            )
            grounding_boxes = await self._ground_single(query, image_path)
            max_confidence = (
                max(b.confidence for b in grounding_boxes) if grounding_boxes else 0.0
            )
            rerank_score = _compute_rerank_score(
                rrf_score=candidate.rrf_score if candidate.rrf_score is not None else candidate.score,
                grounding_confidence=max_confidence,
                alpha=self._alpha,
            )
            results.append(
                RerankResultItem(
                    video_id=candidate.video_id,
                    frame_id=candidate.frame_id,
                    original_score=candidate.score,
                    rerank_score=rerank_score,
                    grounding=grounding_boxes,
                )
            )

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ground_single(
        self,
        query: str,
        image_path: Path,
    ) -> list[GroundingResult]:
        """Run Grounding DINO on one keyframe and return bounding boxes.

        Args:
            query: Text prompt for Grounding DINO (object description).
            image_path: Absolute path to the keyframe JPEG.

        Returns:
            List of :class:`GroundingResult` objects (may be empty if the
            model detects nothing above the confidence threshold, or if the
            image file is unreadable).
        """
        import torch
        from PIL import Image, UnidentifiedImageError

        if not image_path.exists():
            logger.warning("[Reranker] Keyframe not found: %s", image_path)
            return []

        try:
            image: Image.Image = Image.open(image_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            logger.warning("[Reranker] Cannot read keyframe %s: %s", image_path, exc)
            return []

        # Grounding DINO expects the prompt terminated with a period.
        text_prompt = query.strip().rstrip(".") + "."

        inputs = self._processor(  # type: ignore[operator]
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[operator]

        results = self._processor.post_process_grounded_object_detection(  # type: ignore[union-attr]
            outputs,
            inputs["input_ids"],
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )

        return _parse_grounding_output(results[0], image.size)

    @staticmethod
    def _passthrough_results(candidates: list[RerankCandidate]) -> list[RerankResultItem]:
        """Return pass-through results preserving original candidate ordering.

        Used when the Grounding DINO model is not loaded.  Each candidate
        receives a ``grounding_confidence`` of 1.0 so the ``rerank_score``
        equals the original retrieval score.
        """
        return [
            RerankResultItem(
                video_id=c.video_id,
                frame_id=c.frame_id,
                original_score=c.score,
                rerank_score=c.rrf_score if c.rrf_score is not None else c.score,
                grounding=[],
            )
            for c in candidates
        ]


# ---------------------------------------------------------------------------
# Module-level pure functions (deterministic, easily unit-testable)
# ---------------------------------------------------------------------------


def _compute_rerank_score(
    rrf_score: float,
    grounding_confidence: float,
    alpha: float = 0.5,
) -> float:
    """Compute the final rerank score from RRF and grounding signals.

    Formula: ``final_score = rrf_score * (1 + alpha * grounding_confidence)``

    **Design rationale:**
    The previous multiplicative formula (``rrf_score * confidence``) suffered
    from a *zero-score trap*: any frame where Grounding DINO failed to detect
    an object (occlusion, small scale, motion blur) received a score of 0.0,
    completely discarding the prior dense + sparse fusion signal.

    The additive boost avoids this by treating the grounding step as a
    *reward* rather than a *gate*:

    - ``confidence == 0.0``  →  ``final_score = rrf_score * 1.0 = rrf_score``
      (baseline preserved; candidate is not penalised).
    - ``confidence == 1.0``  →  ``final_score = rrf_score * (1 + alpha)``
      (maximum boost; e.g. 1.5× at the default ``alpha = 0.5``).

    Args:
        rrf_score: The RRF fusion score from :func:`weighted_rrf`.
        grounding_confidence: Maximum detection confidence from Grounding
            DINO across all bounding boxes for this candidate (in [0, 1]).
        alpha: Boost weight.  Must be >= 0.  Defaults to ``0.5``.

    Returns:
        Non-negative rerank score.
    """
    return float(rrf_score) * (1.0 + float(alpha) * float(grounding_confidence))


def _parse_grounding_output(
    detection: dict[str, object],
    image_size: tuple[int, int],
) -> list[GroundingResult]:
    """Convert Grounding DINO post-processed output to schema objects.

    Normalises absolute pixel bounding boxes to the [0, 1] range so the
    frontend can render overlays independent of the original resolution.

    Args:
        detection: Single-image dict from
            ``post_process_grounded_object_detection``.  Expected keys:
            ``"boxes"`` (``Tensor[N, 4]``), ``"scores"`` (``Tensor[N]``),
            ``"labels"`` (``list[str]``).
        image_size: ``(width, height)`` of the source image.

    Returns:
        List of :class:`GroundingResult`, one per detected box.  Empty if
        no boxes are present or if expected keys are missing.
    """
    import torch

    boxes: torch.Tensor | None = detection.get("boxes")  # type: ignore[assignment]
    scores: torch.Tensor | None = detection.get("scores")  # type: ignore[assignment]
    labels: list[str] | None = detection.get("labels")  # type: ignore[assignment]

    if boxes is None or scores is None or labels is None:
        return []
    if len(boxes) == 0:
        return []

    width, height = image_size
    results: list[GroundingResult] = []
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels):
        x1, y1, x2, y2 = box
        results.append(
            GroundingResult(
                label=str(label),
                confidence=float(score),
                bbox=[
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
            )
        )
    return results
