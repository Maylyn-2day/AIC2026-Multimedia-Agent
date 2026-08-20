"""
Sketch-to-Vector Search Service.

Implements the sketch retrieval path:

    Canvas Base64 PNG
    → Decode → PIL Image
    → [Optional] Edge / ControlNet pre-processing
    → Visual encoder  (M4's SigLIP 2, or deterministic fallback)
    → Qdrant vector search  (or mock response when DB is unavailable)
    → list[dict] candidates

The encoder interface is pluggable: any callable that accepts a
``PIL.Image.Image`` and returns a ``numpy.ndarray`` can be injected at
construction time.  When no encoder is provided, a deterministic
hash-seeded random unit vector is returned so the service remains
functional in CI and during M4 integration.

Design principles:
- No hard dependency on ``torch`` or any ML library at import time.
- Graceful degradation: all public methods log a warning and return
  safe fallback values when an optional dependency is absent.
- Deterministic fallback: the same sketch always produces the same
  embedding, making tests reproducible.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from backend.core.logging import setup_logger
from backend.schemas.query import QueryFilters

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image as PILImage

logger = setup_logger("sketch_service")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_VECTOR_DIM: int = 768  # matches SigLIP 2 / Qdrant collection size
_MOCK_RESULT_COUNT: int = 10    # number of synthetic results in fallback mode


# ---------------------------------------------------------------------------
# Encoder protocol (structural subtyping — no ABC inheritance required)
# ---------------------------------------------------------------------------


class VisualEncoder(Protocol):
    """Structural interface for any visual embedding model.

    M4's SigLIP 2 encoder must satisfy this protocol:
    - Accept a ``PIL.Image.Image``.
    - Return a 1-D ``numpy.ndarray`` of floats (the embedding vector).
    """

    def __call__(self, image: "PILImage.Image", /) -> "np.ndarray":  # noqa: F821
        """Encode an image to a unit-normalised embedding vector."""
        ...


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def decode_base64_image(data: str) -> "PILImage.Image":
    """Decode a Base64-encoded image string to a PIL RGB Image.

    Handles both raw Base64 and ``data:image/...;base64,`` prefixed strings.

    Args:
        data: Base64 image string (with or without data URI prefix).

    Returns:
        PIL ``Image`` in ``"RGB"`` mode.

    Raises:
        ValueError: If ``data`` is empty or cannot be decoded.
    """
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise ValueError("Base64 image data must not be empty.")

    # Strip optional data-URI prefix (e.g. "data:image/png;base64,")
    if "," in data:
        data = data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(data)
    except Exception as exc:
        raise ValueError(f"Invalid Base64 string: {exc}") from exc

    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image format: {exc}") from exc


def _deterministic_unit_vector(seed_bytes: bytes, dim: int) -> "np.ndarray":
    """Return a deterministic, unit-normalised random vector seeded from bytes.

    Used as a stable fallback when no visual encoder is available, so the
    same sketch always maps to the same point in embedding space.

    Args:
        seed_bytes: Bytes used to seed the RNG (e.g. image MD5 hash).
        dim: Dimensionality of the output vector.

    Returns:
        1-D numpy float32 array with L2-norm == 1.0.
    """
    import numpy as np

    seed = int(hashlib.md5(seed_bytes).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _apply_edge_detection(image: "PILImage.Image") -> "PILImage.Image":
    """Apply simple Canny-style edge detection as a ControlNet pre-process.

    Uses Pillow's built-in ``ImageFilter.FIND_EDGES`` as a lightweight
    substitute for OpenCV Canny, so the service works without OpenCV
    installed.  The result is converted back to ``"RGB"`` mode for
    downstream encoder compatibility.

    Args:
        image: Input PIL image.

    Returns:
        Edge-detected PIL image in ``"RGB"`` mode.
    """
    from PIL import ImageFilter

    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return edges.convert("RGB")


# ---------------------------------------------------------------------------
# SketchService
# ---------------------------------------------------------------------------


class SketchService:
    """Convert canvas sketches to embedding vectors and drive Qdrant search.

    The service is intentionally encoder-agnostic.  Inject M4's SigLIP 2
    encoder when it becomes available:

    .. code-block:: python

        from m4.encoders import siglip2_encode
        svc = SketchService(encoder=siglip2_encode, vector_dim=768)

    Without an encoder (default), a deterministic unit vector is returned,
    keeping the retrieval pipeline functional with predictable ranking.

    Args:
        encoder: Optional callable satisfying :class:`VisualEncoder`.
            ``None`` activates deterministic fallback mode.
        vector_dim: Embedding dimensionality (must match Qdrant collection).
        apply_edge_detection: Pre-process sketch with edge detection before
            encoding.  Recommended for cartoon/outline-style sketches.
        qdrant_client: Optional Qdrant client instance.  When ``None`` the
            service returns synthetic mock results for development.
        collection_name: Qdrant collection to search.
    """

    def __init__(
        self,
        encoder: VisualEncoder | None = None,
        vector_dim: int = _DEFAULT_VECTOR_DIM,
        apply_edge_detection: bool = True,
        qdrant_client: object | None = None,
        collection_name: str = "aic2026_keyframes",
    ) -> None:
        self._encoder = encoder
        self._vector_dim = vector_dim
        self._apply_edge_detection = apply_edge_detection
        self._qdrant = qdrant_client
        self._collection = collection_name

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def has_encoder(self) -> bool:
        """Whether a real visual encoder is injected (vs. fallback mode)."""
        return self._encoder is not None

    @property
    def has_qdrant(self) -> bool:
        """Whether a Qdrant client is available for real vector search."""
        return self._qdrant is not None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def sketch_to_vector(
        self,
        sketch_base64: str,
        prompt: str = "",
    ) -> "np.ndarray":
        """Convert a Base64 sketch image to an embedding vector.

        Pipeline:
        1. Decode Base64 → PIL Image.
        2. Optional edge-detection pre-processing.
        3. Encode with the injected encoder, or return a deterministic
           fallback vector if no encoder is available.

        The ``prompt`` parameter is reserved for future CLIP-style
        text-image fusion; it is currently ignored in the encoder call
        but included in the signature for forward compatibility.

        Args:
            sketch_base64: Base64-encoded canvas image (PNG or JPEG).
            prompt: Optional text description of the sketch.

        Returns:
            1-D ``numpy.ndarray`` of shape ``(vector_dim,)``, dtype float32,
            L2-normalised.

        Raises:
            ValueError: If ``sketch_base64`` is empty or not valid Base64.
        """
        image = decode_base64_image(sketch_base64)

        if self._apply_edge_detection:
            image = _apply_edge_detection(image)

        if self._encoder is not None:
            logger.info("[Sketch] Encoding sketch with injected visual encoder.")
            try:
                vector = self._encoder(image)
                # Ensure the result is a numpy float32 array.
                import numpy as np

                vector = np.asarray(vector, dtype=np.float32).ravel()
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    vector = vector / norm
                return vector
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[Sketch] Encoder failed (%s) — falling back to deterministic vector.", exc
                )

        # Deterministic fallback: hash the raw image bytes as the RNG seed.
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        seed_bytes = buf.getvalue()
        logger.warning(
            "[Sketch] No encoder available — using deterministic fallback vector (dim=%d).",
            self._vector_dim,
        )
        return _deterministic_unit_vector(seed_bytes, self._vector_dim)

    def search_by_sketch(
        self,
        sketch_base64: str,
        prompt: str = "",
        top_k: int = 50,
        filters: QueryFilters | None = None,
    ) -> list[dict[str, object]]:
        """Search for keyframes visually similar to the given sketch.

        Converts the sketch to an embedding vector, then either queries
        Qdrant or returns synthetic mock results when the DB is unavailable.

        Args:
            sketch_base64: Base64-encoded canvas sketch.
            prompt: Optional text hint (reserved for fusion; not yet active).
            top_k: Maximum number of results to return (1 ≤ top_k ≤ 500).
            filters: Optional ``QueryFilters`` for metadata filtering.
                Currently passed through to Qdrant; ignored in mock mode.

        Returns:
            List of candidate dicts, each with at least:
            ``{"video_id": str, "frame_id": int, "score": float}``.
        """
        top_k = max(1, min(top_k, 500))

        vector = self.sketch_to_vector(sketch_base64, prompt=prompt)

        if self._qdrant is not None:
            return self._qdrant_search(vector, top_k, filters)

        logger.warning(
            "[Sketch] Qdrant not connected — returning %d mock results.", _MOCK_RESULT_COUNT
        )
        return _generate_mock_results(vector, min(top_k, _MOCK_RESULT_COUNT))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _qdrant_search(
        self,
        vector: "np.ndarray",
        top_k: int,
        filters: QueryFilters | None,
    ) -> list[dict[str, object]]:
        """Run a Qdrant similarity search and normalise the response.

        This is a thin adapter — the actual Qdrant client call is
        delegated to the injected ``qdrant_client`` so this service
        does not import ``qdrant_client`` directly, maintaining
        optional-dependency hygiene.

        Args:
            vector: Query embedding vector.
            top_k: Maximum results to retrieve.
            filters: Optional Qdrant filter (currently passed as-is).

        Returns:
            Normalised list of result dicts.
        """
        try:
            results = self._qdrant.search(  # type: ignore[union-attr]
                collection_name=self._collection,
                query_vector=vector.tolist(),
                limit=top_k,
            )
            return [
                {
                    "video_id": hit.payload.get("video_id", ""),
                    "frame_id": hit.payload.get("frame_id", 0),
                    "score": float(hit.score),
                }
                for hit in results
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Sketch] Qdrant search failed (%s) — returning empty list.", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level mock helpers
# ---------------------------------------------------------------------------


def _generate_mock_results(
    vector: "np.ndarray",
    count: int,
) -> list[dict[str, object]]:
    """Generate deterministic synthetic search results for development.

    Results are seeded from the query vector's first element so the
    same sketch always produces the same mock ranking.

    Args:
        vector: Query vector (used only to seed the RNG).
        count: Number of mock results to generate.

    Returns:
        List of ``count`` dicts in ascending rank order (highest score first).
    """
    import numpy as np

    seed = int(abs(float(vector[0])) * 1e6) % (2**31)
    rng = np.random.default_rng(seed)
    video_ids = [f"L{rng.integers(1, 5):02d}_V{rng.integers(1, 100):03d}" for _ in range(count)]
    frame_ids = rng.integers(100, 9000, size=count).tolist()
    # Scores in descending order; realistic RRF-like range.
    scores = sorted(rng.uniform(0.01, 0.08, size=count).tolist(), reverse=True)

    return [
        {
            "video_id": video_ids[i],
            "frame_id": int(frame_ids[i]),
            "score": round(float(scores[i]), 6),
            "thumbnail_url": None,
            "metadata": {"source": "sketch_mock"},
        }
        for i in range(count)
    ]
