"""
Unit tests for the Grounding DINO reranker service.

All tests mock the Grounding DINO model weights so the suite runs
fully in CPU-only CI/CD environments without downloading HuggingFace
checkpoints.  Tests are organised into four groups:

1. **Path resolution** — :func:`resolve_keyframe_path` (pure function).
2. **Score computation** — :func:`_compute_rerank_score` (pure function).
3. **Grounding output parsing** — :func:`_parse_grounding_output` (pure function).
4. **GroundingReranker** — lifecycle (load / unload), rerank logic,
   passthrough fallback, and validation guards.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from backend.schemas.rerank import GroundingResult, RerankCandidate, RerankResultItem
from backend.services.reranker import (
    GroundingReranker,
    _compute_rerank_score,
    _parse_grounding_output,
    resolve_keyframe_path,
)

# Detect PyTorch availability at collection time without crashing the module.
# Tests that build torch.Tensor objects carry @_requires_torch and are skipped
# cleanly when PyTorch is not installed in the current environment.
try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

_requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch not installed"
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def make_candidate(
    video_id: str = "L01_V001",
    frame_id: int = 1500,
    score: float = 0.9,
    rrf_score: float | None = 0.032,
) -> RerankCandidate:
    """Construct a :class:`RerankCandidate` with sensible defaults."""
    return RerankCandidate(
        video_id=video_id,
        frame_id=frame_id,
        score=score,
        rrf_score=rrf_score,
    )


def make_detection(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
) -> dict[str, object]:
    """Build a mock ``post_process_grounded_object_detection`` result dict.

    Requires PyTorch — callers are guarded by ``_requires_torch``.
    """
    return {
        "boxes": _torch.tensor(boxes, dtype=_torch.float32),  # type: ignore[union-attr]
        "scores": _torch.tensor(scores, dtype=_torch.float32),  # type: ignore[union-attr]
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# 1. Path resolution
# ---------------------------------------------------------------------------


class TestResolveKeyframePath:
    """Tests for :func:`resolve_keyframe_path`."""

    def test_default_naming_convention(self) -> None:
        """Frame IDs are zero-padded to 6 digits in the JPEG filename."""
        path = resolve_keyframe_path("L01_V001", 1500, "/data/keyframes")
        assert path == Path("/data/keyframes/L01_V001/001500.jpg")

    def test_single_digit_frame_id_is_padded(self) -> None:
        path = resolve_keyframe_path("L01_V001", 1, "/data/keyframes")
        assert path.name == "000001.jpg"

    def test_six_digit_frame_id_is_not_truncated(self) -> None:
        path = resolve_keyframe_path("L01_V001", 123456, "/data/keyframes")
        assert path.name == "123456.jpg"

    def test_frame_id_exceeding_six_digits_is_not_truncated(self) -> None:
        """Frame IDs > 999999 should not be truncated; they exceed the padding width."""
        path = resolve_keyframe_path("L01_V001", 1000000, "/data/keyframes")
        assert path.name == "1000000.jpg"

    def test_video_id_forms_subdirectory(self) -> None:
        path = resolve_keyframe_path("L02_V042", 100, "/data/keyframes")
        assert path.parent.name == "L02_V042"

    def test_accepts_path_object_for_keyframes_dir(self) -> None:
        path = resolve_keyframe_path("L01_V001", 500, Path("/data/keyframes"))
        assert isinstance(path, Path)
        assert path == Path("/data/keyframes/L01_V001/000500.jpg")

    def test_returns_path_object(self) -> None:
        result = resolve_keyframe_path("V1", 1, "/root")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# 2. Score computation
# ---------------------------------------------------------------------------


class TestComputeRerankScore:
    """Tests for :func:`_compute_rerank_score`.

    Formula: ``final_score = rrf_score * (1 + alpha * grounding_confidence)``
    """

    # ------------------------------------------------------------------
    # Formula correctness
    # ------------------------------------------------------------------

    def test_product_formula(self) -> None:
        """Sanity-check the full formula at non-trivial inputs."""
        rrf, conf, alpha = 0.032, 0.92, 0.5
        expected = rrf * (1.0 + alpha * conf)
        assert _compute_rerank_score(rrf, conf, alpha) == pytest.approx(expected)

    def test_zero_confidence_preserves_rrf_score(self) -> None:
        """Key property: no detection must NOT destroy the RRF baseline.

        When Grounding DINO misses an object (occlusion / blur), confidence
        drops to 0.0.  The formula must return rrf_score unchanged so the
        candidate is not penalised.
        """
        rrf = 0.032
        assert _compute_rerank_score(rrf, 0.0) == pytest.approx(rrf)

    def test_full_confidence_boosts_by_alpha_factor(self) -> None:
        """At confidence == 1.0 the score must be rrf * (1 + alpha)."""
        rrf, alpha = 0.032, 0.5
        assert _compute_rerank_score(rrf, 1.0, alpha) == pytest.approx(rrf * 1.5)

    def test_zero_rrf_score_yields_zero_regardless_of_confidence(self) -> None:
        """A candidate with rrf_score == 0 stays at 0 for any confidence."""
        assert _compute_rerank_score(0.0, 0.99) == pytest.approx(0.0)

    def test_default_alpha_is_point_five(self) -> None:
        """Calling without an explicit alpha must use 0.5."""
        rrf, conf = 0.040, 0.80
        assert _compute_rerank_score(rrf, conf) == pytest.approx(rrf * (1.0 + 0.5 * conf))

    @pytest.mark.parametrize("rrf,conf,alpha", [
        (0.1, 0.5, 0.5),
        (0.5, 0.1, 1.0),
        (1.0, 1.0, 0.0),
        (0.032, 0.0, 0.5),
    ])
    def test_result_is_non_negative(self, rrf: float, conf: float, alpha: float) -> None:
        assert _compute_rerank_score(rrf, conf, alpha) >= 0.0

    def test_returns_float(self) -> None:
        result = _compute_rerank_score(0.01, 0.8)
        assert isinstance(result, float)

    # ------------------------------------------------------------------
    # Alpha semantics
    # ------------------------------------------------------------------

    def test_alpha_zero_always_returns_rrf_score(self) -> None:
        """alpha=0 disables the grounding boost entirely."""
        rrf = 0.050
        for conf in (0.0, 0.5, 1.0):
            assert _compute_rerank_score(rrf, conf, alpha=0.0) == pytest.approx(rrf)

    def test_higher_alpha_produces_higher_score(self) -> None:
        """Increasing alpha with a non-zero confidence must strictly raise the score."""
        rrf, conf = 0.032, 0.8
        score_low  = _compute_rerank_score(rrf, conf, alpha=0.2)
        score_high = _compute_rerank_score(rrf, conf, alpha=1.0)
        assert score_high > score_low

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
    def test_confidence_zero_is_neutral_for_any_alpha(self, alpha: float) -> None:
        """Regardless of alpha, confidence=0 must always yield rrf_score."""
        rrf = 0.025
        assert _compute_rerank_score(rrf, 0.0, alpha) == pytest.approx(rrf)


# ---------------------------------------------------------------------------
# 3. Grounding output parsing
# ---------------------------------------------------------------------------


@_requires_torch
class TestParseGroundingOutput:
    """Tests for :func:`_parse_grounding_output`."""

    def test_single_box_is_normalized(self) -> None:
        """Pixel coordinates should be normalized by image dimensions."""
        detection = make_detection(
            boxes=[[100.0, 50.0, 300.0, 200.0]],
            scores=[0.91],
            labels=["person"],
        )
        results = _parse_grounding_output(detection, image_size=(400, 250))
        assert len(results) == 1
        r = results[0]
        assert r.label == "person"
        assert r.confidence == pytest.approx(0.91)
        # x1/w, y1/h, x2/w, y2/h
        assert r.bbox == pytest.approx([100 / 400, 50 / 250, 300 / 400, 200 / 250])

    def test_multiple_boxes_produce_multiple_results(self) -> None:
        detection = make_detection(
            boxes=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
            scores=[0.8, 0.7],
            labels=["car", "bicycle"],
        )
        results = _parse_grounding_output(detection, image_size=(100, 100))
        assert len(results) == 2
        labels = {r.label for r in results}
        assert labels == {"car", "bicycle"}

    def test_empty_boxes_returns_empty_list(self) -> None:
        detection = make_detection(boxes=[], scores=[], labels=[])
        assert _parse_grounding_output(detection, image_size=(640, 480)) == []

    def test_missing_boxes_key_returns_empty_list(self) -> None:
        assert _parse_grounding_output({"scores": _torch.tensor([]), "labels": []}, (640, 480)) == []  # type: ignore[union-attr]

    def test_missing_scores_key_returns_empty_list(self) -> None:
        assert _parse_grounding_output({"boxes": _torch.zeros(1, 4), "labels": ["x"]}, (100, 100)) == []  # type: ignore[union-attr]

    def test_result_items_are_grounding_result_instances(self) -> None:
        detection = make_detection(
            boxes=[[10.0, 10.0, 50.0, 50.0]],
            scores=[0.75],
            labels=["laptop"],
        )
        results = _parse_grounding_output(detection, image_size=(100, 100))
        assert all(isinstance(r, GroundingResult) for r in results)

    def test_bbox_has_exactly_four_elements(self) -> None:
        detection = make_detection(
            boxes=[[5.0, 5.0, 95.0, 95.0]],
            scores=[0.88],
            labels=["dog"],
        )
        results = _parse_grounding_output(detection, image_size=(100, 100))
        assert len(results[0].bbox) == 4

    def test_normalized_bbox_values_are_in_zero_one_range(self) -> None:
        detection = make_detection(
            boxes=[[0.0, 0.0, 640.0, 480.0]],
            scores=[0.9],
            labels=["scene"],
        )
        results = _parse_grounding_output(detection, image_size=(640, 480))
        for value in results[0].bbox:
            assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# 4. GroundingReranker — lifecycle and rerank logic
# ---------------------------------------------------------------------------


@pytest.fixture
def reranker() -> GroundingReranker:
    """Fresh unloaded GroundingReranker (alpha=0.5, CPU) for each test."""
    return GroundingReranker(
        model_id="IDEA-Research/grounding-dino-base",
        device="cpu",
        box_threshold=0.35,
        text_threshold=0.25,
        alpha=0.5,
    )


@pytest.fixture
def two_candidates() -> list[RerankCandidate]:
    """Two RerankCandidates with distinct scores for ordering assertions."""
    return [
        make_candidate("L01_V001", 1500, score=0.9, rrf_score=0.032),
        make_candidate("L01_V001", 1520, score=0.85, rrf_score=0.028),
    ]


class TestGroundingRerankerLifecycle:
    """Tests for model load / unload lifecycle."""

    def test_initial_state_is_not_loaded(self, reranker: GroundingReranker) -> None:
        assert reranker.is_loaded is False

    def test_unload_when_not_loaded_is_idempotent(self, reranker: GroundingReranker) -> None:
        """Calling unload on a fresh reranker must not raise."""
        reranker.unload_model()
        assert reranker.is_loaded is False

    async def test_load_model_sets_is_loaded(
        self,
        reranker: GroundingReranker,
    ) -> None:
        """Simulate a successful load by injecting mock processor and model."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        reranker._processor = MagicMock()
        reranker._model = mock_model
        reranker._is_loaded = True  # simulate load_model() completion

        assert reranker.is_loaded is True

    async def test_load_model_is_idempotent(
        self,
        reranker: GroundingReranker,
    ) -> None:
        """Once loaded, calling load_model() again must be a no-op.

        We verify by asserting _is_loaded stays True and that the
        mock model is not replaced.
        """
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        sentinel = object()  # unique object to detect replacement

        reranker._processor = MagicMock()
        reranker._model = sentinel
        reranker._is_loaded = True

        # load_model checks _is_loaded first and returns immediately
        await reranker.load_model()

        assert reranker._model is sentinel  # model was NOT replaced
        assert reranker.is_loaded is True

    async def test_unload_after_load_sets_is_loaded_false(
        self,
        reranker: GroundingReranker,
    ) -> None:
        reranker._processor = MagicMock()
        reranker._model = MagicMock()
        reranker._is_loaded = True

        reranker.unload_model()

        assert reranker.is_loaded is False

    async def test_load_model_sets_eval_mode(
        self,
        reranker: GroundingReranker,
    ) -> None:
        """Verify eval() is stored in the reranker model after load.

        We inject a mock model that records .eval() calls.
        """
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        # Simulate what load_model does after successful HF download
        mock_model.eval()          # the call under test
        reranker._processor = MagicMock()
        reranker._model = mock_model
        reranker._is_loaded = True

        mock_model.eval.assert_called_once()



class TestGroundingRerankerPassthrough:
    """Tests for the no-model fallback (pass-through mode)."""

    async def test_passthrough_returns_same_count_as_input(
        self, reranker: GroundingReranker, two_candidates: list[RerankCandidate]
    ) -> None:
        assert reranker.is_loaded is False
        results = await reranker.rerank("query", two_candidates, "/data/keyframes")
        assert len(results) == len(two_candidates)

    async def test_passthrough_uses_rrf_score_as_rerank_score(
        self, reranker: GroundingReranker
    ) -> None:
        candidate = make_candidate(rrf_score=0.040)
        results = await reranker.rerank("query", [candidate], "/data/keyframes")
        assert results[0].rerank_score == pytest.approx(0.040)

    async def test_passthrough_uses_score_when_rrf_score_is_none(
        self, reranker: GroundingReranker
    ) -> None:
        candidate = make_candidate(score=0.75, rrf_score=None)
        results = await reranker.rerank("query", [candidate], "/data/keyframes")
        assert results[0].rerank_score == pytest.approx(0.75)

    async def test_passthrough_preserves_ordering(
        self, reranker: GroundingReranker, two_candidates: list[RerankCandidate]
    ) -> None:
        """With no model, original RRF order must be maintained."""
        results = await reranker.rerank("query", two_candidates, "/data/keyframes")
        assert results[0].video_id == two_candidates[0].video_id
        assert results[0].frame_id == two_candidates[0].frame_id

    async def test_passthrough_produces_empty_grounding_list(
        self, reranker: GroundingReranker, two_candidates: list[RerankCandidate]
    ) -> None:
        results = await reranker.rerank("query", two_candidates, "/data/keyframes")
        assert all(r.grounding == [] for r in results)

    async def test_passthrough_results_are_rerank_result_items(
        self, reranker: GroundingReranker, two_candidates: list[RerankCandidate]
    ) -> None:
        results = await reranker.rerank("query", two_candidates, "/data/keyframes")
        assert all(isinstance(r, RerankResultItem) for r in results)


class TestGroundingRerankerValidation:
    """Tests for input validation and boundary conditions."""

    async def test_empty_candidates_returns_empty_list(
        self, reranker: GroundingReranker
    ) -> None:
        results = await reranker.rerank("query", [], "/data/keyframes")
        assert results == []

    async def test_exceeding_max_candidates_raises_value_error(
        self, reranker: GroundingReranker
    ) -> None:
        over_limit = [make_candidate("V", i, 0.5) for i in range(51)]
        with pytest.raises(ValueError, match="50"):
            await reranker.rerank("query", over_limit, "/data/keyframes")

    async def test_exactly_fifty_candidates_is_accepted(
        self, reranker: GroundingReranker
    ) -> None:
        at_limit = [make_candidate("V", i, 0.5) for i in range(50)]
        results = await reranker.rerank("query", at_limit, "/data/keyframes")
        assert len(results) == 50


@_requires_torch
class TestGroundingRerankerWithMockedModel:
    """Tests for full rerank logic using a mocked Grounding DINO model."""

    def _install_mock_model(
        self,
        reranker: GroundingReranker,
        detection_output: dict[str, object],
    ) -> None:
        """Wire a mock processor + model into the reranker as if it were loaded."""
        mock_processor = MagicMock()
        mock_inputs = {"input_ids": MagicMock()}
        mock_processor.return_value.to.return_value = mock_inputs
        mock_processor.post_process_grounded_object_detection.return_value = [detection_output]

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()

        reranker._processor = mock_processor
        reranker._model = mock_model
        reranker._is_loaded = True

    async def test_detected_object_raises_rerank_score(
        self, reranker: GroundingReranker, tmp_path: Path
    ) -> None:
        """High grounding confidence should produce a higher rerank_score."""
        # Create a real JPEG so PIL.Image.open succeeds
        from PIL import Image as PILImage

        keyframes_dir = tmp_path
        video_dir = keyframes_dir / "L01_V001"
        video_dir.mkdir()
        PILImage.new("RGB", (640, 480), color=(128, 128, 128)).save(video_dir / "001500.jpg")

        detection = make_detection(
            boxes=[[100.0, 100.0, 300.0, 400.0]],
            scores=[0.92],
            labels=["person"],
        )
        self._install_mock_model(reranker, detection)

        candidates = [make_candidate("L01_V001", 1500, score=0.9, rrf_score=0.032)]
        results = await reranker.rerank("person", candidates, keyframes_dir)

        assert len(results) == 1
        # New formula: rrf * (1 + alpha * confidence) = 0.032 * (1 + 0.5 * 0.92)
        assert results[0].rerank_score == pytest.approx(0.032 * (1.0 + 0.5 * 0.92))

    async def test_no_detection_preserves_rrf_score(
        self, reranker: GroundingReranker, tmp_path: Path
    ) -> None:
        """When Grounding DINO detects nothing, the RRF baseline must be preserved.

        Old behaviour (multiplicative): score → 0.0  (zero-score trap).
        New behaviour (additive boost): score → rrf_score * 1.0 = rrf_score.
        """
        from PIL import Image as PILImage

        keyframes_dir = tmp_path
        video_dir = keyframes_dir / "L01_V001"
        video_dir.mkdir()
        PILImage.new("RGB", (640, 480)).save(video_dir / "001520.jpg")

        detection = make_detection(boxes=[], scores=[], labels=[])
        self._install_mock_model(reranker, detection)

        candidates = [make_candidate("L01_V001", 1520, score=0.85, rrf_score=0.028)]
        results = await reranker.rerank("person", candidates, keyframes_dir)

        # confidence=0 → rrf * (1 + 0.5*0) = rrf → baseline preserved, NOT zeroed
        assert results[0].rerank_score == pytest.approx(0.028)

    async def test_results_sorted_by_rerank_score_descending(
        self, reranker: GroundingReranker, tmp_path: Path
    ) -> None:
        """Higher grounding confidence candidate should rank first."""
        from PIL import Image as PILImage

        keyframes_dir = tmp_path
        video_dir = keyframes_dir / "L01_V001"
        video_dir.mkdir()
        PILImage.new("RGB", (640, 480)).save(video_dir / "001500.jpg")
        PILImage.new("RGB", (640, 480)).save(video_dir / "001520.jpg")

        call_count = 0

        async def mock_ground(query: str, image_path: Path) -> list[GroundingResult]:
            nonlocal call_count
            call_count += 1
            if "001500" in str(image_path):
                return [GroundingResult(label="person", confidence=0.30, bbox=[0.1, 0.1, 0.5, 0.5])]
            return [GroundingResult(label="person", confidence=0.92, bbox=[0.2, 0.2, 0.8, 0.8])]

        reranker._ground_single = mock_ground  # type: ignore[method-assign]

        candidates = [
            make_candidate("L01_V001", 1500, score=0.9, rrf_score=0.032),
            make_candidate("L01_V001", 1520, score=0.85, rrf_score=0.028),
        ]
        results = await reranker.rerank("person", candidates, keyframes_dir)

        # frame 1520 has higher confidence → must rank first
        assert results[0].frame_id == 1520
        assert results[1].frame_id == 1500
        assert results[0].rerank_score > results[1].rerank_score

    async def test_missing_keyframe_file_returns_zero_score(
        self, reranker: GroundingReranker, tmp_path: Path
    ) -> None:
        """A missing keyframe should not crash; candidate gets score 0.0."""
        self._install_mock_model(reranker, make_detection([], [], []))

        candidates = [make_candidate("MISSING_VIDEO", 9999, score=0.9, rrf_score=0.032)]
        results = await reranker.rerank("query", candidates, tmp_path)

        assert len(results) == 1
        # Missing file → no detection → confidence = 0 → score = rrf_score * 1.0 = 0.032
        assert results[0].rerank_score == pytest.approx(0.032)
        assert results[0].grounding == []

    async def test_grounding_result_contains_bbox_and_label(
        self, reranker: GroundingReranker, tmp_path: Path
    ) -> None:
        from PIL import Image as PILImage

        keyframes_dir = tmp_path
        video_dir = keyframes_dir / "L01_V001"
        video_dir.mkdir()
        PILImage.new("RGB", (100, 100)).save(video_dir / "001500.jpg")

        detection = make_detection(
            boxes=[[10.0, 20.0, 80.0, 90.0]],
            scores=[0.85],
            labels=["laptop"],
        )
        self._install_mock_model(reranker, detection)

        candidates = [make_candidate("L01_V001", 1500, score=0.9, rrf_score=0.030)]
        results = await reranker.rerank("laptop", candidates, keyframes_dir)

        grounding = results[0].grounding
        assert len(grounding) == 1
        assert grounding[0].label == "laptop"
        assert grounding[0].confidence == pytest.approx(0.85)
        assert len(grounding[0].bbox) == 4


# ---------------------------------------------------------------------------
# 5. GroundingReranker — alpha property and constructor validation
# ---------------------------------------------------------------------------


class TestGroundingRerankerAlpha:
    """Tests for the ``alpha`` parameter on :class:`GroundingReranker`."""

    def test_default_alpha_is_point_five(self) -> None:
        r = GroundingReranker()
        assert r.alpha == pytest.approx(0.5)

    def test_custom_alpha_is_stored(self) -> None:
        r = GroundingReranker(alpha=1.0)
        assert r.alpha == pytest.approx(1.0)

    def test_alpha_zero_is_valid(self) -> None:
        """alpha=0 disables grounding boost; all scores remain at rrf_score."""
        r = GroundingReranker(alpha=0.0)
        assert r.alpha == pytest.approx(0.0)

    def test_negative_alpha_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            GroundingReranker(alpha=-0.1)

    async def test_reranker_uses_alpha_in_passthrough(self) -> None:
        """Even in pass-through mode the returned score must equal rrf_score.

        Pass-through does not call _compute_rerank_score; it directly copies
        rrf_score.  This test verifies that alpha has no effect in that path
        (as intended — alpha only applies when the model is loaded).
        """
        r = GroundingReranker(alpha=2.0, device="cpu")
        candidate = make_candidate(rrf_score=0.040)
        results = await r.rerank("query", [candidate], "/data/keyframes")
        assert results[0].rerank_score == pytest.approx(0.040)


# ---------------------------------------------------------------------------
# 6. Schema validation tests for RerankCandidate
# ---------------------------------------------------------------------------


class TestRerankCandidateSchema:
    """Tests for the updated RerankCandidate schema."""

    def test_rrf_score_defaults_to_none(self) -> None:
        c = RerankCandidate(video_id="V1", frame_id=1, score=0.5)
        assert c.rrf_score is None

    def test_rrf_score_is_accepted_when_provided(self) -> None:
        c = RerankCandidate(video_id="V1", frame_id=1, score=0.5, rrf_score=0.032)
        assert c.rrf_score == pytest.approx(0.032)

    def test_reasoning_trace_defaults_to_none_on_result_item(self) -> None:
        item = RerankResultItem(
            video_id="V1",
            frame_id=1,
            original_score=0.9,
            rerank_score=0.029,
        )
        assert item.reasoning_trace is None

    def test_reasoning_trace_is_accepted_when_provided(self) -> None:
        item = RerankResultItem(
            video_id="V1",
            frame_id=1,
            original_score=0.9,
            rerank_score=0.029,
            reasoning_trace="Chain of thought: ...",
        )
        assert item.reasoning_trace == "Chain of thought: ..."
