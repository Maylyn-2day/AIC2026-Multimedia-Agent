"""Unit tests for v2_vlm_reranker and v2_pipeline (weight-free)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.v2_vlm_reranker import (
    M2ScoredCandidate,
    parse_vlm_json_response,
)
from scripts.v2_pipeline import FusedResult, _minmax_normalize, stage3_fuse


# ---------------------------------------------------------------------------
# parse_vlm_json_response tests (Fix 2 validation)
# ---------------------------------------------------------------------------


class TestParseVlmJsonResponse:
    """Verify multi-level JSON fallback parsing."""

    def test_valid_json(self) -> None:
        raw = '{"relevance_score": 0.85, "confidence": 4, "answer": "red car", "reasoning": "visible"}'
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == pytest.approx(0.85)
        assert result["confidence"] == 4
        assert result["answer"] == "red car"
        assert result["reasoning"] == "visible"

    def test_json_in_code_fence(self) -> None:
        raw = '```json\n{"relevance_score": 0.7, "confidence": 3, "answer": "blue", "reasoning": "ok"}\n```'
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == pytest.approx(0.7)
        assert result["answer"] == "blue"

    def test_json_embedded_in_text(self) -> None:
        raw = 'Here is the answer: {"relevance_score": 0.6, "confidence": 2, "answer": "tree", "reasoning": "green"} done.'
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == pytest.approx(0.6)

    def test_key_value_fallback(self) -> None:
        raw = "RELEVANT: YES\nSCORE: 0.9\nANSWER: dog\nREASONING: clear"
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == pytest.approx(0.9)
        assert result["answer"] == "dog"

    def test_not_relevant_penalty(self) -> None:
        raw = "RELEVANT: NO\nSCORE: 0.5"
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == pytest.approx(0.2)

    def test_empty_input(self) -> None:
        result = parse_vlm_json_response("")
        assert result["relevance_score"] == 0.5
        assert result["confidence"] == 3
        assert result["answer"] == ""

    def test_garbage_input(self) -> None:
        result = parse_vlm_json_response("asdf jkl; qwerty 12345 !@#$%")
        assert result["relevance_score"] == 0.5  # default
        assert isinstance(result["reasoning"], str)

    def test_score_clamped_to_unit(self) -> None:
        raw = '{"relevance_score": 5.0, "confidence": 10}'
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == 1.0
        assert result["confidence"] == 5

    def test_score_clamped_to_zero(self) -> None:
        raw = '{"relevance_score": -2.0, "confidence": -1}'
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == 0.0
        assert result["confidence"] == 1

    def test_partial_json(self) -> None:
        raw = '{"relevance_score": 0.4}'
        result = parse_vlm_json_response(raw)
        assert result["relevance_score"] == pytest.approx(0.4)
        assert result["confidence"] == 3  # default preserved


# ---------------------------------------------------------------------------
# Min-Max Normalization tests (Fix 1 validation)
# ---------------------------------------------------------------------------


class TestMinMaxNormalize:
    """Verify score normalization edge cases."""

    def test_basic(self) -> None:
        result = _minmax_normalize([0.01, 0.02, 0.03])
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_already_unit(self) -> None:
        result = _minmax_normalize([0.0, 0.5, 1.0])
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_identical_scores(self) -> None:
        """All scores equal → 0.5 midpoint (no division by zero)."""
        result = _minmax_normalize([0.5, 0.5, 0.5])
        assert result == [0.5, 0.5, 0.5]

    def test_single_value(self) -> None:
        result = _minmax_normalize([0.42])
        assert result == [0.5]

    def test_empty(self) -> None:
        assert _minmax_normalize([]) == []

    def test_negative_values(self) -> None:
        result = _minmax_normalize([-1.0, 0.0, 1.0])
        assert result == pytest.approx([0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# Stage 3 fusion tests (Fix 1: scale normalization)
# ---------------------------------------------------------------------------


class TestStage3Fuse:
    """Verify normalised late fusion prevents RRF score override."""

    @staticmethod
    def _make_scored(
        rank: int, retrieval: float, vlm: float, vid: str = "V01", fid: int = 100
    ) -> M2ScoredCandidate:
        return M2ScoredCandidate(
            rank=rank,
            video_id=vid,
            frame_id=fid,
            keyframe_id="001",
            image_path=Path("/fake/path.jpg"),
            retrieval_score=retrieval,
            vlm_relevance=vlm,
            vlm_confidence=3,
            vlm_answer="",
            vlm_reasoning="",
        )

    def test_fusion_respects_both_signals(self) -> None:
        """High VLM score on a low-retrieval candidate should boost it."""
        candidates = [
            self._make_scored(1, retrieval=0.03, vlm=0.2, vid="V01", fid=10),
            self._make_scored(2, retrieval=0.01, vlm=0.9, vid="V02", fid=20),
        ]
        fused = stage3_fuse(candidates, alpha=0.6)
        # V02 has lower retrieval but much higher VLM → should rank first
        assert fused[0].video_id == "V02"
        assert fused[1].video_id == "V01"

    def test_alpha_zero_ignores_vlm(self) -> None:
        candidates = [
            self._make_scored(1, retrieval=0.03, vlm=0.9, vid="V01", fid=10),
            self._make_scored(2, retrieval=0.01, vlm=0.1, vid="V02", fid=20),
        ]
        fused = stage3_fuse(candidates, alpha=0.0)
        assert fused[0].video_id == "V01"

    def test_alpha_one_uses_only_vlm(self) -> None:
        candidates = [
            self._make_scored(1, retrieval=0.03, vlm=0.1, vid="V01", fid=10),
            self._make_scored(2, retrieval=0.01, vlm=0.9, vid="V02", fid=20),
        ]
        fused = stage3_fuse(candidates, alpha=1.0)
        assert fused[0].video_id == "V02"

    def test_ranks_sequential(self) -> None:
        candidates = [self._make_scored(i, 0.01 * i, 0.5, fid=i * 10) for i in range(1, 6)]
        fused = stage3_fuse(candidates, alpha=0.5)
        assert [r.rank for r in fused] == [1, 2, 3, 4, 5]

    def test_empty_input(self) -> None:
        assert stage3_fuse([], alpha=0.5) == []

    def test_identical_vlm_preserves_retrieval_order(self) -> None:
        """When VLM scores are equal, retrieval ranking should dominate."""
        candidates = [
            self._make_scored(1, retrieval=0.03, vlm=0.5, vid="V01", fid=10),
            self._make_scored(2, retrieval=0.01, vlm=0.5, vid="V02", fid=20),
        ]
        fused = stage3_fuse(candidates, alpha=0.5)
        assert fused[0].video_id == "V01"
