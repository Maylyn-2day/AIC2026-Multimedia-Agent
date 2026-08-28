"""Unit tests for v2_trake_engine — weight-free, no model downloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.services.v2_trake_engine import (
    TRAKEEngine,
    TrakeSequence,
    _variance,
    format_trake_submission_row,
    sequence_to_dict,
)

# ---------------------------------------------------------------------------
# Synthetic candidate factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FakeCandidate:
    """Minimal duck-type of SearchCandidate for TRAKE tests."""
    rank: int
    video_id: str
    frame_id: int
    keyframe_id: str
    image_path: Path
    score: float
    pts_time: float = 0.0
    feature_row: int = 0


def _c(rank: int, vid: str, frame: int, score: float, pts: float = 0.0) -> _FakeCandidate:
    return _FakeCandidate(
        rank=rank, video_id=vid, frame_id=frame,
        keyframe_id=f"{frame:03d}", image_path=Path(f"/fake/{vid}/{frame:03d}.jpg"),
        score=score, pts_time=pts,
    )


ENGINE = TRAKEEngine(min_gap_seconds=1.0, lambda_penalty=0.005, beam_width=200, pool_top_k=50)


# ---------------------------------------------------------------------------
# 2-event tests
# ---------------------------------------------------------------------------

class TestTwoEventAssembly:
    def test_basic_valid_sequence(self) -> None:
        e1 = [_c(1, "V01", 10, 0.9, pts=0.0), _c(2, "V01", 20, 0.8, pts=1.0)]
        e2 = [_c(1, "V01", 30, 0.7, pts=2.0), _c(2, "V01", 40, 0.6, pts=3.0)]
        seqs = ENGINE.assemble([e1, e2], top_k=5)
        assert seqs, "Expected at least one sequence"
        assert seqs[0].n_events == 2
        f1, f2 = seqs[0].frame_ids
        assert f1 < f2, "frames must be strictly increasing"
        assert seqs[0].video_id == "V01"

    def test_monotonic_constraint_enforced(self) -> None:
        """Frame ordering E1=30 > E2=20 must be rejected."""
        e1 = [_c(1, "V01", 30, 0.9)]
        e2 = [_c(1, "V01", 20, 0.9)]  # frame_id < e1 → invalid
        seqs = ENGINE.assemble([e1, e2], top_k=5)
        assert seqs == [], "Expected no valid sequences when ordering is violated"

    def test_cross_video_rejected(self) -> None:
        """Sequences mixing video IDs are never produced."""
        e1 = [_c(1, "V01", 10, 0.9)]
        e2 = [_c(1, "V02", 20, 0.9)]  # different video
        seqs = ENGINE.assemble([e1, e2])
        assert seqs == [], "Cross-video sequences must be rejected"

    def test_top_k_limits_results(self) -> None:
        e1 = [_c(i, "V01", i * 10, 0.9 - i * 0.01, pts=float(i)) for i in range(1, 6)]
        e2 = [_c(i, "V01", i * 10 + 5, 0.8 - i * 0.01, pts=float(i) + 1.0) for i in range(1, 6)]
        seqs = ENGINE.assemble([e1, e2], top_k=3)
        assert len(seqs) <= 3

    def test_ranks_are_sequential(self) -> None:
        e1 = [_c(1, "V01", 10, 0.9)]
        e2 = [_c(1, "V01", 20, 0.8), _c(2, "V01", 30, 0.7)]
        seqs = ENGINE.assemble([e1, e2], top_k=10)
        assert [s.rank for s in seqs] == list(range(1, len(seqs) + 1))

    def test_deterministic_ordering(self) -> None:
        """Same input always produces same output ordering."""
        e1 = [_c(1, "V01", 10, 0.9), _c(1, "V02", 10, 0.9)]
        e2 = [_c(1, "V01", 20, 0.8), _c(1, "V02", 20, 0.8)]
        seqs1 = ENGINE.assemble([e1, e2])
        seqs2 = ENGINE.assemble([e1, e2])
        assert [(s.video_id, s.frame_ids) for s in seqs1] == [(s.video_id, s.frame_ids) for s in seqs2]


# ---------------------------------------------------------------------------
# 3-event tests
# ---------------------------------------------------------------------------

class TestThreeEventAssembly:
    def test_three_events_valid(self) -> None:
        e1 = [_c(1, "V01", 10, 0.9, pts=0.0)]
        e2 = [_c(1, "V01", 20, 0.8, pts=2.0)]
        e3 = [_c(1, "V01", 30, 0.7, pts=4.0)]
        seqs = ENGINE.assemble([e1, e2, e3])
        assert seqs and seqs[0].n_events == 3
        frames = seqs[0].frame_ids
        assert all(a < b for a, b in zip(frames, frames[1:], strict=False))

    def test_three_events_no_valid_sequence(self) -> None:
        """All candidates in same video but frames not satisfiable in order."""
        e1 = [_c(1, "V01", 100, 0.9)]
        e2 = [_c(1, "V01", 50, 0.9)]   # < e1
        e3 = [_c(1, "V01", 200, 0.9)]
        seqs = ENGINE.assemble([e1, e2, e3])
        # e2 frame is less than e1 — beam should collapse
        assert seqs == []

    def test_equal_frame_id_rejected(self) -> None:
        """frame_1 == frame_2 violates strict monotonic ordering."""
        e1 = [_c(1, "V01", 10, 0.9)]
        e2 = [_c(1, "V01", 10, 0.9)]  # same frame
        e3 = [_c(1, "V01", 20, 0.9)]
        seqs = ENGINE.assemble([e1, e2, e3])
        assert seqs == []


# ---------------------------------------------------------------------------
# 4-event tests
# ---------------------------------------------------------------------------

class TestFourEventAssembly:
    def test_four_events_valid(self) -> None:
        events = [
            [_c(1, "V01", 10 * (i + 1), 0.9 - i * 0.05, pts=float(i * 2))]
            for i in range(4)
        ]
        seqs = ENGINE.assemble(events)
        assert seqs and seqs[0].n_events == 4
        frames = seqs[0].frame_ids
        assert all(a < b for a, b in zip(frames, frames[1:], strict=False))

    def test_four_events_multi_video_prefers_best_score(self) -> None:
        """Both V01 and V02 have valid sequences; engine picks highest joint score."""
        events_v01 = [[_c(1, "V01", 10 * (i + 1), 0.95, pts=float(i * 2))] for i in range(4)]
        events_v02 = [[_c(2, "V02", 10 * (i + 1), 0.50, pts=float(i * 2))] for i in range(4)]
        # Merge per-event lists
        merged = [a + b for a, b in zip(events_v01, events_v02, strict=True)]
        seqs = ENGINE.assemble(merged, top_k=5)
        assert seqs[0].video_id == "V01"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_event_raises(self) -> None:
        with pytest.raises(ValueError, match="candidate list is empty"):
            ENGINE.assemble([[_c(1, "V01", 10, 0.9)], []])

    def test_no_events_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ENGINE.assemble([])

    def test_single_event(self) -> None:
        """Single event: no temporal constraint, all candidates become sequences."""
        candidates = [_c(i, "V01", i * 10, 0.9 - i * 0.1) for i in range(1, 4)]
        seqs = ENGINE.assemble([candidates], top_k=10)
        assert len(seqs) == 3
        assert all(s.n_events == 1 for s in seqs)

    def test_timestamp_variance_tie_breaking(self) -> None:
        """When joint scores are equal, smaller variance wins."""
        # Two sequences with same score but different timestamp spread
        e1 = [_c(1, "V01", 10, 0.5, pts=0.0), _c(1, "V02", 10, 0.5, pts=0.0)]
        e2 = [_c(1, "V01", 20, 0.5, pts=100.0), _c(1, "V02", 20, 0.5, pts=5.0)]
        seqs = ENGINE.assemble([e1, e2], top_k=10)
        # V02 has smaller timestamp variance (5-0=5 vs 100-0=100)
        assert seqs[0].video_id == "V02"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_variance_empty(self) -> None:
        assert _variance([]) == 0.0

    def test_variance_single(self) -> None:
        assert _variance([42.0]) == 0.0

    def test_variance_known(self) -> None:
        assert _variance([0.0, 10.0]) == pytest.approx(25.0)

    def test_format_submission_row(self) -> None:
        from backend.services.v2_trake_engine import TrakeEventFrame
        ef1 = TrakeEventFrame(0, "V01", 100, "100", Path("/fake"), 0.0, 0.9, 1)
        ef2 = TrakeEventFrame(1, "V01", 200, "200", Path("/fake"), 2.0, 0.8, 2)
        seq = TrakeSequence(rank=1, video_id="V01", joint_score=1.7,
                            timestamp_variance=1.0, events=(ef1, ef2))
        row = format_trake_submission_row(seq)
        assert row == ["V01", 100, 200]

    def test_sequence_to_dict(self) -> None:
        from backend.services.v2_trake_engine import TrakeEventFrame
        ef = TrakeEventFrame(0, "V01", 10, "010", Path("/fake"), 0.0, 0.9, 1)
        seq = TrakeSequence(rank=1, video_id="V01", joint_score=0.9,
                            timestamp_variance=0.0, events=(ef,))
        d = sequence_to_dict(seq)
        assert d["video_id"] == "V01"
        assert d["frame_ids"] == [10]
        assert len(d["events"]) == 1
