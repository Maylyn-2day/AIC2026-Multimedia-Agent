"""Unit tests for v2_pipeline submission packager & validator — weight-free."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest
from scripts.v2_pipeline import (
    FusedResult,
    _minmax_normalize,
    _package_submission,
    _validate_submission_zip,
    _write_kis_submission,
    _write_qa_submission,
    _write_trake_submission,
)

from backend.services.v2_trake_engine import TrakeEventFrame, TrakeSequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fused(rank: int, vid: str, fid: int, score: float, answer: str = "") -> FusedResult:
    return FusedResult(
        rank=rank, video_id=vid, frame_id=fid, keyframe_id=f"{fid:03d}",
        image_path=f"/fake/{vid}/{fid:03d}.jpg",
        final_score=score, retrieval_score_raw=score, retrieval_score_norm=score,
        vlm_relevance_raw=0.5, vlm_relevance_norm=0.5, vlm_confidence=3,
        vlm_answer=answer, vlm_reasoning="",
    )


def _trake_sequence(vid: str, frames: list[int]) -> TrakeSequence:
    events = tuple(
        TrakeEventFrame(i, vid, f, f"{f:03d}", Path(f"/fake/{vid}/{f:03d}.jpg"), float(i * 2), 0.9, i + 1)
        for i, f in enumerate(frames)
    )
    return TrakeSequence(rank=1, video_id=vid, joint_score=2.5, timestamp_variance=1.0, events=events)


# ---------------------------------------------------------------------------
# KIS submission writer
# ---------------------------------------------------------------------------

class TestWriteKisSubmission:
    def test_writes_two_columns(self, tmp_path: Path) -> None:
        results = [_fused(i, "V01", i * 10, 1.0 - i * 0.01) for i in range(1, 5)]
        path = tmp_path / "kis.csv"
        _write_kis_submission(results, path)
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 4
        assert all(len(r) == 2 for r in rows)

    def test_capped_at_100_rows(self, tmp_path: Path) -> None:
        results = [_fused(i, "V01", i * 10, 1.0) for i in range(1, 200)]
        path = tmp_path / "kis.csv"
        _write_kis_submission(results, path)
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 100

    def test_no_header(self, tmp_path: Path) -> None:
        results = [_fused(1, "V01", 10, 0.9)]
        path = tmp_path / "kis.csv"
        _write_kis_submission(results, path)
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first.split(",")[0] != "video_id"

    def test_frame_ids_are_integers(self, tmp_path: Path) -> None:
        results = [_fused(1, "V01", 12345, 0.9)]
        path = tmp_path / "kis.csv"
        _write_kis_submission(results, path)
        rows = list(csv.reader(path.read_text().splitlines()))
        assert int(rows[0][1]) == 12345


# ---------------------------------------------------------------------------
# QA submission writer
# ---------------------------------------------------------------------------

class TestWriteQaSubmission:
    def test_single_row_three_cols(self, tmp_path: Path) -> None:
        results = [_fused(1, "V01", 100, 0.9, answer="red car")]
        path = tmp_path / "qa.csv"
        _write_qa_submission(results, path)
        rows = list(csv.reader(path.read_text().splitlines()))
        assert len(rows) == 1 and len(rows[0]) == 3

    def test_answer_written_correctly(self, tmp_path: Path) -> None:
        results = [_fused(1, "V01", 100, 0.9, answer="blue sky")]
        path = tmp_path / "qa.csv"
        _write_qa_submission(results, path)
        rows = list(csv.reader(path.read_text().splitlines()))
        assert rows[0][2] == "blue sky"

    def test_blank_answer_raises(self, tmp_path: Path) -> None:
        results = [_fused(1, "V01", 100, 0.9, answer="")]
        with pytest.raises(ValueError, match="blank"):
            _write_qa_submission(results, tmp_path / "qa.csv")

    def test_answer_truncated_to_100_chars(self, tmp_path: Path) -> None:
        long = "x" * 200
        results = [_fused(1, "V01", 100, 0.9, answer=long)]
        path = tmp_path / "qa.csv"
        _write_qa_submission(results, path)
        rows = list(csv.reader(path.read_text().splitlines()))
        assert len(rows[0][2]) == 100

    def test_empty_results_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _write_qa_submission([], tmp_path / "qa.csv")


# ---------------------------------------------------------------------------
# TRAKE submission writer
# ---------------------------------------------------------------------------

class TestWriteTrakeSubmission:
    def test_single_row_with_frames(self, tmp_path: Path) -> None:
        seq = _trake_sequence("V01", [10, 20, 30])
        path = tmp_path / "trake.csv"
        _write_trake_submission([seq], path, n_events=3)
        rows = list(csv.reader(path.read_text().splitlines()))
        assert len(rows) == 1
        assert rows[0][0] == "V01"
        assert [int(x) for x in rows[0][1:]] == [10, 20, 30]

    def test_no_header(self, tmp_path: Path) -> None:
        seq = _trake_sequence("V01", [10, 20])
        path = tmp_path / "trake.csv"
        _write_trake_submission([seq], path, n_events=2)
        assert path.read_text().splitlines()[0].split(",")[0] != "video_id"

    def test_empty_sequences_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _write_trake_submission([], tmp_path / "trake.csv", n_events=2)

    def test_frame_count_mismatch_raises(self, tmp_path: Path) -> None:
        seq = _trake_sequence("V01", [10, 20])  # 2 frames
        with pytest.raises(ValueError, match="mismatch"):
            _write_trake_submission([seq], tmp_path / "trake.csv", n_events=3)  # expects 3


# ---------------------------------------------------------------------------
# Package & validate submission ZIP
# ---------------------------------------------------------------------------

class TestSubmissionPackager:
    def _make_submission_dir(self, tmp_path: Path, specs: list[tuple[str, list[list[str]]]]) -> Path:
        d = tmp_path / "submission"
        d.mkdir()
        for name, rows in specs:
            with (d / name).open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(rows)
        return d

    def test_zip_contains_submission_prefix(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-kis.csv", [["V01", "10"]] * 5),
        ])
        final_zip = tmp_path / "submission.zip"
        _package_submission(sub_dir, final_zip)
        with zipfile.ZipFile(final_zip) as z:
            assert all(n.startswith("submission/") for n in z.namelist())

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            _package_submission(empty, tmp_path / "out.zip")

    def test_validate_kis_passes(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-kis.csv", [[f"V{i:02d}", str(i * 10)] for i in range(1, 11)]),
        ])
        final_zip = tmp_path / "sub.zip"
        _package_submission(sub_dir, final_zip)
        result = _validate_submission_zip(final_zip)
        assert result["valid"] is True
        assert result["counts"]["kis"] == 1

    def test_validate_qa_passes(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-qa.csv", [["V01", "100", "red car"]]),
        ])
        final_zip = tmp_path / "sub.zip"
        _package_submission(sub_dir, final_zip)
        result = _validate_submission_zip(final_zip)
        assert result["valid"] is True

    def test_validate_trake_passes(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-trake.csv", [["V01", "10", "20", "30"]]),
        ])
        final_zip = tmp_path / "sub.zip"
        _package_submission(sub_dir, final_zip)
        result = _validate_submission_zip(final_zip)
        assert result["valid"] is True

    def test_validate_trake_non_monotonic_fails(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-trake.csv", [["V01", "30", "20", "40"]]),  # 30 > 20 invalid
        ])
        final_zip = tmp_path / "sub.zip"
        _package_submission(sub_dir, final_zip)
        result = _validate_submission_zip(final_zip)
        assert result["valid"] is False
        assert any("strictly increasing" in e for e in result["errors"])

    def test_validate_qa_blank_answer_fails(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-qa.csv", [["V01", "100", "  "]]),
        ])
        final_zip = tmp_path / "sub.zip"
        _package_submission(sub_dir, final_zip)
        result = _validate_submission_zip(final_zip)
        assert result["valid"] is False

    def test_validate_header_in_csv_fails(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-kis.csv", [["video_id", "frame_id"], ["V01", "10"]]),
        ])
        final_zip = tmp_path / "sub.zip"
        _package_submission(sub_dir, final_zip)
        result = _validate_submission_zip(final_zip)
        assert result["valid"] is False
        assert any("header" in e for e in result["errors"])

    def test_validate_unknown_task_fails(self, tmp_path: Path) -> None:
        sub_dir = self._make_submission_dir(tmp_path, [
            ("q01-unknown.csv", [["V01", "10"]]),
        ])
        final_zip = tmp_path / "sub.zip"
        _package_submission(sub_dir, final_zip)
        result = _validate_submission_zip(final_zip)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Stage 3 fusion (regression from original test_v2_pipeline.py)
# ---------------------------------------------------------------------------

class TestMinMaxNormalize:
    def test_identical_returns_midpoint(self) -> None:
        assert _minmax_normalize([0.5, 0.5, 0.5]) == [0.5, 0.5, 0.5]

    def test_empty(self) -> None:
        assert _minmax_normalize([]) == []

    def test_basic(self) -> None:
        result = _minmax_normalize([0.0, 0.5, 1.0])
        assert result == pytest.approx([0.0, 0.5, 1.0])
