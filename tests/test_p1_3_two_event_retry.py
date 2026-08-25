import json
from pathlib import Path

import pytest
from scripts.p1_3_two_event_retry import EventCandidate, fuse_event_rankings, pair_ordered_events, read_event_variants

from backend.services.batch1_retrieval import SearchCandidate


def event(video: str, frame: int, row: int, score: float = 0.1) -> EventCandidate:
    return EventCandidate(video, frame, f"{row:03d}", Path(f"C:/{video}/{row:03d}.jpg"), row, score, (1,))


def test_pairs_require_same_video_and_a_before_b_without_mutation() -> None:
    left = [event("V1", 10, 0), event("V2", 30, 0)]
    right = [event("V1", 20, 1), event("V2", 20, 1), event("V3", 40, 0)]
    snapshot = (list(left), list(right))
    pairs = pair_ordered_events(left, right, {("V1", 0): 1.0, ("V1", 1): 2.0})
    assert len(pairs) == 1 and pairs[0].video_id == "V1"
    assert pairs[0].event_a.frame_id < pairs[0].event_b.frame_id
    assert pairs[0].event_a.image_path == Path("C:/V1/000.jpg")
    assert (left, right) == snapshot


def test_pair_order_is_deterministic() -> None:
    left = [event("V2", 1, 0), event("V1", 1, 0)]
    right = [event("V2", 2, 1), event("V1", 2, 1)]
    timestamps = {(video, row): float(row + 1) for video in ("V1", "V2") for row in (0, 1)}
    assert [pair.video_id for pair in pair_ordered_events(left, right, timestamps)] == ["V1", "V2"]


def test_event_fusion_is_rank_only_and_keeps_explicit_path() -> None:
    image = Path("C:/frames/007.jpg")
    source = [[SearchCandidate(1, "V", 50, "007", 0.9, image, 6)]]
    snapshot = list(source[0])
    fused = fuse_event_rankings(source)
    assert fused[0].rrf_score == 1 / 61
    assert fused[0].image_path == image
    assert source[0] == snapshot


def test_event_variants_are_external_strict_utf8(tmp_path: Path) -> None:
    path = tmp_path / "variants.json"
    path.write_text(json.dumps([" scene one ", "scene two"]), encoding="utf-8")
    assert read_event_variants(path) == ("scene one", "scene two")
    path.write_text(json.dumps(["same", " same "]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        read_event_variants(path)
