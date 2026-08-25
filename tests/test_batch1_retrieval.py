from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from backend.services.batch1_retrieval import (
    DenseRetrievalService,
    OfficialClipShardIndex,
    SearchCandidate,
    fuse_variant_rankings,
)


class FakeEncoder:
    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1
        return vector


def make_data(root: Path) -> OfficialClipShardIndex:
    features = root / "features"
    mappings = root / "mappings"
    keyframes = root / "keyframes" / "L21_V001"
    features.mkdir()
    mappings.mkdir()
    keyframes.mkdir(parents=True)
    matrix = np.zeros((3, 512), dtype=np.float16)
    matrix[0, :2] = (0.5, 0.866)
    matrix[1, 0] = 1
    matrix[2, :2] = (0.8, 0.6)
    np.save(features / "L21_V001.npy", matrix)
    with (mappings / "L21_V001.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerows(
            [
                {"n": 1, "pts_time": 0, "fps": 30, "frame_idx": 0},
                {"n": 2, "pts_time": 0.03, "fps": 30, "frame_idx": 0},
                {"n": 3, "pts_time": 3, "fps": 30, "frame_idx": 90},
            ]
        )
    for name in ("001.jpg", "002.jpg", "003.jpg"):
        (keyframes / name).write_bytes(b"jpg")
    return OfficialClipShardIndex(features, mappings, root / "keyframes")


def test_search_maps_rows_and_deduplicates_before_top_k(tmp_path: Path) -> None:
    index = make_data(tmp_path)
    query = np.zeros(512, dtype=np.float32)
    query[0] = 2
    original = query.copy()
    results = index.search(query, top_k=2)
    assert query.tolist() == original.tolist()
    assert [(item.frame_id, item.keyframe_id, item.feature_row) for item in results] == [
        (0, "002", 1),
        (90, "003", 2),
    ]
    assert results[0].image_path == tmp_path / "keyframes" / "L21_V001" / "002.jpg"
    assert [item.rank for item in results] == [1, 2]


@pytest.mark.parametrize("top_k", [True, 0, 101, "2"])
def test_rejects_invalid_top_k(tmp_path: Path, top_k: object) -> None:
    index = make_data(tmp_path)
    query = np.ones(512, dtype=np.float32)
    with pytest.raises(ValueError, match="top_k"):
        index.search(query, top_k=top_k)  # type: ignore[arg-type]


def test_dense_service_rejects_blank_and_does_not_pad(tmp_path: Path) -> None:
    service = DenseRetrievalService(make_data(tmp_path), FakeEncoder())
    with pytest.raises(ValueError, match="blank"):
        service.search(" ")
    assert len(service.search("người đang nói", top_k=100)) == 2


def test_index_rejects_mapping_count_mismatch(tmp_path: Path) -> None:
    index = make_data(tmp_path)
    mapping = tmp_path / "mappings" / "L21_V001.csv"
    mapping.write_text("n,pts_time,fps,frame_idx\n1,0,30,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="3 feature rows but 1 mapping rows"):
        OfficialClipShardIndex(tmp_path / "features", tmp_path / "mappings", tmp_path / "keyframes")
    assert index.record_count == 3


def candidate(rank: int, video_id: str, frame_id: int, score: float) -> SearchCandidate:
    return SearchCandidate(rank, video_id, frame_id, f"{frame_id:03d}", score, Path(f"{frame_id}.jpg"), frame_id)


def test_variant_rrf_manual_values_presence_and_duplicate_identity() -> None:
    first = [candidate(1, "A", 1, 0.9), candidate(2, "B", 2, 0.8), candidate(3, "A", 1, 0.7)]
    second = [candidate(1, "B", 2, 0.1)]
    original = [[item for item in ranking] for ranking in (first, second)]
    results = fuse_variant_rankings([first, second], top_k=20)
    assert [(item.video_id, item.frame_id) for item in results] == [("B", 2), ("A", 1)]
    assert results[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].rrf_score == pytest.approx(1 / 61)
    assert [(match.variant_index, match.rank) for match in results[0].source_matches] == [(0, 2), (1, 1)]
    assert [[item for item in ranking] for ranking in (first, second)] == original


def test_variant_rrf_deterministic_tie_break_and_top_k() -> None:
    ranking = [candidate(1, "B", 3, 0.9), candidate(1, "A", 2, 0.8), candidate(1, "A", 1, 0.7)]
    results = fuse_variant_rankings([ranking], top_k=2)
    assert [(item.video_id, item.frame_id) for item in results] == [("A", 1), ("A", 2)]
    assert [item.rank for item in results] == [1, 2]
