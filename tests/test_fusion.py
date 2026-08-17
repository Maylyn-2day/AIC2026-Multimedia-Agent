"""Tests for deterministic weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from copy import deepcopy

import pytest

from backend.services.fusion import weighted_rrf


def candidate(video_id: str, frame_id: int, rank: int, score: float, **metadata: object) -> dict[str, object]:
    return {"video_id": video_id, "frame_id": frame_id, "rank": rank, "score": score, **metadata}


def test_candidate_in_both_sources_preserves_breakdown_and_metadata() -> None:
    dense = [candidate("V1", 10, 1, 0.8, keyframe_path="frames/a.jpg", shared="dense")]
    sparse = [candidate("V1", 10, 3, 14.2, ocr_text="AIC", shared="sparse")]

    result = weighted_rrf(dense, sparse)

    assert result[0] == {
        "video_id": "V1",
        "frame_id": 10,
        "keyframe_path": "frames/a.jpg",
        "shared": "dense",
        "ocr_text": "AIC",
        "dense_score": 0.8,
        "dense_rank": 1,
        "sparse_score": 14.2,
        "sparse_rank": 3,
        "rrf_score": pytest.approx(1 / 61 + 1 / 63),
        "rrf_k": 60,
        "rank": 1,
    }


@pytest.mark.parametrize(
    ("dense", "sparse", "expected_dense_rank", "expected_sparse_rank"),
    [
        ([candidate("V1", 1, 2, 0.7)], [], 2, None),
        ([], [candidate("V1", 1, 4, 7.0)], None, 4),
    ],
)
def test_candidate_in_only_one_source_is_kept(
    dense: list[dict[str, object]],
    sparse: list[dict[str, object]],
    expected_dense_rank: int | None,
    expected_sparse_rank: int | None,
) -> None:
    result = weighted_rrf(dense, sparse)
    assert len(result) == 1
    assert result[0]["dense_rank"] == expected_dense_rank
    assert result[0]["sparse_rank"] == expected_sparse_rank


def test_weighted_rrf_uses_weights_without_adding_source_scores() -> None:
    result = weighted_rrf(
        [candidate("V1", 1, 2, 999.0)],
        [candidate("V1", 1, 5, 999.0)],
        k=10,
        weight_dense=2.0,
        weight_sparse=0.5,
    )
    assert result[0]["rrf_score"] == pytest.approx(2 / 12 + 0.5 / 15)


def test_mock_hand_calculated_values_and_output_ranking() -> None:
    dense = [candidate("A", 1, 1, 0.8), candidate("B", 2, 2, 0.7), candidate("C", 3, 3, 0.6)]
    sparse = [candidate("A", 1, 3, 10.0), candidate("B", 2, 1, 11.0), candidate("C", 3, 12, 5.0)]

    result = weighted_rrf(dense, sparse)
    by_identity = {(item["video_id"], item["frame_id"]): item for item in result}

    assert by_identity[("A", 1)]["rrf_score"] == pytest.approx(0.032266, abs=0.000001)
    assert by_identity[("B", 2)]["rrf_score"] == pytest.approx(0.032522, abs=0.000001)
    assert by_identity[("C", 3)]["rrf_score"] == pytest.approx(0.029762, abs=0.000001)
    assert [(item["video_id"], item["frame_id"]) for item in result] == [("B", 2), ("A", 1), ("C", 3)]
    assert [item["rank"] for item in result] == [1, 2, 3]


@pytest.mark.parametrize("top_k", [1, 100])
def test_top_k_boundaries(top_k: int) -> None:
    dense = [candidate("V", index, index, 1.0) for index in range(1, 102)]
    result = weighted_rrf(dense, [], top_k=top_k)
    assert len(result) == top_k


@pytest.mark.parametrize("top_k", [0, 101])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        weighted_rrf([], [], top_k=top_k)


def test_fewer_candidates_than_top_k_are_not_duplicated() -> None:
    result = weighted_rrf([candidate("V", 1, 1, 0.9)], [candidate("V", 2, 1, 8.0)], top_k=100)
    identities = [(item["video_id"], item["frame_id"]) for item in result]
    assert identities == [("V", 1), ("V", 2)]
    assert len(identities) == len(set(identities)) == 2


def test_empty_inputs_return_empty_output() -> None:
    assert weighted_rrf([], []) == []


def test_missing_optional_metadata_is_accepted_without_frame_inference() -> None:
    result = weighted_rrf([{"video_id": "V", "frame_id": 42, "rank": 1}], [])
    assert result[0]["frame_id"] == 42
    assert result[0]["dense_score"] is None
    assert "objects" not in result[0]
    assert "ocr_text" not in result[0]


def test_inputs_are_not_mutated_or_exposed_by_reference() -> None:
    dense = [candidate("V", 1, 1, 0.9, metadata={"tags": ["original"]})]
    sparse = [candidate("V", 1, 2, 4.0, sparse_metadata={"tokens": ["original"]})]
    before_dense, before_sparse = deepcopy(dense), deepcopy(sparse)

    result = weighted_rrf(dense, sparse)
    result[0]["metadata"]["tags"].append("changed")  # type: ignore[index,union-attr]
    result[0]["sparse_metadata"]["tokens"].append("changed")  # type: ignore[index,union-attr]

    assert dense == before_dense
    assert sparse == before_sparse


def test_duplicate_in_source_keeps_smallest_rank_and_first_equal_rank() -> None:
    dense = [
        candidate("V", 1, 5, 0.5, label="late"),
        candidate("V", 1, 2, 0.8, label="winner"),
        candidate("V", 1, 2, 0.9, label="ignored-equal-rank"),
    ]
    result = weighted_rrf(dense, [])
    assert len(result) == 1
    assert result[0]["dense_rank"] == 2
    assert result[0]["dense_score"] == 0.8
    assert result[0]["label"] == "winner"


def test_tie_break_is_deterministic_by_identity() -> None:
    dense = [candidate("V2", 2, 1, 0.9), candidate("V1", 9, 1, 0.8), candidate("V1", 3, 1, 0.7)]
    expected = [("V1", 3), ("V1", 9), ("V2", 2)]
    assert [(item["video_id"], item["frame_id"]) for item in weighted_rrf(dense, [])] == expected
    assert [(item["video_id"], item["frame_id"]) for item in weighted_rrf(list(reversed(dense)), [])] == expected


@pytest.mark.parametrize("rank", [0, -1])
def test_non_positive_source_rank_is_rejected(rank: int) -> None:
    with pytest.raises(ValueError, match="rank"):
        weighted_rrf([candidate("V", 1, rank, 1.0)], [])


@pytest.mark.parametrize("parameter", ["k", "top_k"])
def test_boolean_integer_parameter_is_rejected(parameter: str) -> None:
    with pytest.raises(ValueError, match=parameter):
        weighted_rrf([], [], **{parameter: True})  # type: ignore[arg-type]


def test_boolean_source_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="rank"):
        weighted_rrf([candidate("V", 1, True, 1.0)], [])


@pytest.mark.parametrize("frame_id", [True, "42", 42.0])
def test_frame_id_must_be_an_integer_and_not_boolean(frame_id: object) -> None:
    invalid = {"video_id": "V", "frame_id": frame_id, "rank": 1}
    with pytest.raises(ValueError, match="frame_id must be an integer"):
        weighted_rrf([invalid], [])


@pytest.mark.parametrize("missing_field", ["video_id", "frame_id", "rank"])
def test_missing_candidate_field_has_clear_error(missing_field: str) -> None:
    invalid = {"video_id": "V", "frame_id": 1, "rank": 1}
    del invalid[missing_field]
    with pytest.raises(ValueError, match=rf"missing required field.*{missing_field}"):
        weighted_rrf([invalid], [])


def test_non_mapping_candidate_has_clear_error() -> None:
    with pytest.raises(ValueError, match="candidate must be a mapping"):
        weighted_rrf([None], [])  # type: ignore[list-item]


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_invalid_k_is_rejected(k: object) -> None:
    with pytest.raises(ValueError, match="k"):
        weighted_rrf([], [], k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize(("weight_name", "weight"), [("weight_dense", -1.0), ("weight_sparse", -0.1)])
def test_negative_weight_is_rejected(weight_name: str, weight: float) -> None:
    kwargs = {weight_name: weight}
    with pytest.raises(ValueError, match=weight_name):
        weighted_rrf([], [], **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("weight_name", ["weight_dense", "weight_sparse"])
def test_non_finite_weight_is_rejected(weight_name: str, weight: float) -> None:
    with pytest.raises(ValueError, match=weight_name):
        weighted_rrf([], [], **{weight_name: weight})  # type: ignore[arg-type]


def test_both_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot both be 0"):
        weighted_rrf([], [], weight_dense=0, weight_sparse=0)
