"""Tests for strict local submission validation and packaging."""

from __future__ import annotations

import importlib
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.core.exceptions import SubmissionFormatError
from backend.schemas.submission import SubmissionPayload
from backend.services.submission_service import validate_and_package_submission


def payload(task_type: str, results: list[dict], question_id: str = " Q1 ") -> SubmissionPayload:
    return SubmissionPayload.model_validate({"task_type": task_type, "question_id": question_id, "results": results})


def test_valid_kis_and_order_is_preserved() -> None:
    source = payload("KIS", [{"video_id": " V1 ", "frame_id": 9}, {"video_id": "V2", "frame_id": 2}])
    before = source.model_copy(deep=True)
    packaged = validate_and_package_submission(source)
    assert [(item.video_id, item.frame_id) for item in packaged.results] == [("V1", 9), ("V2", 2)]
    assert source == before
    assert packaged.validated is True
    assert packaged.submitted is False
    assert packaged.question_id == "Q1"


def test_valid_vqa() -> None:
    packaged = validate_and_package_submission(payload("VQA", [{"video_id": "V", "frame_id": 1, "answer": " red "}]))
    assert packaged.results[0].answer == "red"


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_vqa_requires_non_blank_answer(answer: str | None) -> None:
    with pytest.raises(SubmissionFormatError):
        validate_and_package_submission(payload("VQA", [{"video_id": "V", "frame_id": 1, "answer": answer}]))


def test_kis_rejects_answer() -> None:
    with pytest.raises(SubmissionFormatError):
        validate_and_package_submission(payload("KIS", [{"video_id": "V", "frame_id": 1, "answer": "x"}]))


def test_valid_trake() -> None:
    packaged = validate_and_package_submission(
        payload("TRAKE", [{"video_id": "V", "frame_id": 1}, {"video_id": "V", "frame_id": 5}])
    )
    assert packaged.result_count == 2


def test_trake_rejects_different_videos() -> None:
    with pytest.raises(SubmissionFormatError, match="same video"):
        validate_and_package_submission(
            payload("TRAKE", [{"video_id": "A", "frame_id": 1}, {"video_id": "B", "frame_id": 2}])
        )


@pytest.mark.parametrize("frame_ids", [[2, 1], [1, 1]])
def test_trake_rejects_non_increasing_frames(frame_ids: list[int]) -> None:
    with pytest.raises(SubmissionFormatError, match="strictly increasing"):
        validate_and_package_submission(
            payload("TRAKE", [{"video_id": "V", "frame_id": frame_id} for frame_id in frame_ids])
        )


@pytest.mark.parametrize("task_type", ["KIS", "VQA"])
def test_kis_and_vqa_reject_duplicates(task_type: str) -> None:
    answer = "x" if task_type == "VQA" else None
    with pytest.raises(SubmissionFormatError, match="duplicate"):
        validate_and_package_submission(
            payload(
                task_type,
                [
                    {"video_id": "V", "frame_id": 1, "answer": answer},
                    {"video_id": "V", "frame_id": 1, "answer": answer},
                ],
            )
        )


@pytest.mark.parametrize("frame_id", [True, "1", 1.0])
def test_frame_id_is_strict_integer(frame_id: object) -> None:
    with pytest.raises(ValidationError):
        payload("KIS", [{"video_id": "V", "frame_id": frame_id}])


def test_extra_submission_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SubmissionPayload.model_validate(
            {"task_type": "KIS", "results": [{"video_id": "V", "frame_id": 1, "score": 0.9}]}
        )


@pytest.mark.parametrize("count", [0, 101])
def test_result_count_bounds(count: int) -> None:
    data = payload("KIS", [{"video_id": f"V{index}", "frame_id": index} for index in range(count)])
    with pytest.raises(SubmissionFormatError, match="between 1 and 100"):
        validate_and_package_submission(data)


def test_endpoint_is_validate_only_and_does_not_import_network_client(client: TestClient) -> None:
    endpoint_module = importlib.import_module("backend.api.v1.submission")
    service_module = importlib.import_module("backend.services.submission_service")
    assert "requests" not in vars(endpoint_module)
    assert "requests" not in vars(service_module)
    request = {"task_type": "KIS", "question_id": "Q", "results": [{"video_id": "V", "frame_id": 1}]}
    before = deepcopy(request)
    response = client.post("/v1/submission/submit", json=request)
    assert response.status_code == 200
    assert response.json()["data"]["validated"] is True
    assert response.json()["data"]["submitted"] is False
    assert request == before
