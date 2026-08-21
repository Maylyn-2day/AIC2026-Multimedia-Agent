"""Validation tests for the POST /v1/db/query request contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.schemas.query import HybridQueryRequest


def test_timestamp_range_accepts_full_iso_8601_datetimes() -> None:
    """A timezone-aware ordered range should satisfy the contract."""
    request = HybridQueryRequest.model_validate(
        {
            "raw_query": "Người phụ nữ mặc áo đỏ tại HTV9",
            "filters": {
                "timestamp_range": [
                    "2026-03-14T10:00:00Z",
                    "2026-03-14T12:00:00Z",
                ]
            },
        }
    )

    assert request.filters.timestamp_range is not None
    assert request.filters.timestamp_range[0] <= request.filters.timestamp_range[1]


def test_timestamp_range_accepts_equal_start_and_end() -> None:
    """A range with equal start and end timestamps should be valid."""
    request = HybridQueryRequest.model_validate(
        {
            "raw_query": "test query",
            "filters": {
                "timestamp_range": [
                    "2026-03-14T10:00:00Z",
                    "2026-03-14T10:00:00Z",
                ]
            },
        }
    )

    assert request.filters.timestamp_range is not None
    assert request.filters.timestamp_range[0] == request.filters.timestamp_range[1]


@pytest.mark.parametrize(
    "timestamp_range",
    [
        ["2026-03-14T10:00:00Z"],
        [
            "2026-03-14T10:00:00Z",
            "2026-03-14T11:00:00Z",
            "2026-03-14T12:00:00Z",
        ],
    ],
)
def test_timestamp_range_rejects_wrong_number_of_items(timestamp_range: list[str]) -> None:
    """A timestamp range must contain exactly two items."""
    with pytest.raises(ValidationError):
        HybridQueryRequest.model_validate(
            {
                "raw_query": "test query",
                "filters": {"timestamp_range": timestamp_range},
            }
        )


@pytest.mark.parametrize(
    "timestamp_range",
    [
        ["10:00", "12:00"],
        ["2026-03-14T10:00:00", "2026-03-14T12:00:00"],
        ["2026-03-14T12:00:00Z", "2026-03-14T10:00:00Z"],
    ],
)
def test_timestamp_range_rejects_invalid_contract_values(timestamp_range: list[str]) -> None:
    """Short, timezone-naive, and reverse-ordered ranges must be rejected."""
    with pytest.raises(ValidationError):
        HybridQueryRequest.model_validate(
            {
                "raw_query": "test query",
                "filters": {"timestamp_range": timestamp_range},
            }
        )


def test_query_endpoint_rejects_reverse_timestamp_range(
    client: TestClient,
    session_headers: dict[str, str],
) -> None:
    """The HTTP boundary should surface invalid timestamp ordering as 422."""
    response = client.post(
        "/v1/db/query",
        json={
            "raw_query": "test query",
            "filters": {
                "timestamp_range": [
                    "2026-03-14T12:00:00Z",
                    "2026-03-14T10:00:00Z",
                ]
            },
        },
        headers=session_headers,
    )

    assert response.status_code == 422
