"""
Pytest Configuration and Shared Fixtures.

Provides the FastAPI test client and common fixtures used
across all test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Provide a FastAPI TestClient scoped to the entire test session.

    Uses ``TestClient`` from Starlette which wraps ``httpx`` and
    handles the ASGI lifespan automatically.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def session_headers() -> dict[str, str]:
    """Provide standard request headers including X-Session-ID."""
    return {
        "Content-Type": "application/json",
        "X-Session-ID": "test-session-001",
    }


@pytest.fixture
def sample_query_payload() -> dict:
    """Provide a sample hybrid query request payload."""
    return {
        "raw_query": "Người phụ nữ mặc áo đỏ tại HTV9",
        "filters": {
            "objects": ["person"],
            "ocr": "HTV9",
        },
        "top_k": 100,
    }


@pytest.fixture
def sample_submission_payload() -> dict:
    """Provide a sample submission request payload."""
    return {
        "task_type": "KIS",
        "question_id": "Q001",
        "results": [
            {"video_id": "L01_V001", "frame_id": 1500},
            {"video_id": "L01_V001", "frame_id": 1520},
        ],
    }


@pytest.fixture
def sample_rerank_payload() -> dict:
    """Provide a sample rerank request payload."""
    return {
        "query": "A woman in a red shirt",
        "candidates": [
            {"video_id": "L01_V001", "frame_id": 1500, "score": 0.92},
            {"video_id": "L01_V001", "frame_id": 1520, "score": 0.88},
        ],
        "task_type": "VQA",
        "extract_answer": True,
    }


@pytest.fixture
def sample_temporal_payload() -> dict:
    """Provide a sample temporal alignment request payload."""
    return {
        "raw_query": "A woman enters a coffee shop, sits down, and opens her laptop",
        "auto_decompose": True,
        "top_k": 100,
    }
