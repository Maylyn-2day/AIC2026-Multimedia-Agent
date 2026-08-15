"""
Backend API Client for Streamlit Frontend.

Provides a clean interface for calling all 7 backend endpoints.
In Phase 1, falls back to loading mock JSON files directly when
the backend is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

# Default backend URL
BACKEND_URL = "http://localhost:8000"

# Mock data directory
MOCK_DIR = Path(__file__).resolve().parents[2] / "data" / "mock"


def _load_mock(filename: str) -> dict[str, Any]:
    """Load a mock JSON response file."""
    path = MOCK_DIR / filename
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "error", "data": None, "message": f"Mock file not found: {filename}"}


def _post(endpoint: str, payload: dict, timeout: float = 5.0) -> dict[str, Any]:
    """POST to a backend endpoint with fallback to mock data."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload,
            headers={"Content-Type": "application/json", "X-Session-ID": "streamlit-dev"},
            timeout=timeout,
        )
        return resp.json()
    except (requests.ConnectionError, requests.Timeout):
        return None


def _get(endpoint: str, timeout: float = 5.0) -> dict[str, Any]:
    """GET from a backend endpoint with fallback to mock data."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}{endpoint}",
            headers={"X-Session-ID": "streamlit-dev"},
            timeout=timeout,
        )
        return resp.json()
    except (requests.ConnectionError, requests.Timeout):
        return None


def check_health() -> dict[str, Any]:
    """Call GET /v1/health or return mock data."""
    result = _get("/v1/health")
    return result if result else _load_mock("health_response.json")


def hybrid_search(query: str, filters: dict | None = None, top_k: int = 100) -> dict[str, Any]:
    """Call POST /v1/db/query or return mock data."""
    payload = {"raw_query": query, "filters": filters or {}, "top_k": top_k}
    result = _post("/v1/db/query", payload)
    return result if result else _load_mock("query_response.json")


def rerank(query: str, candidates: list[dict]) -> dict[str, Any]:
    """Call POST /v1/rerank/early-fusion or return mock data."""
    payload = {"query": query, "candidates": candidates}
    result = _post("/v1/rerank/early-fusion", payload)
    return result if result else _load_mock("rerank_response.json")


def temporal_align(query: str) -> dict[str, Any]:
    """Call POST /v1/temporal/align or return mock data."""
    payload = {"raw_query": query, "auto_decompose": True}
    result = _post("/v1/temporal/align", payload)
    return result if result else _load_mock("temporal_response.json")


def submit_results(task_type: str, results: list[dict], question_id: str = "") -> dict[str, Any]:
    """Call POST /v1/submission/submit."""
    payload = {"task_type": task_type, "question_id": question_id, "results": results}
    result = _post("/v1/submission/submit", payload)
    return result if result else {"status": "error", "message": "Backend unavailable"}
