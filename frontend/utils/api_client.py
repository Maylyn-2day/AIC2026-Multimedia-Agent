"""Resilient HTTP client for the Streamlit frontend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

BACKEND_URL = os.getenv("AIC_BACKEND_URL", "http://localhost:8000").rstrip("/")
MOCK_DIR = Path(__file__).resolve().parents[2] / "data" / "mock"


def _error(message: str, *, status_code: int | None = None) -> dict[str, Any]:
    return {"status": "error", "data": None, "message": message, "status_code": status_code}


def _request_session_id(session_id: str | None) -> str:
    return session_id if session_id else str(uuid4())


def _load_mock(filename: str) -> dict[str, Any]:
    path = MOCK_DIR / filename
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _error(f"Mock file not found: {filename}")


def _decode_response(response: requests.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except requests.JSONDecodeError:
        return _error(f"Backend returned non-JSON HTTP {response.status_code}", status_code=response.status_code)
    if not isinstance(body, dict):
        return _error(f"Backend returned invalid JSON HTTP {response.status_code}", status_code=response.status_code)
    if response.status_code >= 400:
        body.setdefault("status", "error")
        body.setdefault("message", f"Backend request failed with HTTP {response.status_code}")
        body["status_code"] = response.status_code
    return body


def _post(endpoint: str, payload: dict[str, Any], *, session_id: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload,
            headers={"Content-Type": "application/json", "X-Session-ID": session_id},
            timeout=timeout,
        )
    except (requests.ConnectionError, requests.Timeout) as error:
        return _error(f"Backend unavailable: {type(error).__name__}")
    return _decode_response(response)


def _get(endpoint: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as error:
        return _error(f"Backend unavailable: {type(error).__name__}")
    return _decode_response(response)


def check_health() -> dict[str, Any]:
    result = _get("/v1/health")
    return _load_mock("health_response.json") if result["status"] == "error" else result


def route_agent(raw_query: str, session_id: str, task_type: str | None = None) -> dict[str, Any]:
    payload = {"raw_query": raw_query, "session_id": session_id, "task_type": task_type}
    return _post("/v1/agent/route", payload, session_id=session_id)


def clear_agent_session(session_id: str) -> dict[str, Any]:
    try:
        response = requests.delete(
            f"{BACKEND_URL}/v1/agent/session/{session_id}", headers={"X-Session-ID": session_id}, timeout=5.0
        )
    except (requests.ConnectionError, requests.Timeout) as error:
        return _error(f"Backend unavailable: {type(error).__name__}")
    return _decode_response(response)


def hybrid_search(
    query: str, filters: dict[str, Any] | None = None, top_k: int = 100, session_id: str | None = None
) -> dict[str, Any]:
    result = _post(
        "/v1/db/query",
        {"raw_query": query, "filters": filters or {}, "top_k": top_k},
        session_id=_request_session_id(session_id),
    )
    return _load_mock("query_response.json") if result["status"] == "error" else result


def rerank(query: str, candidates: list[dict[str, Any]], session_id: str | None = None) -> dict[str, Any]:
    return _post(
        "/v1/rerank/early-fusion",
        {"query": query, "candidates": candidates},
        session_id=_request_session_id(session_id),
    )


def temporal_align(query: str, session_id: str | None = None) -> dict[str, Any]:
    return _post(
        "/v1/temporal/align",
        {"raw_query": query, "auto_decompose": True},
        session_id=_request_session_id(session_id),
    )


def submit_results(
    task_type: str,
    results: list[dict[str, Any]],
    question_id: str = "",
    session_id: str | None = None,
) -> dict[str, Any]:
    return _post(
        "/v1/submission/submit",
        {"task_type": task_type, "question_id": question_id, "results": results},
        session_id=_request_session_id(session_id),
    )
