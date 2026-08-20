"""API tests for local Agent routing and session isolation."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


def route(client: TestClient, query: str, session: str) -> dict:
    response = client.post(
        "/v1/agent/route",
        json={"raw_query": query, "session_id": session},
        headers={"X-Session-ID": session},
    )
    assert response.status_code == 200
    return response.json()


def route_with_task(client: TestClient, query: str, session: str, task_type: str | None) -> dict:
    response = client.post(
        "/v1/agent/route",
        json={"raw_query": query, "session_id": session, "task_type": task_type},
        headers={"X-Session-ID": session},
    )
    assert response.status_code == 200
    return response.json()


def test_kis_route(client: TestClient) -> None:
    body = route(client, "Người phụ nữ mặc áo đỏ", "api-kis")
    assert body["data"]["task_type"] == "KIS"
    assert body["data"]["top_k"] == 100
    assert body["agent_reasoning"] is None


def test_vqa_route(client: TestClient) -> None:
    body = route(client, "Người này đang cầm gì?", "api-vqa")
    assert body["data"]["task_type"] == "VQA"
    assert body["data"]["requires_rerank"] is True


def test_explicit_kis_with_question_phrase_remains_kis(client: TestClient) -> None:
    body = route_with_task(client, "Tìm cảnh ở đâu có biển báo", "explicit-kis", "KIS")
    assert body["data"]["task_type"] == "KIS"
    assert body["data"]["requires_rerank"] is False


def test_explicit_vqa_always_requires_rerank(client: TestClient) -> None:
    body = route_with_task(client, "Mô tả màu chiếc xe", "explicit-vqa", "VQA")
    assert body["data"]["task_type"] == "VQA"
    assert body["data"]["requires_rerank"] is True


def test_explicit_trake_creates_ordered_events(client: TestClient) -> None:
    body = route_with_task(client, "Mở cửa rồi bước vào rồi ngồi xuống", "explicit-trake", "TRAKE")
    assert body["data"]["task_type"] == "TRAKE"
    assert [event["order"] for event in body["data"]["events"]] == [1, 2, 3]


def test_auto_fallback_prefers_kis_without_enough_evidence(client: TestClient) -> None:
    body = route_with_task(client, "Tìm người ở đâu đó trong công viên", "fallback-kis", None)
    assert body["data"]["task_type"] == "KIS"


def test_invalid_explicit_task_type_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/route",
        json={"raw_query": "query", "session_id": "invalid-task", "task_type": "QAQ"},
        headers={"X-Session-ID": "invalid-task"},
    )
    assert response.status_code == 422


def test_trake_route_has_one_to_n_order(client: TestClient) -> None:
    body = route(client, "Người đàn ông vào phòng rồi ngồi xuống rồi mở laptop", "api-trake")
    assert body["data"]["task_type"] == "TRAKE"
    assert [event["order"] for event in body["data"]["events"]] == [1, 2, 3]


def test_missing_session_header_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/agent/route", json={"raw_query": "query", "session_id": "body"})
    assert response.status_code == 422


@pytest.mark.parametrize("header", ["", "   ", "x" * 201])
def test_invalid_session_header_is_rejected(client: TestClient, header: str) -> None:
    response = client.post(
        "/v1/agent/route",
        json={"raw_query": "query", "session_id": "body"},
        headers={"X-Session-ID": header},
    )
    assert response.status_code == 422


def test_body_header_session_mismatch_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/route",
        json={"raw_query": "query", "session_id": "body"},
        headers={"X-Session-ID": "header"},
    )
    assert response.status_code == 400


def test_sessions_are_independent_and_clear_is_scoped(client: TestClient) -> None:
    route(client, "first query", "session-A")
    route(client, "second query", "session-B")
    cleared = client.delete("/v1/agent/session/session-A", headers={"X-Session-ID": "session-A"})
    assert cleared.status_code == 200
    route(client, "third query", "session-B")
    rejected = client.delete("/v1/agent/session/session-B", headers={"X-Session-ID": "session-A"})
    assert rejected.status_code == 400


def test_response_does_not_expose_hidden_reasoning(client: TestClient) -> None:
    body = route(client, "Find a blue bus", "api-safe")
    serialized = str(body).casefold()
    assert body["agent_reasoning"] is None
    assert "chain_of_thought" not in serialized
    assert "hidden reasoning" not in serialized
    assert "provider payload" not in serialized


def test_frontend_client_sends_explicit_task_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"status": "success", "data": {}}

    def post(url: str, **kwargs: object) -> Response:
        captured.update(kwargs)
        return Response()

    fake_requests = SimpleNamespace(
        post=post,
        get=lambda *args, **kwargs: Response(),
        delete=lambda *args, **kwargs: Response(),
        ConnectionError=OSError,
        Timeout=TimeoutError,
        JSONDecodeError=ValueError,
        Response=Response,
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    sys.modules.pop("frontend.utils.api_client", None)
    api_client = importlib.import_module("frontend.utils.api_client")
    api_client.route_agent("query", "session", "KIS")
    assert captured["json"] == {"raw_query": "query", "session_id": "session", "task_type": "KIS"}
