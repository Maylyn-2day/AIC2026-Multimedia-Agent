"""
Health Endpoint Tests.

Validates the ``GET /v1/health`` endpoint returns the correct
response structure and status codes.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test suite for the /v1/health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return HTTP 200."""
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        """Health response should follow BaseResponse schema."""
        response = client.get("/v1/health")
        data = response.json()

        assert "status" in data
        assert "data" in data
        assert "message" in data
        assert "execution_time" in data
        assert data["status"] == "success"

    def test_health_contains_service_status(self, client: TestClient) -> None:
        """Health data should report Qdrant and ES status."""
        response = client.get("/v1/health")
        health_data = response.json()["data"]

        assert "qdrant" in health_data
        assert "elasticsearch" in health_data

    def test_health_contains_model_status(self, client: TestClient) -> None:
        """Health data should report loaded/unloaded model status."""
        response = client.get("/v1/health")
        health_data = response.json()["data"]

        assert "models" in health_data
        models = health_data["models"]
        assert isinstance(models, dict)

    def test_health_execution_time_present(self, client: TestClient) -> None:
        """Response should include execution_time field."""
        response = client.get("/v1/health")
        data = response.json()
        assert data["execution_time"].endswith("s")

    def test_health_has_execution_time_header(self, client: TestClient) -> None:
        """Response should include X-Execution-Time header from middleware."""
        response = client.get("/v1/health")
        assert "x-execution-time" in response.headers


class TestAllEndpointsReturnMock:
    """Smoke tests ensuring all 7 endpoints return 200 with mock data."""

    def test_query_endpoint(self, client: TestClient, session_headers: dict) -> None:
        """POST /v1/db/query should return 200."""
        response = client.post(
            "/v1/db/query",
            json={"raw_query": "test query"},
            headers=session_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_rerank_endpoint(self, client: TestClient, session_headers: dict) -> None:
        """POST /v1/rerank/early-fusion should return 200."""
        response = client.post(
            "/v1/rerank/early-fusion",
            json={
                "query": "test",
                "candidates": [{"video_id": "V001", "frame_id": 1, "score": 0.5}],
            },
            headers=session_headers,
        )
        assert response.status_code == 200

    def test_image_query_endpoint(self, client: TestClient, session_headers: dict) -> None:
        """POST /v1/query/image-example should return 200 with Mode A (base64 image)."""
        import io, base64
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (100, 150, 200)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.post(
            "/v1/query/image-example",
            json={"image_base64": b64, "top_k": 5},
            headers=session_headers,
        )
        assert response.status_code == 200

    def test_sketch_endpoint(self, client: TestClient, session_headers: dict) -> None:
        """POST /v1/query/sketch should return 200 with a valid PNG sketch."""
        import io, base64
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (200, 200, 200)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.post(
            "/v1/query/sketch",
            json={"sketch_base64": b64},
            headers=session_headers,
        )
        assert response.status_code == 200

    def test_temporal_endpoint(self, client: TestClient, session_headers: dict) -> None:
        """POST /v1/temporal/align should return 200."""
        response = client.post(
            "/v1/temporal/align",
            json={"raw_query": "temporal test"},
            headers=session_headers,
        )
        assert response.status_code == 200

    def test_submission_endpoint(self, client: TestClient, session_headers: dict) -> None:
        """POST /v1/submission/submit should return 200."""
        response = client.post(
            "/v1/submission/submit",
            json={
                "task_type": "KIS",
                "results": [{"video_id": "V001", "frame_id": 100}],
            },
            headers=session_headers,
        )
        assert response.status_code == 200
        assert response.json()["data"]["submitted"] is True
