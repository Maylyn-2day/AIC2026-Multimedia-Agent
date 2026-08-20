"""
Unit tests for Step 3: Sketch Service & Frontend Image Utilities.

Groups:
1. ``frontend.utils.image_utils`` — Base64 encode/decode, draw_bboxes,
   create_thumbnail.
2. ``backend.services.sketch_service`` — decode_base64_image, deterministic
   vector, SketchService lifecycle, encoder injection, Qdrant mock.
"""

from __future__ import annotations

import base64
import io
import sys

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Shared PIL image factory
# ---------------------------------------------------------------------------

def _make_pil_image(w: int = 64, h: int = 48, color: tuple = (200, 100, 50)) -> "PILImage":
    """Return a solid-colour PIL Image for testing."""
    from PIL import Image
    return Image.new("RGB", (w, h), color)


def _image_to_b64(image: "PILImage", fmt: str = "PNG") -> str:
    """Encode a PIL image to a raw Base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ===========================================================================
# 1. frontend.utils.image_utils
# ===========================================================================


class TestBase64Encode:
    """Tests for :func:`base64_encode`."""

    def test_returns_string(self) -> None:
        from frontend.utils.image_utils import base64_encode
        img = _make_pil_image()
        result = base64_encode(img)
        assert isinstance(result, str)

    def test_result_is_valid_base64(self) -> None:
        from frontend.utils.image_utils import base64_encode
        img = _make_pil_image()
        data = base64_encode(img, fmt="PNG")
        decoded = base64.b64decode(data)
        assert len(decoded) > 0

    def test_jpeg_format_produces_smaller_output_than_png(self) -> None:
        import numpy as np
        from PIL import Image
        from frontend.utils.image_utils import base64_encode
        # Use a large image with random noise — PNG can't compress it well,
        # while JPEG lossy compression reliably wins.
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        jpeg_len = len(base64_encode(img, fmt="JPEG"))
        png_len = len(base64_encode(img, fmt="PNG"))
        assert jpeg_len < png_len

    def test_unsupported_format_raises_value_error(self) -> None:
        from frontend.utils.image_utils import base64_encode
        img = _make_pil_image()
        with pytest.raises(ValueError, match="Unsupported"):
            base64_encode(img, fmt="WEBP2_INVALID")

    def test_rgba_image_converted_to_rgb_before_encode(self) -> None:
        from PIL import Image
        from frontend.utils.image_utils import base64_encode
        img = Image.new("RGBA", (32, 32), (100, 150, 200, 128))
        # Should not raise — RGBA converted to RGB internally.
        data = base64_encode(img, fmt="JPEG")
        assert len(data) > 0


class TestBase64Decode:
    """Tests for :func:`base64_decode`."""

    def test_round_trip_png(self) -> None:
        from frontend.utils.image_utils import base64_encode, base64_decode
        original = _make_pil_image(40, 30)
        recovered = base64_decode(base64_encode(original, fmt="PNG"))
        assert recovered.size == original.size
        assert recovered.mode == "RGB"

    def test_accepts_data_uri_prefix(self) -> None:
        from frontend.utils.image_utils import base64_decode
        img = _make_pil_image()
        raw_b64 = _image_to_b64(img, "PNG")
        uri = f"data:image/png;base64,{raw_b64}"
        decoded = base64_decode(uri)
        assert decoded.size == img.size

    def test_empty_string_raises_value_error(self) -> None:
        from frontend.utils.image_utils import base64_decode
        with pytest.raises(ValueError, match="empty"):
            base64_decode("")

    def test_invalid_base64_raises_value_error(self) -> None:
        from frontend.utils.image_utils import base64_decode
        with pytest.raises(ValueError, match="Invalid Base64"):
            base64_decode("!!!NOT_VALID_BASE64!!!")

    def test_valid_base64_non_image_raises_value_error(self) -> None:
        from frontend.utils.image_utils import base64_decode
        garbage = base64.b64encode(b"this is not an image").decode()
        with pytest.raises(ValueError, match="Unrecognised"):
            base64_decode(garbage)

    def test_returns_rgb_mode(self) -> None:
        from frontend.utils.image_utils import base64_decode
        img = _make_pil_image()
        decoded = base64_decode(_image_to_b64(img, "PNG"))
        assert decoded.mode == "RGB"


class TestDrawBboxes:
    """Tests for :func:`draw_bboxes`."""

    _BBOX = {"label": "person", "confidence": 0.92, "bbox": [0.1, 0.1, 0.5, 0.8]}

    def test_returns_pil_image(self) -> None:
        from PIL import Image
        from frontend.utils.image_utils import draw_bboxes
        img = _make_pil_image(200, 150)
        result = draw_bboxes(img, [self._BBOX])
        assert isinstance(result, Image.Image)

    def test_output_same_size_as_input(self) -> None:
        from frontend.utils.image_utils import draw_bboxes
        img = _make_pil_image(200, 150)
        result = draw_bboxes(img, [self._BBOX])
        assert result.size == img.size

    def test_does_not_mutate_original(self) -> None:
        from frontend.utils.image_utils import draw_bboxes
        img = _make_pil_image(100, 80)
        original_data = list(img.getdata())
        draw_bboxes(img, [self._BBOX])
        assert list(img.getdata()) == original_data

    def test_empty_bboxes_returns_identical_image(self) -> None:
        from frontend.utils.image_utils import draw_bboxes
        img = _make_pil_image(100, 80)
        result = draw_bboxes(img, [])
        assert result.size == img.size

    def test_accepts_pydantic_grounding_result(self) -> None:
        from frontend.utils.image_utils import draw_bboxes
        from backend.schemas.rerank import GroundingResult
        img = _make_pil_image(200, 150)
        gr = GroundingResult(label="car", confidence=0.75, bbox=[0.2, 0.2, 0.7, 0.6])
        # Must not raise when given a Pydantic model instead of a dict.
        result = draw_bboxes(img, [gr])
        assert result.size == img.size

    def test_multiple_bboxes(self) -> None:
        from frontend.utils.image_utils import draw_bboxes
        img = _make_pil_image(300, 200)
        boxes = [
            {"label": "person", "confidence": 0.9, "bbox": [0.0, 0.0, 0.4, 0.4]},
            {"label": "car",    "confidence": 0.7, "bbox": [0.5, 0.5, 0.9, 0.9]},
        ]
        result = draw_bboxes(img, boxes)
        assert result.size == img.size

    def test_show_labels_false_does_not_raise(self) -> None:
        from frontend.utils.image_utils import draw_bboxes
        img = _make_pil_image(100, 80)
        result = draw_bboxes(img, [self._BBOX], show_labels=False)
        assert result.size == img.size


class TestCreateThumbnail:
    """Tests for :func:`create_thumbnail`."""

    def test_output_is_exact_target_size(self) -> None:
        from frontend.utils.image_utils import create_thumbnail
        img = _make_pil_image(1920, 1080)
        thumb = create_thumbnail(img, size=(320, 180))
        assert thumb.size == (320, 180)

    def test_output_mode_is_rgb(self) -> None:
        from frontend.utils.image_utils import create_thumbnail
        img = _make_pil_image(100, 100)
        thumb = create_thumbnail(img)
        assert thumb.mode == "RGB"

    def test_portrait_image_fits_in_target(self) -> None:
        from frontend.utils.image_utils import create_thumbnail
        img = _make_pil_image(100, 300)  # portrait
        thumb = create_thumbnail(img, size=(320, 180))
        assert thumb.size == (320, 180)

    def test_small_image_does_not_upscale_beyond_target(self) -> None:
        from frontend.utils.image_utils import create_thumbnail
        img = _make_pil_image(10, 10)
        thumb = create_thumbnail(img, size=(320, 180))
        # Output canvas is exact size; the image may be letterboxed.
        assert thumb.size == (320, 180)

    def test_square_image_centred_with_letterbox(self) -> None:
        from PIL import Image
        from frontend.utils.image_utils import create_thumbnail
        # A 100×100 square image in a 320×180 canvas should have black bars.
        img = Image.new("RGB", (100, 100), (255, 0, 0))  # solid red
        thumb = create_thumbnail(img, size=(320, 180))
        # Top-left corner should be black (letterbox region).
        top_left_pixel = thumb.getpixel((0, 0))
        assert top_left_pixel == (0, 0, 0)


# ===========================================================================
# 2. backend.services.sketch_service
# ===========================================================================


class TestDecodeBase64Image:
    """Tests for :func:`decode_base64_image` in sketch_service."""

    def test_valid_png_returns_rgb_image(self) -> None:
        from backend.services.sketch_service import decode_base64_image
        img = _make_pil_image()
        data = _image_to_b64(img, "PNG")
        result = decode_base64_image(data)
        assert result.mode == "RGB"
        assert result.size == img.size

    def test_accepts_data_uri_prefix(self) -> None:
        from backend.services.sketch_service import decode_base64_image
        img = _make_pil_image(30, 30)
        uri = f"data:image/png;base64,{_image_to_b64(img, 'PNG')}"
        result = decode_base64_image(uri)
        assert result.size == (30, 30)

    def test_empty_string_raises_value_error(self) -> None:
        from backend.services.sketch_service import decode_base64_image
        with pytest.raises(ValueError, match="empty"):
            decode_base64_image("")

    def test_invalid_base64_raises_value_error(self) -> None:
        from backend.services.sketch_service import decode_base64_image
        with pytest.raises(ValueError, match="Invalid Base64"):
            decode_base64_image("not!valid!base64")

    def test_non_image_bytes_raises_value_error(self) -> None:
        from backend.services.sketch_service import decode_base64_image
        garbage = base64.b64encode(b"plaintext").decode()
        with pytest.raises(ValueError, match="identify"):
            decode_base64_image(garbage)


class TestDeterministicUnitVector:
    """Tests for :func:`_deterministic_unit_vector`."""

    def test_returns_correct_dimension(self) -> None:
        from backend.services.sketch_service import _deterministic_unit_vector
        vec = _deterministic_unit_vector(b"seed", dim=768)
        assert vec.shape == (768,)

    def test_is_unit_normalised(self) -> None:
        import numpy as np
        from backend.services.sketch_service import _deterministic_unit_vector
        vec = _deterministic_unit_vector(b"test", dim=128)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_same_seed_same_output(self) -> None:
        import numpy as np
        from backend.services.sketch_service import _deterministic_unit_vector
        v1 = _deterministic_unit_vector(b"fixed_seed", dim=64)
        v2 = _deterministic_unit_vector(b"fixed_seed", dim=64)
        assert np.allclose(v1, v2)

    def test_different_seeds_produce_different_vectors(self) -> None:
        import numpy as np
        from backend.services.sketch_service import _deterministic_unit_vector
        v1 = _deterministic_unit_vector(b"seed_a", dim=64)
        v2 = _deterministic_unit_vector(b"seed_b", dim=64)
        assert not np.allclose(v1, v2)

    def test_returns_float32(self) -> None:
        import numpy as np
        from backend.services.sketch_service import _deterministic_unit_vector
        vec = _deterministic_unit_vector(b"dtype_test", dim=32)
        assert vec.dtype == np.float32


class TestSketchServiceProperties:
    """Tests for SketchService constructor and properties."""

    def test_no_encoder_has_encoder_is_false(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None)
        assert svc.has_encoder is False

    def test_injected_encoder_has_encoder_is_true(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=lambda img: None)  # type: ignore[arg-type]
        assert svc.has_encoder is True

    def test_no_qdrant_has_qdrant_is_false(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService()
        assert svc.has_qdrant is False

    def test_injected_qdrant_has_qdrant_is_true(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(qdrant_client=MagicMock())
        assert svc.has_qdrant is True


class TestSketchServiceSketchToVector:
    """Tests for :meth:`SketchService.sketch_to_vector`."""

    def _make_sketch_b64(self) -> str:
        img = _make_pil_image(64, 64)
        return _image_to_b64(img, "PNG")

    def test_fallback_returns_correct_dim(self) -> None:
        import numpy as np
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None, vector_dim=768)
        vec = svc.sketch_to_vector(self._make_sketch_b64())
        assert vec.shape == (768,)

    def test_fallback_is_unit_normalised(self) -> None:
        import numpy as np
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None, vector_dim=256)
        vec = svc.sketch_to_vector(self._make_sketch_b64())
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_same_sketch_produces_same_vector(self) -> None:
        import numpy as np
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None, vector_dim=128)
        b64 = self._make_sketch_b64()
        v1 = svc.sketch_to_vector(b64)
        v2 = svc.sketch_to_vector(b64)
        assert np.allclose(v1, v2)

    def test_injected_encoder_is_called(self) -> None:
        import numpy as np
        from backend.services.sketch_service import SketchService
        mock_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        encoder = MagicMock(return_value=mock_vec)
        svc = SketchService(encoder=encoder, apply_edge_detection=False)
        svc.sketch_to_vector(self._make_sketch_b64())
        encoder.assert_called_once()

    def test_encoder_result_is_normalised(self) -> None:
        import numpy as np
        from backend.services.sketch_service import SketchService
        unnormalised = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        svc = SketchService(encoder=lambda img: unnormalised, apply_edge_detection=False)
        vec = svc.sketch_to_vector(self._make_sketch_b64())
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_encoder_failure_falls_back_to_deterministic(self) -> None:
        import numpy as np
        from backend.services.sketch_service import SketchService
        def bad_encoder(img):
            raise RuntimeError("GPU OOM")
        svc = SketchService(encoder=bad_encoder, vector_dim=64, apply_edge_detection=False)
        vec = svc.sketch_to_vector(self._make_sketch_b64())
        assert vec.shape == (64,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_empty_sketch_raises_value_error(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService()
        with pytest.raises(ValueError):
            svc.sketch_to_vector("")

    def test_edge_detection_disabled_still_produces_vector(self) -> None:
        import numpy as np
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None, vector_dim=32, apply_edge_detection=False)
        vec = svc.sketch_to_vector(self._make_sketch_b64())
        assert vec.shape == (32,)


class TestSketchServiceSearchBySketch:
    """Tests for :meth:`SketchService.search_by_sketch`."""

    def _make_sketch_b64(self) -> str:
        return _image_to_b64(_make_pil_image(32, 32), "PNG")

    def test_mock_mode_returns_list(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None)
        results = svc.search_by_sketch(self._make_sketch_b64(), top_k=5)
        assert isinstance(results, list)

    def test_mock_results_have_required_keys(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None)
        results = svc.search_by_sketch(self._make_sketch_b64(), top_k=5)
        for r in results:
            assert "video_id" in r
            assert "frame_id" in r
            assert "score" in r

    def test_mock_results_count_bounded_by_mock_limit(self) -> None:
        from backend.services.sketch_service import SketchService, _MOCK_RESULT_COUNT
        svc = SketchService(encoder=None)
        results = svc.search_by_sketch(self._make_sketch_b64(), top_k=500)
        assert len(results) <= _MOCK_RESULT_COUNT

    def test_scores_are_descending(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None)
        results = svc.search_by_sketch(self._make_sketch_b64(), top_k=10)
        scores = [float(r["score"]) for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_one_returns_single_result(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None)
        results = svc.search_by_sketch(self._make_sketch_b64(), top_k=1)
        assert len(results) == 1

    def test_qdrant_client_is_called_when_available(self) -> None:
        from backend.services.sketch_service import SketchService
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.payload = {"video_id": "V1", "frame_id": 1}
        mock_hit.score = 0.95
        mock_qdrant.search.return_value = [mock_hit]

        svc = SketchService(encoder=None, qdrant_client=mock_qdrant)
        results = svc.search_by_sketch(self._make_sketch_b64(), top_k=5)

        mock_qdrant.search.assert_called_once()
        assert results[0]["video_id"] == "V1"
        assert results[0]["score"] == pytest.approx(0.95)

    def test_qdrant_failure_returns_empty_list(self) -> None:
        from backend.services.sketch_service import SketchService
        mock_qdrant = MagicMock()
        mock_qdrant.search.side_effect = ConnectionError("Qdrant down")
        svc = SketchService(encoder=None, qdrant_client=mock_qdrant)
        results = svc.search_by_sketch(self._make_sketch_b64())
        assert results == []


    def test_deterministic_same_sketch_same_results(self) -> None:
        from backend.services.sketch_service import SketchService
        svc = SketchService(encoder=None)
        b64 = self._make_sketch_b64()
        r1 = svc.search_by_sketch(b64, top_k=5)
        r2 = svc.search_by_sketch(b64, top_k=5)
        assert r1 == r2


# ===========================================================================
# 3. HTTP Integration tests — POST /v1/query/sketch
# ===========================================================================


@pytest.fixture
def mock_sketch_service() -> "SketchService":
    from backend.services.sketch_service import SketchService
    return SketchService(encoder=None, vector_dim=768, apply_edge_detection=False)


@pytest.fixture
def sketch_api_client(mock_sketch_service: "SketchService"):
    """TestClient with SketchService dependency override for sketch route."""
    from fastapi.testclient import TestClient
    from backend.api.v1.sketch import get_sketch_service
    from backend.main import app

    app.dependency_overrides[get_sketch_service] = lambda: mock_sketch_service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _make_sketch_payload(n: int = 1) -> dict:
    """Build a minimal valid SketchQueryRequest payload."""
    import io, base64
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 200, 200)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"sketch_base64": b64, "prompt": "test sketch", "top_k": n}


class TestSketchRouteSchema:
    """Integration tests for POST /v1/query/sketch."""

    def test_returns_200(self, sketch_api_client) -> None:
        resp = sketch_api_client.post("/v1/query/sketch", json=_make_sketch_payload())
        assert resp.status_code == 200

    def test_response_has_base_response_envelope(self, sketch_api_client) -> None:
        body = sketch_api_client.post("/v1/query/sketch", json=_make_sketch_payload()).json()
        assert "status" in body
        assert "data" in body
        assert "execution_time" in body

    def test_data_has_results_and_total(self, sketch_api_client) -> None:
        data = sketch_api_client.post("/v1/query/sketch", json=_make_sketch_payload()).json()["data"]
        assert "results" in data
        assert "total" in data

    def test_total_matches_results_len(self, sketch_api_client) -> None:
        data = sketch_api_client.post("/v1/query/sketch", json=_make_sketch_payload(5)).json()["data"]
        assert data["total"] == len(data["results"])

    def test_results_have_required_fields(self, sketch_api_client) -> None:
        data = sketch_api_client.post("/v1/query/sketch", json=_make_sketch_payload(3)).json()["data"]
        for r in data["results"]:
            assert "video_id" in r
            assert "frame_id" in r
            assert "score" in r

    def test_status_is_success(self, sketch_api_client) -> None:
        body = sketch_api_client.post("/v1/query/sketch", json=_make_sketch_payload()).json()
        assert body["status"] == "success"

    def test_session_id_header_accepted(self, sketch_api_client) -> None:
        resp = sketch_api_client.post(
            "/v1/query/sketch",
            json=_make_sketch_payload(),
            headers={"X-Session-ID": "sess-abc"},
        )
        assert resp.status_code == 200

    def test_missing_sketch_base64_returns_422(self, sketch_api_client) -> None:
        resp = sketch_api_client.post(
            "/v1/query/sketch",
            json={"prompt": "missing sketch", "top_k": 10},
        )
        assert resp.status_code == 422

    def test_top_k_above_limit_returns_422(self, sketch_api_client) -> None:
        payload = _make_sketch_payload()
        payload["top_k"] = 501
        resp = sketch_api_client.post("/v1/query/sketch", json=payload)
        assert resp.status_code == 422

    def test_top_k_zero_returns_422(self, sketch_api_client) -> None:
        payload = _make_sketch_payload()
        payload["top_k"] = 0
        resp = sketch_api_client.post("/v1/query/sketch", json=payload)
        assert resp.status_code == 422


# ===========================================================================
# 4. HTTP Integration tests — POST /v1/query/image-example
# ===========================================================================


@pytest.fixture
def image_query_api_client(mock_sketch_service: "SketchService", tmp_path):
    """TestClient with SketchService + keyframes_dir overrides for image-query route."""
    from fastapi.testclient import TestClient
    from backend.api.v1.image_query import get_sketch_service, get_keyframes_dir
    from backend.main import app

    app.dependency_overrides[get_sketch_service] = lambda: mock_sketch_service
    app.dependency_overrides[get_keyframes_dir] = lambda: tmp_path
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, tmp_path
    app.dependency_overrides.clear()


def _make_b64_image() -> str:
    import io, base64
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 150, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestImageQueryRouteSchema:
    """Integration tests for POST /v1/query/image-example."""

    def test_mode_a_returns_200(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        resp = client.post(
            "/v1/query/image-example",
            json={"image_base64": _make_b64_image(), "top_k": 5},
        )
        assert resp.status_code == 200

    def test_mode_a_response_has_results(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        data = client.post(
            "/v1/query/image-example",
            json={"image_base64": _make_b64_image(), "top_k": 5},
        ).json()["data"]
        assert "results" in data
        assert "total" in data

    def test_mode_a_source_mode_is_base64(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        data = client.post(
            "/v1/query/image-example",
            json={"image_base64": _make_b64_image(), "top_k": 3},
        ).json()["data"]
        assert data["source"]["mode"] == "base64"

    def test_mode_b_valid_keyframe_returns_200(self, image_query_api_client) -> None:
        from PIL import Image as PILImage
        client, tmp_path = image_query_api_client
        video_dir = tmp_path / "L01_V001"
        video_dir.mkdir()
        PILImage.new("RGB", (64, 64), (0, 128, 255)).save(video_dir / "000100.jpg")

        resp = client.post(
            "/v1/query/image-example",
            json={"video_id": "L01_V001", "frame_id": 100, "top_k": 5},
        )
        assert resp.status_code == 200

    def test_mode_b_missing_keyframe_returns_404(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        resp = client.post(
            "/v1/query/image-example",
            json={"video_id": "MISSING", "frame_id": 9999, "top_k": 5},
        )
        assert resp.status_code == 404

    def test_no_input_returns_400(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        resp = client.post(
            "/v1/query/image-example",
            json={"top_k": 5},
        )
        assert resp.status_code == 400

    def test_only_video_id_no_frame_id_returns_400(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        resp = client.post(
            "/v1/query/image-example",
            json={"video_id": "L01_V001", "top_k": 5},
        )
        assert resp.status_code == 400

    def test_results_have_keyframe_result_fields(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        data = client.post(
            "/v1/query/image-example",
            json={"image_base64": _make_b64_image(), "top_k": 5},
        ).json()["data"]
        for r in data["results"]:
            assert "video_id" in r
            assert "frame_id" in r
            assert "score" in r

    def test_execution_time_is_parseable(self, image_query_api_client) -> None:
        client, _ = image_query_api_client
        body = client.post(
            "/v1/query/image-example",
            json={"image_base64": _make_b64_image(), "top_k": 3},
        ).json()
        et = body["execution_time"]
        assert et.endswith("s") and float(et[:-1]) >= 0.0
