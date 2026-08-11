"""Dependency-free HTTP API for inspecting and searching a normalized index."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .vector_index import VectorIndex


def serve_index(index_directory: Path, host: str, port: int) -> None:
    index = VectorIndex(index_directory)

    class SearchRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            url = urlparse(self.path)
            if url.path == "/health":
                self._send_json(
                    200,
                    {"status": "ok", "frames": len(index.records), "feature_dim": index.features.shape[1]},
                )
                return
            if url.path == "/frames":
                try:
                    query = parse_qs(url.query)
                    video_id = query.get("video_id", [None])[0]
                    limit = min(max(1, int(query.get("limit", [100])[0])), 100)
                    rows = [row for row in index.records if video_id is None or row["video_id"] == video_id][:limit]
                    self._send_json(200, rows)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                return
            self._send_json(404, {"error": "use GET /health, GET /frames, or POST /search"})

        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path != "/search":
                self._send_json(404, {"error": "use POST /search"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    raise ValueError("request body must be between 1 byte and 2 MB")
                payload = json.loads(self.rfile.read(length))
                result = index.search(payload["vector"], payload.get("top_k", 100), payload.get("video_id"))
                self._send_json(200, result)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    print(f"Raw search API: http://{host}:{port}")
    ThreadingHTTPServer((host, port), SearchRequestHandler).serve_forever()
