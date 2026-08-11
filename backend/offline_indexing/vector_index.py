"""In-process cosine search over the normalized feature matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class VectorIndex:
    """Baseline index that can later be replaced by Qdrant"""

    def __init__(self, index_directory: Path):
        self.features = np.load(index_directory / "features.npy", mmap_mode="r")
        with (index_directory / "records.jsonl").open(encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file]
        if len(self.features) != len(self.records):
            raise ValueError("features.npy and records.jsonl have different row counts")
        self.feature_norms = np.linalg.norm(self.features, axis=1)

    def search(
        self,
        vector: list[float],
        top_k: int = 100,
        video_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self.features.shape[1],) or not np.isfinite(query).all():
            raise ValueError(f"vector must contain {self.features.shape[1]} finite numbers")
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            raise ValueError("vector must not be all zeros")

        candidates = np.asarray(
            [index for index, record in enumerate(self.records) if video_id is None or record["video_id"] == video_id],
            dtype=np.int64,
        )
        if not len(candidates):
            return []
        scores = (self.features[candidates] @ query) / np.maximum(
            self.feature_norms[candidates] * query_norm,
            1e-12,
        )
        result_count = min(max(1, int(top_k)), 100, len(candidates))
        best = np.argpartition(scores, -result_count)[-result_count:]
        best = best[np.argsort(scores[best])[::-1]]
        return [{**self.records[int(candidates[index])], "score": float(scores[index])} for index in best]
