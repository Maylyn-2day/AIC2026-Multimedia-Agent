"""Official BTC CLIP shard retrieval without concatenating feature arrays."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.services.clip_text_encoder import TextEncoder


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    rank: int
    video_id: str
    frame_id: int
    keyframe_id: str
    score: float
    image_path: Path
    feature_row: int


@dataclass(frozen=True, slots=True)
class VariantMatch:
    variant_index: int
    rank: int
    cosine_score: float


@dataclass(frozen=True, slots=True)
class FusedSearchCandidate:
    rank: int
    video_id: str
    frame_id: int
    keyframe_id: str
    image_path: Path
    feature_row: int
    rrf_score: float
    source_matches: tuple[VariantMatch, ...]


@dataclass(frozen=True, slots=True)
class _MappingRow:
    frame_id: int
    keyframe_id: str
    image_path: Path


class OfficialClipShardIndex:
    """Search official per-video 512-D arrays with mandatory BTC mappings."""

    DIMENSION = 512

    def __init__(self, feature_root: Path, mapping_root: Path, keyframe_root: Path) -> None:
        self.feature_root = Path(feature_root)
        self.mapping_root = Path(mapping_root)
        self.keyframe_root = Path(keyframe_root)
        self._shards = sorted(self.feature_root.glob("*.npy"))
        if not self._shards:
            raise ValueError("feature_root contains no .npy shards")
        self._mappings: dict[str, tuple[_MappingRow, ...]] = {}
        for shard in self._shards:
            features = np.load(shard, mmap_mode="r")
            if features.ndim != 2 or features.shape[1] != self.DIMENSION:
                raise ValueError(f"{shard.name} must have shape (N, {self.DIMENSION})")
            rows = self._load_mapping(shard.stem)
            if len(rows) != features.shape[0]:
                raise ValueError(f"{shard.stem}: {features.shape[0]} feature rows but {len(rows)} mapping rows")
            self._mappings[shard.stem] = rows

    @property
    def video_count(self) -> int:
        return len(self._shards)

    @property
    def record_count(self) -> int:
        return sum(len(rows) for rows in self._mappings.values())

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 100,
        video_ids: Sequence[str] | None = None,
    ) -> list[SearchCandidate]:
        _validate_top_k(top_k)
        return self._search(query_vector, top_k=top_k, video_ids=video_ids)

    def search_pool(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        video_ids: Sequence[str] | None = None,
    ) -> list[SearchCandidate]:
        """Return an enlarged, bounded pool for temporal sequence assembly."""
        if type(top_k) is not int or not 1 <= top_k <= 500:
            raise ValueError("pool top_k must be an integer in [1, 500]")
        return self._search(query_vector, top_k=top_k, video_ids=video_ids)

    def _search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        video_ids: Sequence[str] | None,
    ) -> list[SearchCandidate]:
        vector = np.array(query_vector, dtype=np.float32, copy=True).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (self.DIMENSION,) or not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"query vector must be finite, non-zero, and have shape ({self.DIMENSION},)")
        vector /= norm
        allowed = None if video_ids is None else _normalize_video_ids(video_ids)
        candidates: list[tuple[float, str, int, int, _MappingRow]] = []
        for shard in self._shards:
            video_id = shard.stem
            if allowed is not None and video_id not in allowed:
                continue
            features = np.load(shard, mmap_mode="r")
            matrix = np.asarray(features, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError(f"{shard.name} contains invalid feature rows")
            scores = np.asarray(matrix @ vector, dtype=np.float32) / norms
            best_by_frame: dict[int, tuple[float, int, _MappingRow]] = {}
            for row_index, (mapping, score) in enumerate(zip(self._mappings[video_id], scores, strict=True)):
                value = float(score)
                previous = best_by_frame.get(mapping.frame_id)
                if previous is None or value > previous[0] or (value == previous[0] and row_index < previous[1]):
                    best_by_frame[mapping.frame_id] = (value, row_index, mapping)
            candidates.extend(
                (score, video_id, mapping.frame_id, row_index, mapping)
                for score, row_index, mapping in best_by_frame.values()
            )
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        return [
            SearchCandidate(rank, video_id, frame_id, mapping.keyframe_id, score, mapping.image_path, row_index)
            for rank, (score, video_id, frame_id, row_index, mapping) in enumerate(candidates[:top_k], start=1)
        ]

    def _load_mapping(self, video_id: str) -> tuple[_MappingRow, ...]:
        path = self.mapping_root / f"{video_id}.csv"
        if not path.is_file():
            raise ValueError(f"missing mapping CSV: {path}")
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
                raise ValueError(f"invalid mapping columns: {path}")
            for line_number, raw in enumerate(reader, start=2):
                try:
                    n = int(raw["n"])
                    frame_id = int(raw["frame_idx"])
                except (TypeError, ValueError):
                    raise ValueError(f"invalid mapping row {line_number}: {path}") from None
                if n < 0 or frame_id < 0:
                    raise ValueError(f"negative mapping value at row {line_number}: {path}")
                keyframe_id = f"{n:03d}"
                image_path = self.keyframe_root / video_id / f"{keyframe_id}.jpg"
                if not image_path.is_file():
                    raise ValueError(f"missing keyframe: {image_path}")
                rows.append(_MappingRow(frame_id, keyframe_id, image_path))
        return tuple(rows)


class DenseRetrievalService:
    def __init__(self, index: OfficialClipShardIndex, encoder: TextEncoder) -> None:
        self._index = index
        self._encoder = encoder

    def search(
        self,
        query: str,
        *,
        top_k: int = 100,
        video_ids: Sequence[str] | None = None,
    ) -> list[SearchCandidate]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        return self._index.search(self._encoder.encode(query.strip()), top_k=top_k, video_ids=video_ids)


def fuse_variant_rankings(
    rankings: Sequence[Sequence[SearchCandidate]],
    *,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[FusedSearchCandidate]:
    """Fuse independent query rankings using rank-only RRF.

    An identity contributes at most once per variant. Duplicate occurrences in
    one variant deterministically keep the lowest source rank, then the highest
    cosine score. Cosine scores are retained only as diagnostics.
    """
    _validate_top_k(top_k)
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    aggregated: dict[tuple[str, int], tuple[SearchCandidate, dict[int, VariantMatch]]] = {}
    for variant_index, ranking in enumerate(rankings):
        per_variant: dict[tuple[str, int], SearchCandidate] = {}
        for candidate in ranking:
            identity = (candidate.video_id, candidate.frame_id)
            previous = per_variant.get(identity)
            if previous is None or (candidate.rank, -candidate.score) < (previous.rank, -previous.score):
                per_variant[identity] = candidate
        for identity, candidate in per_variant.items():
            if identity not in aggregated:
                aggregated[identity] = (candidate, {})
            representative, matches = aggregated[identity]
            if (candidate.rank, -candidate.score, candidate.keyframe_id) < (
                representative.rank,
                -representative.score,
                representative.keyframe_id,
            ):
                representative = candidate
                aggregated[identity] = (representative, matches)
            matches[variant_index] = VariantMatch(variant_index, candidate.rank, candidate.score)
    fused = []
    for (video_id, frame_id), (representative, matches_by_variant) in aggregated.items():
        matches = tuple(matches_by_variant[index] for index in sorted(matches_by_variant))
        score = sum(1.0 / (rrf_k + match.rank) for match in matches)
        fused.append((score, min(match.rank for match in matches), video_id, frame_id, representative, matches))
    fused.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [
        FusedSearchCandidate(
            rank=rank,
            video_id=video_id,
            frame_id=frame_id,
            keyframe_id=representative.keyframe_id,
            image_path=representative.image_path,
            feature_row=representative.feature_row,
            rrf_score=score,
            source_matches=matches,
        )
        for rank, (score, _, video_id, frame_id, representative, matches) in enumerate(fused[:top_k], start=1)
    ]


def _validate_top_k(top_k: int) -> None:
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer in [1, 100]")


def _normalize_video_ids(video_ids: Sequence[str]) -> frozenset[str]:
    if isinstance(video_ids, (str, bytes)):
        raise ValueError("video_ids must be a sequence of identifiers")
    normalized = frozenset(video_id.strip() for video_id in video_ids if isinstance(video_id, str) and video_id.strip())
    if len(normalized) != len(video_ids):
        raise ValueError("video_ids must contain unique non-blank strings")
    return normalized
