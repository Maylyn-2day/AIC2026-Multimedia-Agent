"""Deterministic weighted Reciprocal Rank Fusion for retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Real
from typing import TypeAlias

Candidate: TypeAlias = Mapping[str, object]
CandidateIdentity: TypeAlias = tuple[object, object]


def weighted_rrf(
    dense_candidates: Sequence[Candidate],
    sparse_candidates: Sequence[Candidate],
    *,
    k: int = 60,
    weight_dense: float = 1.0,
    weight_sparse: float = 1.0,
    top_k: int = 100,
) -> list[dict[str, object]]:
    """Fuse dense and sparse rankings using weighted Reciprocal Rank Fusion.

    Candidates are identified by ``(video_id, frame_id)`` and must provide a
    one-based integer ``rank``. A candidate absent from one source receives no
    contribution from that source.

    Duplicate identities within one source are resolved deterministically by
    keeping the occurrence with the smallest rank. If ranks are equal, the
    first occurrence is kept. After descending RRF score, ties are ordered by
    a normalized representation of ``(video_id, frame_id)``.

    Dense metadata takes precedence when both sources provide the same key;
    sparse metadata fills keys missing from the dense candidate. Inputs and
    their nested values are never mutated or exposed by reference in outputs.
    """
    _validate_parameters(k, weight_dense, weight_sparse, top_k)
    dense_by_id = _deduplicate(dense_candidates, "dense")
    sparse_by_id = _deduplicate(sparse_candidates, "sparse")

    fused: list[dict[str, object]] = []
    for identity in dense_by_id.keys() | sparse_by_id.keys():
        dense = dense_by_id.get(identity)
        sparse = sparse_by_id.get(identity)
        dense_rank = dense["rank"] if dense is not None else None
        sparse_rank = sparse["rank"] if sparse is not None else None

        rrf_score = 0.0
        if dense_rank is not None:
            rrf_score += float(weight_dense) / (k + dense_rank)
        if sparse_rank is not None:
            rrf_score += float(weight_sparse) / (k + sparse_rank)

        result = _merge_metadata(dense, sparse)
        result.update(
            {
                "video_id": deepcopy(identity[0]),
                "frame_id": deepcopy(identity[1]),
                "dense_score": _source_score(dense, "dense"),
                "dense_rank": dense_rank,
                "sparse_score": _source_score(sparse, "sparse"),
                "sparse_rank": sparse_rank,
                "rrf_score": rrf_score,
                "rrf_k": k,
            }
        )
        fused.append(result)

    fused.sort(key=lambda candidate: (-float(candidate["rrf_score"]), _identity_sort_key(candidate)))
    selected = fused[:top_k]
    for output_rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = output_rank
    return selected


def _validate_parameters(k: int, weight_dense: float, weight_sparse: float, top_k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be an integer greater than 0")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer in [1, 100]")

    for name, weight in (("weight_dense", weight_dense), ("weight_sparse", weight_sparse)):
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"{name} must be a finite number greater than or equal to 0")
    if weight_dense == 0 and weight_sparse == 0:
        raise ValueError("weight_dense and weight_sparse cannot both be 0")


def _deduplicate(candidates: Sequence[Candidate], source: str) -> dict[CandidateIdentity, dict[str, object]]:
    unique: dict[CandidateIdentity, dict[str, object]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{source} candidate must be a mapping")
        missing_fields = [field for field in ("video_id", "frame_id", "rank") if field not in candidate]
        if missing_fields:
            raise ValueError(f"{source} candidate is missing required field(s): {', '.join(missing_fields)}")

        frame_id = candidate["frame_id"]
        if type(frame_id) is not int:
            raise ValueError(f"{source} candidate frame_id must be an integer")
        rank = candidate["rank"]
        if type(rank) is not int or rank <= 0:
            raise ValueError(f"{source} candidate rank must be an integer starting from 1")

        identity = (candidate["video_id"], frame_id)
        try:
            existing = unique.get(identity)
        except TypeError as error:
            raise ValueError("video_id and frame_id must be hashable") from error
        if existing is None or rank < existing["rank"]:
            unique[identity] = deepcopy(dict(candidate))
    return unique


def _merge_metadata(dense: dict[str, object] | None, sparse: dict[str, object] | None) -> dict[str, object]:
    reserved = {
        "rank",
        "score",
        "dense_score",
        "dense_rank",
        "sparse_score",
        "sparse_rank",
        "rrf_score",
        "rrf_k",
    }
    result: dict[str, object] = {}
    for candidate in (dense, sparse):
        if candidate is not None:
            for key, value in candidate.items():
                if key not in reserved and key not in result:
                    result[key] = deepcopy(value)
    return result


def _source_score(candidate: dict[str, object] | None, source: str) -> object:
    if candidate is None:
        return None
    return deepcopy(candidate.get(f"{source}_score", candidate.get("score")))


def _identity_sort_key(candidate: dict[str, object]) -> tuple[str, str, str]:
    frame_id = candidate["frame_id"]
    return (str(candidate["video_id"]), type(frame_id).__qualname__, repr(frame_id))
