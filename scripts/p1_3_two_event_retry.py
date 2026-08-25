"""Retrieve and review the two ordered visual events in official query p1-3."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.batch1_retrieval import OfficialClipShardIndex, SearchCandidate  # noqa: E402
from backend.services.clip_text_encoder import OpenAIClipTextEncoder  # noqa: E402


@dataclass(frozen=True, slots=True)
class EventCandidate:
    video_id: str
    frame_id: int
    keyframe_id: str
    image_path: Path
    feature_row: int
    rrf_score: float
    source_ranks: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class EventPair:
    rank: int
    video_id: str
    event_a: EventCandidate
    event_b: EventCandidate
    timestamp_a: float
    timestamp_b: float
    temporal_gap: float
    temporal_bonus: float
    pair_score: float


def read_event_variants(path: Path) -> tuple[str, ...]:
    """Read a strict UTF-8 JSON array of unique, non-blank variants."""
    payload = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("event variants must be a non-empty JSON array")
    variants = []
    for raw in payload:
        if not isinstance(raw, str) or not (variant := raw.strip()):
            raise ValueError("event variants must contain only non-blank strings")
        variants.append(variant)
    if len(set(variants)) != len(variants):
        raise ValueError("event variants must be unique after stripping")
    return tuple(variants)


def fuse_event_rankings(
    rankings: Sequence[Sequence[SearchCandidate]], *, top_k: int = 1000, rrf_k: int = 60
) -> list[EventCandidate]:
    """Fuse variant ranks without adding cosine scores or mutating rankings."""
    if type(top_k) is not int or top_k <= 0 or type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("top_k and rrf_k must be positive integers")
    sources: dict[tuple[str, int], dict[int, SearchCandidate]] = {}
    for source_index, ranking in enumerate(rankings):
        per_source: dict[tuple[str, int], SearchCandidate] = {}
        for candidate in ranking:
            identity = (candidate.video_id, candidate.frame_id)
            previous = per_source.get(identity)
            if previous is None or (candidate.rank, -candidate.score) < (previous.rank, -previous.score):
                per_source[identity] = candidate
        for identity, candidate in per_source.items():
            sources.setdefault(identity, {})[source_index] = candidate
    fused = []
    for (video_id, frame_id), matches in sources.items():
        representative = min(matches.values(), key=lambda item: (item.rank, -item.score, item.keyframe_id))
        source_ranks = tuple(matches[index].rank if index in matches else None for index in range(len(rankings)))
        score = sum(1.0 / (rrf_k + rank) for rank in source_ranks if rank is not None)
        fused.append(
            EventCandidate(
                video_id,
                frame_id,
                representative.keyframe_id,
                representative.image_path,
                representative.feature_row,
                score,
                source_ranks,
            )
        )
    fused.sort(key=lambda item: (-item.rrf_score, item.video_id, item.frame_id, item.keyframe_id))
    return fused[:top_k]


def pair_ordered_events(
    event_a: Sequence[EventCandidate],
    event_b: Sequence[EventCandidate],
    timestamps: dict[tuple[str, int], float],
    *,
    top_k: int = 50,
) -> list[EventPair]:
    """Join same-video A-before-B candidates with a deterministic gap bonus."""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    b_by_video: dict[str, list[EventCandidate]] = {}
    for candidate in event_b:
        b_by_video.setdefault(candidate.video_id, []).append(candidate)
    scored = []
    for left in event_a:
        for right in b_by_video.get(left.video_id, ()):
            if left.frame_id >= right.frame_id:
                continue
            timestamp_a = timestamps[(left.video_id, left.feature_row)]
            timestamp_b = timestamps[(right.video_id, right.feature_row)]
            gap = timestamp_b - timestamp_a
            if not math.isfinite(gap) or gap <= 0:
                continue
            bonus = 1.0 / (60.0 + gap)
            score = left.rrf_score + right.rrf_score + bonus
            scored.append((-score, gap, left.video_id, left.frame_id, right.frame_id, left, right, bonus))
    scored.sort(key=lambda item: item[:5])
    return [
        EventPair(
            rank,
            video_id,
            left,
            right,
            timestamps[(video_id, left.feature_row)],
            timestamps[(video_id, right.feature_row)],
            gap,
            bonus,
            -negative_score,
        )
        for rank, (negative_score, gap, video_id, _, _, left, right, bonus) in enumerate(scored[:top_k], start=1)
    ]


def load_timestamps(
    mapping_root: Path, video_ids: set[str]
) -> tuple[dict[tuple[str, int], float], dict[str, list[dict[str, str]]]]:
    timestamps = {}
    mappings = {}
    for video_id in sorted(video_ids):
        with (mapping_root / f"{video_id}.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        mappings[video_id] = rows
        for row_index, row in enumerate(rows):
            timestamps[(video_id, row_index)] = float(row["pts_time"])
    return timestamps, mappings


def _write_neighborhood(path: Path, candidate: EventCandidate, rows: list[dict[str, str]], keyframe_root: Path) -> None:
    start = max(0, candidate.feature_row - 15)
    stop = min(len(rows), candidate.feature_row + 16)
    cards = []
    for row_index in range(start, stop):
        row = rows[row_index]
        keyframe_id = f"{int(row['n']):03d}"
        image = keyframe_root / candidate.video_id / f"{keyframe_id}.jpg"
        uri = html.escape(image.resolve().as_uri(), quote=True)
        marker = "★ EVENT A — " if row_index == candidate.feature_row else ""
        cards.append(
            f'<article class="{"candidate" if row_index == candidate.feature_row else ""}"><h2>{marker}row {row_index}</h2>'
            f'<a href="{uri}"><img src="{uri}"></a><p>keyframe={keyframe_id}; frame={html.escape(row["frame_idx"])}; '
            f"pts_time={html.escape(row['pts_time'])}</p></article>"
        )
    path.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><style>body{font:15px system-ui;background:#111;color:#eee}'
        "article{margin:18px;border:2px solid #555;padding:10px}.candidate{border:6px solid #fc0}img{width:600px;max-width:100%}"
        f"</style></head><body><h1>Event A neighborhood — {html.escape(candidate.video_id)}</h1>{''.join(cards)}</body></html>",
        encoding="utf-8",
    )


def write_review(
    output_root: Path, pairs: list[EventPair], mappings: dict[str, list[dict[str, str]]], keyframe_root: Path
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    neighborhood_root = output_root / "event_a_neighborhoods"
    neighborhood_root.mkdir(exist_ok=True)
    rows = []
    for pair in pairs:
        neighborhood = neighborhood_root / f"pair-{pair.rank:02d}.html"
        _write_neighborhood(neighborhood, pair.event_a, mappings[pair.video_id], keyframe_root)
        left_uri = html.escape(pair.event_a.image_path.resolve().as_uri(), quote=True)
        right_uri = html.escape(pair.event_b.image_path.resolve().as_uri(), quote=True)
        rows.append(
            f"<article><h2>Pair {pair.rank} — {html.escape(pair.video_id)} — score={pair.pair_score:.9f}</h2><div>"
            f'<section><h3>Event A</h3><a href="{left_uri}"><img src="{left_uri}"></a>'
            f"<p>frame={pair.event_a.frame_id}; keyframe={html.escape(pair.event_a.keyframe_id)}; timestamp={pair.timestamp_a:.3f}; RRF={pair.event_a.rrf_score:.9f}</p>"
            f'<a href="{html.escape(neighborhood.relative_to(output_root).as_posix(), quote=True)}">±15 keyframe neighborhood</a></section>'
            f'<section><h3>Event B</h3><a href="{right_uri}"><img src="{right_uri}"></a>'
            f"<p>frame={pair.event_b.frame_id}; keyframe={html.escape(pair.event_b.keyframe_id)}; timestamp={pair.timestamp_b:.3f}; RRF={pair.event_b.rrf_score:.9f}</p>"
            f"</section></div><p>gap={pair.temporal_gap:.3f}s; temporal bonus={pair.temporal_bonus:.9f}. No OCR or inferred answer.</p></article>"
        )
    review = output_root / "query-p1-3-two-event-top50.html"
    review.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>p1-3 two-event retry</title><style>'
        "body{font:15px system-ui;background:#111;color:#eee}article{border:2px solid #555;padding:12px;margin:18px 0}"
        "article>div{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:500px;max-width:100%;height:auto}a{color:#9cf}"
        f"</style></head><body><h1>P1-3 two-event retry</h1><p>No OCR and no inferred answer.</p>{''.join(rows)}</body></html>",
        encoding="utf-8",
    )
    return review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-a-variants", type=Path, required=True)
    parser.add_argument("--event-b-variants", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    index = OfficialClipShardIndex(args.features, args.mappings, args.keyframes)
    encoder = OpenAIClipTextEncoder(model_cache=args.model_cache)
    event_a_variants = read_event_variants(args.event_a_variants)
    event_b_variants = read_event_variants(args.event_b_variants)
    rankings_a = [index.search_pool(encoder.encode(query), top_k=500) for query in event_a_variants]
    rankings_b = [index.search_pool(encoder.encode(query), top_k=500) for query in event_b_variants]
    event_a = fuse_event_rankings(rankings_a)
    event_b = fuse_event_rankings(rankings_b)
    common_videos = {item.video_id for item in event_a} & {item.video_id for item in event_b}
    timestamps, mappings = load_timestamps(args.mappings, common_videos)
    pairs = pair_ordered_events(event_a, event_b, timestamps)
    if not pairs:
        raise ValueError("no ordered same-video event pairs found")
    review = write_review(args.output, pairs, mappings, args.keyframes)
    payload = {
        "event_a_variants": event_a_variants,
        "event_b_variants": event_b_variants,
        "videos_with_both_events": len(common_videos),
        "pairs": [asdict(pair) for pair in pairs],
        "elapsed_seconds": time.perf_counter() - started,
    }
    for pair in payload["pairs"]:
        pair["event_a"]["image_path"] = str(pair["event_a"]["image_path"])
        pair["event_b"]["image_path"] = str(pair["event_b"]["image_path"])
    (args.output / "query-p1-3-two-event-top50.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sys.stdout.write(
        json.dumps(
            {
                "videos_with_both_events": len(common_videos),
                "pairs": len(pairs),
                "html": str(review),
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
