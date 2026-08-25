"""Build the emergency official-round submission without rerunning KIS retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_search_results import render_results  # noqa: E402
from scripts.search_batch1 import (  # noqa: E402
    parse_trake_events,
    read_classified_queries,
    validate_variant_coverage,
    write_candidates,
)

from backend.services.batch1_retrieval import (  # noqa: E402
    OfficialClipShardIndex,
    SearchCandidate,
    fuse_variant_rankings,
)
from backend.services.clip_text_encoder import OpenAIClipTextEncoder  # noqa: E402


def _render_trake(path: Path, query: str, sequences: list[dict[str, object]]) -> None:
    cards = []
    for sequence in sequences:
        images = "".join(
            f'<figure><img src="{html.escape(Path(item["image_path"]).resolve().as_uri(), quote=True)}">'
            f"<figcaption>frame={item['frame_id']}; source rank={item['source_rank']}</figcaption></figure>"
            for item in sequence["events"]
        )
        cards.append(
            f"<article><h2>Sequence {sequence['rank']} — {sequence['score']:.9f}</h2><div>{images}</div></article>"
        )
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><title>TRAKE review</title>'
        "<style>body{font:14px system-ui}article div{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}"
        "img{width:100%;height:180px;object-fit:contain}</style></head><body>"
        f"<h1>{html.escape(query)}</h1>{''.join(cards)}</body></html>"
    )
    path.write_text(document, encoding="utf-8")


def _join_sequences(rankings: list[list[SearchCandidate]], top_k: int = 20) -> list[dict[str, object]]:
    by_event: list[dict[str, list[SearchCandidate]]] = []
    for ranking in rankings:
        grouped: dict[str, list[SearchCandidate]] = {}
        for candidate in ranking:
            grouped.setdefault(candidate.video_id, []).append(candidate)
        by_event.append(grouped)
    common_videos = set.intersection(*(set(grouped) for grouped in by_event))
    scored = []
    for video_id in sorted(common_videos):
        for choices in itertools.product(*(grouped[video_id] for grouped in by_event)):
            frames = tuple(item.frame_id for item in choices)
            if all(left < right for left, right in itertools.pairwise(frames)):
                source_ranks = tuple(item.rank for item in choices)
                score = sum(1.0 / (60 + rank) for rank in source_ranks)
                scored.append((-score, source_ranks, video_id, frames, choices))
    scored.sort(key=lambda item: item[:4])
    return [
        {
            "rank": rank,
            "score": -neg_score,
            "video_id": video_id,
            "frame_ids": list(frames),
            "events": [
                {
                    "frame_id": item.frame_id,
                    "keyframe_id": item.keyframe_id,
                    "image_path": str(item.image_path.resolve()),
                    "source_rank": item.rank,
                    "cosine_score": item.score,
                }
                for item in choices
            ],
        }
        for rank, (neg_score, _, video_id, frames, choices) in enumerate(scored[:top_k], start=1)
    ]


def _package(
    kis_root: Path,
    qa_root: Path,
    trake_json: Path,
    answers_path: Path,
    output_root: Path,
    final_zip: Path,
    trake_filename: str,
    qa_filenames: set[str],
) -> None:
    submission = output_root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    kis_files = sorted(kis_root.glob("*-kis.json"))
    if len(kis_files) != 20:
        raise ValueError(f"expected 20 KIS JSON files, found {len(kis_files)}")
    for source in kis_files:
        payload = json.loads(source.read_text(encoding="utf-8"))
        candidates = payload["candidates"]
        identities = [(item["video_id"], item["frame_id"]) for item in candidates]
        if len(candidates) != 100 or len(set(identities)) != 100:
            raise ValueError(f"invalid KIS candidates: {source}")
        with (submission / f"{source.stem}.csv").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(identities)
    for filename, record in answers.items():
        if filename not in qa_filenames or set(record) != {"candidate_rank", "answer"}:
            raise ValueError("invalid manual answer record")
        answer = record["answer"].strip() if isinstance(record["answer"], str) else ""
        rank = record["candidate_rank"]
        if type(rank) is not int or rank < 1 or not answer or len(answer) > 100:
            raise ValueError("invalid manual QA answer")
        payload = json.loads((qa_root / f"{Path(filename).stem}.json").read_text(encoding="utf-8"))
        candidate = payload["candidates"][rank - 1]
        with (submission / f"{Path(filename).stem}.csv").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow((candidate["video_id"], candidate["frame_id"], answer))
    trake = json.loads(trake_json.read_text(encoding="utf-8"))["sequences"][0]
    with (submission / f"{Path(trake_filename).stem}.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow((trake["video_id"], *trake["frame_ids"]))
    files = sorted(submission.glob("*.csv"))
    if len(files) != 25:
        raise ValueError(f"expected 25 submission CSV files, found {len(files)}")
    final_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"submission/{path.name}")


def _validate_zip(final_zip: Path) -> dict[str, object]:
    with zipfile.ZipFile(final_zip) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        if bad is not None or len(names) != 25 or any(not name.startswith("submission/") for name in names):
            raise ValueError("invalid ZIP structure")
        counts = {"kis": 0, "qa": 0, "trake": 0}
        for name in names:
            task = Path(name).stem.rsplit("-", 1)[-1]
            rows = list(csv.reader(archive.read(name).decode("utf-8", errors="strict").splitlines()))
            expected_columns = {"kis": 2, "qa": 3}.get(task)
            if task not in counts or not rows:
                raise ValueError(f"invalid task or empty CSV: {name}")
            if expected_columns is not None and any(len(row) != expected_columns for row in rows):
                raise ValueError(f"invalid CSV shape: {name}")
            if task == "trake" and any(len(row) < 2 for row in rows):
                raise ValueError(f"invalid CSV shape: {name}")
            if rows[0][0] == "video_id" or any(not row[0].strip() for row in rows):
                raise ValueError(f"header or blank video ID: {name}")
            if task == "kis" and (len(rows) != 100 or len({tuple(row) for row in rows}) != 100):
                raise ValueError(f"invalid KIS rows: {name}")
            if task == "qa" and (len(rows) != 1 or not rows[0][2].strip()):
                raise ValueError(f"invalid QA row: {name}")
            if task == "trake" and (
                len(rows) != 1 or not all(int(a) < int(b) for a, b in itertools.pairwise(rows[0][1:]))
            ):
                raise ValueError(f"invalid TRAKE row: {name}")
            for row in rows:
                for frame in row[1:2] if task != "trake" else row[1:]:
                    if int(frame) < 0:
                        raise ValueError(f"invalid frame: {name}")
            counts[task] += 1
    digest = hashlib.sha256(final_zip.read_bytes()).hexdigest()
    return {"sha256": digest, "size_bytes": final_zip.stat().st_size, "csv_count": len(names), "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-zip", type=Path, required=True)
    parser.add_argument("--kis-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--qa-variants", type=Path, required=True)
    parser.add_argument("--trake-variants", type=Path, required=True)
    parser.add_argument("--final-zip", type=Path, required=True)
    args = parser.parse_args()
    queries = read_classified_queries(args.query_zip)
    if len(queries["kis"]) != 20 or len(queries["qa"]) != 4 or len(queries["trake"]) != 1:
        raise ValueError("submission query set must contain 20 KIS, 4 QA, and 1 TRAKE")
    trake_name, trake_text = next(iter(queries["trake"].items()))
    if len(parse_trake_events(trake_text)) != 4:
        raise ValueError("TRAKE query must contain exactly four events")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.answers.is_file():
        raise ValueError("manual answers file is required; fallback answers are not generated")
    qa_variants = validate_variant_coverage(args.qa_variants, queries["qa"])
    trake_variants = validate_variant_coverage(args.trake_variants, queries["trake"])[trake_name]
    if len(trake_variants) != len(parse_trake_events(trake_text)):
        raise ValueError("TRAKE variants must match the number of ordered events")
    index = OfficialClipShardIndex(args.features, args.mappings, args.keyframes)
    encoder = OpenAIClipTextEncoder(model_cache=args.model_cache)
    qa_root = args.output_root / "qa"
    qa_root.mkdir(exist_ok=True)
    timings: dict[str, object] = {}
    for filename, variants in qa_variants.items():
        started = time.perf_counter()
        stem = Path(filename).stem
        json_path = qa_root / f"{stem}.json"
        if not json_path.is_file() or len(json.loads(json_path.read_text(encoding="utf-8"))["candidates"]) != 20:
            rankings = [index.search(encoder.encode(variant), top_k=100) for variant in variants]
            candidates = fuse_variant_rankings(rankings, top_k=20, rrf_k=60)
            write_candidates(
                qa_root / f"{stem}.csv",
                json_path,
                candidates,
                query_filename=filename,
                query_text=queries["qa"][filename],
                query_variants=variants,
            )
            render_results(qa_root / f"{stem}.csv", qa_root / f"{stem}.html", limit=20)
        timings[filename] = time.perf_counter() - started
    trake_root = args.output_root / "trake"
    trake_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    rankings = [index.search_pool(encoder.encode(variant), top_k=300) for variant in trake_variants]
    sequences = _join_sequences(rankings)
    used_top_k = 300
    if not sequences:
        rankings = [index.search_pool(encoder.encode(variant), top_k=500) for variant in trake_variants]
        sequences = _join_sequences(rankings)
        used_top_k = 500
    if not sequences:
        raise ValueError("no valid TRAKE sequence at top 500")
    trake_path = trake_root / f"{Path(trake_name).stem}.json"
    trake_path.write_text(
        json.dumps(
            {
                "query_filename": trake_name,
                "query_text": trake_text,
                "variants": trake_variants,
                "source_top_k": used_top_k,
                "sequences": sequences,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _render_trake(trake_root / f"{Path(trake_name).stem}.html", trake_text, sequences)
    timings[trake_name] = time.perf_counter() - started
    _package(
        args.kis_root,
        qa_root,
        trake_path,
        args.answers,
        args.output_root,
        args.final_zip,
        trake_name,
        set(queries["qa"]),
    )
    result = _validate_zip(args.final_zip)
    result.update(
        zip=str(args.final_zip),
        qa_html=[str(qa_root / f"{Path(name).stem}.html") for name in qa_variants],
        trake_html=str(trake_root / f"{Path(trake_name).stem}.html"),
        timings=timings,
        trake_source_top_k=used_top_k,
    )
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
