"""Run official BTC CLIP retrieval and write review artifacts outside Git."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import zipfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_search_results import render_results  # noqa: E402

from backend.services.batch1_retrieval import (  # noqa: E402
    FusedSearchCandidate,
    OfficialClipShardIndex,
    SearchCandidate,
    fuse_variant_rankings,
)
from backend.services.clip_text_encoder import OpenAIClipTextEncoder  # noqa: E402

CSV_FIELDS = [
    "query_filename",
    "query_text",
    "query_variants",
    "rank",
    "video_id",
    "keyframe_id",
    "frame_id",
    "score",
    "rrf_score",
    "source_ranks",
    "source_scores",
    "image_path",
    "feature_row",
]

_QUERY_PATTERN = re.compile(r"^query-p1-(\d+)-(kis|qa|trake)\.txt$")
_EVENT_LABEL_PATTERN = re.compile(r"^E\d+(?:\s*[:.\-]\s*|\s+)(.+)$", re.IGNORECASE)


def _query_sort_key(filename: str) -> tuple[int, str]:
    match = _QUERY_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"invalid query filename: {filename}")
    return int(match.group(1)), filename


def read_classified_queries(query_zip: Path) -> dict[str, dict[str, str]]:
    """Strictly read top-level UTF-8 queries grouped by filename suffix."""
    grouped: dict[str, dict[str, str]] = {"kis": {}, "qa": {}, "trake": {}}
    with zipfile.ZipFile(query_zip) as archive:
        seen: set[str] = set()
        for entry in archive.infolist():
            name = entry.filename
            if entry.is_dir() or Path(name).name != name or "/" in name or "\\" in name or ".." in Path(name).parts:
                raise ValueError(f"query ZIP contains unsafe or nested entry: {name}")
            if name in seen:
                raise ValueError(f"duplicate query entry: {name}")
            seen.add(name)
            match = _QUERY_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(f"unsupported query filename: {name}")
            grouped[match.group(2)][name] = archive.read(entry).decode("utf-8", errors="strict")
    return {
        task: dict(sorted(queries.items(), key=lambda item: _query_sort_key(item[0])))
        for task, queries in grouped.items()
    }


def parse_trake_events(text: str) -> tuple[str, ...]:
    """Preserve event order while stripping optional ``E<number>`` labels."""
    if not isinstance(text, str) or not text:
        raise ValueError("TRAKE text must not be empty")
    events = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            raise ValueError("TRAKE events must not be blank")
        match = _EVENT_LABEL_PATTERN.fullmatch(line)
        event = match.group(1).strip() if match else line
        if not event:
            raise ValueError("TRAKE events must not be blank")
        events.append(event)
    if not events:
        raise ValueError("TRAKE must contain at least one event")
    return tuple(events)


def read_query_zip(query_zip: Path, query_name: str) -> str:
    """Read one unique top-level ZIP entry as strict UTF-8 without extraction."""
    if not query_name or Path(query_name).name != query_name or "/" in query_name or "\\" in query_name:
        raise ValueError("query_name must be a top-level filename")
    with zipfile.ZipFile(query_zip) as archive:
        matches = [info for info in archive.infolist() if info.filename == query_name]
        if len(matches) != 1:
            raise ValueError(f"query ZIP must contain exactly one entry named {query_name!r}")
        entry = matches[0]
        if entry.is_dir():
            raise ValueError("query ZIP entry must be a file")
        return archive.read(entry).decode("utf-8", errors="strict")


def read_query_variants(path: Path, query_filename: str, original_query: str) -> tuple[str, ...]:
    """Load a selected query's immutable variants from strict UTF-8 JSON."""
    payload = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
    if not isinstance(payload, dict) or query_filename not in payload:
        raise ValueError("variants JSON must contain the selected query filename")
    record = payload[query_filename]
    if not isinstance(record, dict) or set(record) != {"original", "variants"}:
        raise ValueError("query variant record must contain only original and variants")
    if not isinstance(record["original"], str) or record["original"] != original_query:
        raise ValueError("variant original must exactly match the decoded ZIP query")
    raw_variants = record["variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("variants must be a non-empty list")
    variants = []
    for raw in raw_variants:
        if not isinstance(raw, str) or not (variant := raw.strip()):
            raise ValueError("each variant must be a non-blank string")
        variants.append(variant)
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique after stripping")
    return tuple(variants)


def validate_variant_coverage(path: Path, queries: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """Require exact variant keys and validate every record before model use."""
    payload = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
    if not isinstance(payload, dict) or set(payload) != set(queries):
        raise ValueError("variant keys must exactly match selected query filenames")
    return {name: read_query_variants(path, name, text) for name, text in queries.items()}


def read_kis_queries(query_zip: Path) -> dict[str, str]:
    return read_classified_queries(query_zip)["kis"]


def read_qa_queries(query_zip: Path) -> dict[str, str]:
    return read_classified_queries(query_zip)["qa"]


def write_candidates(
    csv_path: Path,
    json_path: Path,
    candidates: list[SearchCandidate] | list[FusedSearchCandidate],
    *,
    query_filename: str,
    query_text: str,
    query_variants: tuple[str, ...] = (),
    timing: list[dict[str, float | int | None]] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> None:
    """Persist immutable candidate values as CSV and JSON."""
    rows = []
    for candidate in candidates:
        row = asdict(candidate)  # type: ignore[arg-type]
        row["image_path"] = str(row["image_path"])
        row["query_filename"] = query_filename
        row["query_text"] = query_text
        row["query_variants"] = json.dumps(query_variants, ensure_ascii=False)
        if isinstance(candidate, FusedSearchCandidate):
            row["source_ranks"] = json.dumps(
                {str(match.variant_index): match.rank for match in candidate.source_matches}
            )
            row["source_scores"] = json.dumps(
                {str(match.variant_index): match.cosine_score for match in candidate.source_matches}
            )
            row.pop("source_matches")
            row["score"] = ""
        else:
            row["rrf_score"] = ""
            row["source_ranks"] = ""
            row["source_scores"] = ""
        rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "query_filename": query_filename,
        "query_text": query_text,
        "query_variants": list(query_variants),
        "timing": [] if timing is None else timing,
        "provenance": {} if provenance is None else dict(provenance),
        "candidates": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_resume_json(
    path: Path,
    query_filename: str,
    query_text: str,
    variants: tuple[str, ...],
    provenance: Mapping[str, object],
) -> bool:
    try:
        payload = json.loads(path.read_bytes().decode("utf-8", errors="strict"))
        candidates = payload["candidates"]
        identities = {(item["video_id"], item["frame_id"]) for item in candidates}
        return (
            payload["query_filename"] == query_filename
            and payload["query_text"] == query_text
            and payload["query_variants"] == list(variants)
            and payload["provenance"] == dict(provenance)
            and len(candidates) == 100
            and len(identities) == 100
            and [item["rank"] for item in candidates] == list(range(1, 101))
            and all(Path(item["image_path"]).is_file() for item in candidates)
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        return False


def _search_variants(
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    variants: tuple[str, ...],
    *,
    top_k: int,
) -> tuple[list[FusedSearchCandidate], list[dict[str, float | int | None]]]:
    rankings = []
    timings: list[dict[str, float | int | None]] = []
    for variant_index, variant in enumerate(variants):
        encode_started = time.perf_counter()
        query_vector = encoder.encode(variant)
        encode_seconds = time.perf_counter() - encode_started
        search_started = time.perf_counter()
        rankings.append(index.search(query_vector, top_k=100))
        search_seconds = time.perf_counter() - search_started
        timings.append(
            {"variant_index": variant_index, "encode_seconds": encode_seconds, "search_seconds": search_seconds}
        )
    return fuse_variant_rankings(rankings, top_k=top_k), timings


def _render_batch_index(output_root: Path, entries: list[dict[str, object]]) -> None:
    cards = []
    for entry in entries:
        stem = Path(str(entry["query_filename"])).stem
        query = html.escape(str(entry["query_text"]), quote=True)
        status = html.escape(str(entry["status"]), quote=True)
        image_path = Path(str(entry.get("top_image_path", "")))
        image_uri = html.escape(image_path.resolve().as_uri(), quote=True) if image_path.is_file() else ""
        cards.append(
            f'<article><a href="{html.escape(stem + ".html", quote=True)}"><h2>{html.escape(stem)}</h2></a>'
            f'<img src="{image_uri}" alt=""><p>{query}</p><b>{status}</b> '
            f"candidates={entry.get('candidate_count', 0)}</article>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>KIS batch</title>
<style>body{{font:14px system-ui}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}img{{width:100%;height:160px;object-fit:contain}}</style></head>
<body><h1>KIS batch</h1><main>{"".join(cards)}</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def run_kis_batch(
    *,
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    queries: Mapping[str, str],
    variants_path: Path,
    output_root: Path,
    resume: bool,
    overwrite: bool,
    query_zip: Path | None = None,
) -> dict[str, object]:
    """Run an isolated KIS batch while recording clean per-query failures."""
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    entries: list[dict[str, object]] = []
    raw_variant_payload = json.loads(variants_path.read_bytes().decode("utf-8", errors="strict"))
    if not isinstance(raw_variant_payload, dict) or set(raw_variant_payload) != set(queries):
        raise ValueError("variant keys must exactly match selected query filenames")
    validated_variants: dict[str, tuple[str, ...] | Exception] = {}
    for query_filename, query_text in queries.items():
        try:
            validated_variants[query_filename] = read_query_variants(variants_path, query_filename, query_text)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            validated_variants[query_filename] = error
    zip_hash = _sha256_file(query_zip) if query_zip is not None else None
    variants_hash = _sha256_file(variants_path)
    for query_filename, query_text in queries.items():
        stem = Path(query_filename).stem
        csv_path = output_root / f"{stem}.csv"
        json_path = output_root / f"{stem}.json"
        html_path = output_root / f"{stem}.html"
        entry: dict[str, object] = {"query_filename": query_filename, "query_text": query_text}
        try:
            variants_or_error = validated_variants[query_filename]
            if isinstance(variants_or_error, Exception):
                raise variants_or_error
            variants = variants_or_error
            provenance = {
                "query_zip_sha256": zip_hash,
                "original_text_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                "variants_sha256": variants_hash,
                "model": "OpenAI CLIP ViT-B/32",
                "feature_dimension": 512,
                "variant_top_k": 100,
                "final_top_k": 100,
                "rrf_k": 60,
            }
            existing = any(path.exists() for path in (csv_path, json_path, html_path))
            if resume and _valid_resume_json(json_path, query_filename, query_text, variants, provenance):
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                entry.update(
                    status="skipped",
                    candidate_count=100,
                    top_image_path=payload["candidates"][0]["image_path"],
                    output_paths={"csv": str(csv_path), "json": str(json_path), "html": str(html_path)},
                )
                entries.append(entry)
                continue
            if existing and not (resume or overwrite):
                raise ValueError("outputs already exist; use --resume or --overwrite")
            candidates, timings = _search_variants(index, encoder, variants, top_k=100)
            if len(candidates) != 100:
                raise ValueError(f"retrieval returned {len(candidates)} candidates, expected 100")
            write_candidates(
                csv_path,
                json_path,
                candidates,
                query_filename=query_filename,
                query_text=query_text,
                query_variants=variants,
                timing=timings,
                provenance=provenance,
            )
            render_results(csv_path, html_path, limit=20)
            entry.update(
                status="success",
                candidate_count=len(candidates),
                top_image_path=str(candidates[0].image_path),
                timing=timings,
                output_paths={"csv": str(csv_path), "json": str(json_path), "html": str(html_path)},
            )
        except Exception as error:  # batch isolation boundary
            entry.update(status="failure", candidate_count=0, error=f"{type(error).__name__}: {error}")
        entries.append(entry)
    _render_batch_index(output_root, entries)
    classified = (
        read_classified_queries(query_zip) if query_zip is not None else {"kis": dict(queries), "qa": {}, "trake": {}}
    )
    report = {
        "total_kis": len(queries),
        "succeeded": sum(entry["status"] == "success" for entry in entries),
        "failed": sum(entry["status"] == "failure" for entry in entries),
        "skipped": sum(entry["status"] == "skipped" for entry in entries),
        "total_seconds": time.perf_counter() - started,
        "model": "OpenAI CLIP ViT-B/32",
        "checkpoint": "official OpenAI ViT-B/32",
        "feature_dimension": 512,
        "query_zip_filename": None if query_zip is None else query_zip.name,
        "query_zip_sha256": zip_hash,
        "query_counts": {task: len(items) for task, items in classified.items()},
        "variants_sha256": variants_hash,
        "output_root": str(output_root),
        "queries": entries,
    }
    (output_root / "batch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _write_vlm_manifest(path: Path, question: str, candidates: list[FusedSearchCandidate]) -> None:
    manifest = {
        "question": question,
        "candidates": [
            {
                "candidate_rank": candidate.rank,
                "video_id": candidate.video_id,
                "frame_id": candidate.frame_id,
                "keyframe_id": candidate.keyframe_id,
                "image_path": str(candidate.image_path.resolve()),
                "retrieval_score": candidate.rrf_score,
            }
            for candidate in candidates
        ],
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_qa_batch(
    *,
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    queries: Mapping[str, str],
    variants_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Retrieve QA candidates and explicit-path VLM manifests without invoking a VLM."""
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    entries: list[dict[str, object]] = []
    validated_variants: dict[str, tuple[str, ...] | Exception] = {}
    for query_filename, query_text in queries.items():
        try:
            validated_variants[query_filename] = read_query_variants(variants_path, query_filename, query_text)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            validated_variants[query_filename] = error
    for query_filename, query_text in queries.items():
        stem = Path(query_filename).stem
        csv_path = output_root / f"{stem}.csv"
        json_path = output_root / f"{stem}.json"
        html_path = output_root / f"{stem}.html"
        manifest_path = output_root / f"{stem}-vlm-input.json"
        entry: dict[str, object] = {"query_filename": query_filename, "query_text": query_text}
        try:
            if any(path.exists() for path in (csv_path, json_path, html_path, manifest_path)):
                raise ValueError("QA output already exists")
            variants_or_error = validated_variants[query_filename]
            if isinstance(variants_or_error, Exception):
                raise variants_or_error
            variants = variants_or_error
            candidates, timings = _search_variants(index, encoder, variants, top_k=30)
            if len(candidates) != 30:
                raise ValueError(f"retrieval returned {len(candidates)} candidates, expected 30")
            write_candidates(
                csv_path,
                json_path,
                candidates,
                query_filename=query_filename,
                query_text=query_text,
                query_variants=variants,
                timing=timings,
            )
            render_results(csv_path, html_path, limit=30)
            _write_vlm_manifest(manifest_path, query_text, candidates)
            entry.update(
                status="success",
                candidate_count=30,
                top_image_path=str(candidates[0].image_path),
                timing=timings,
                output_paths={
                    "csv": str(csv_path),
                    "json": str(json_path),
                    "html": str(html_path),
                    "vlm_manifest": str(manifest_path),
                },
            )
        except Exception as error:  # batch isolation boundary
            entry.update(status="failure", candidate_count=0, error=f"{type(error).__name__}: {error}")
        entries.append(entry)
    _render_batch_index(output_root, entries)
    report = {
        "total_qa": len(queries),
        "succeeded": sum(entry["status"] == "success" for entry in entries),
        "failed": sum(entry["status"] == "failure" for entry in entries),
        "total_seconds": time.perf_counter() - started,
        "model": "OpenAI CLIP ViT-B/32",
        "feature_dimension": 512,
        "vlm_loaded": False,
        "output_root": str(output_root),
        "queries": entries,
    }
    (output_root / "batch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official OpenAI CLIP retrieval-only baseline")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    query_source = parser.add_mutually_exclusive_group()
    query_source.add_argument("--query")
    query_source.add_argument("--query-zip", type=Path)
    parser.add_argument("--query-name")
    parser.add_argument("--query-variants-file", type=Path)
    batch_mode = parser.add_mutually_exclusive_group()
    batch_mode.add_argument("--batch-kis", action="store_true")
    batch_mode.add_argument("--batch-qa", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_qa:
        if args.query_zip is None or args.query_variants_file is None or args.output_root is None:
            parser.error("--batch-qa requires --query-zip, --query-variants-file, and --output-root")
        if (
            args.query is not None
            or args.query_name is not None
            or args.output_csv is not None
            or args.output_json is not None
        ):
            parser.error("batch mode does not accept single-query arguments")
        queries = read_qa_queries(args.query_zip)
        if not queries:
            parser.error("query ZIP contains no QA queries")
        report = run_qa_batch(
            index=OfficialClipShardIndex(args.features, args.mappings, args.keyframes),
            encoder=OpenAIClipTextEncoder(model_cache=args.model_cache),
            queries=queries,
            variants_path=args.query_variants_file,
            output_root=args.output_root,
        )
        sys.stdout.write(json.dumps(report, ensure_ascii=True) + "\n")
        if report["failed"]:
            raise SystemExit(1)
        return
    if args.batch_kis:
        if args.query_zip is None or args.query_variants_file is None or args.output_root is None:
            parser.error("--batch-kis requires --query-zip, --query-variants-file, and --output-root")
        if (
            args.query is not None
            or args.query_name is not None
            or args.output_csv is not None
            or args.output_json is not None
        ):
            parser.error("batch mode does not accept single-query arguments")
        queries = read_kis_queries(args.query_zip)
        if not queries:
            parser.error("query ZIP contains no KIS queries")
        report = run_kis_batch(
            index=OfficialClipShardIndex(args.features, args.mappings, args.keyframes),
            encoder=OpenAIClipTextEncoder(model_cache=args.model_cache),
            queries=queries,
            variants_path=args.query_variants_file,
            output_root=args.output_root,
            resume=args.resume,
            overwrite=args.overwrite,
            query_zip=args.query_zip,
        )
        sys.stdout.write(json.dumps(report, ensure_ascii=True) + "\n")
        if report["failed"]:
            raise SystemExit(1)
        return
    if args.output_csv is None or (args.query is None and args.query_zip is None):
        parser.error("single-query mode requires a query source and --output-csv")
    if args.query_zip is not None:
        if not args.query_name:
            parser.error("--query-name is required with --query-zip")
        query = read_query_zip(args.query_zip, args.query_name)
        query_filename = args.query_name
    else:
        if args.query_name:
            parser.error("--query-name may only be used with --query-zip")
        query = args.query
        query_filename = "<command-line>"
    json_path = args.output_json or args.output_csv.with_suffix(".json")
    index = OfficialClipShardIndex(args.features, args.mappings, args.keyframes)
    encoder = OpenAIClipTextEncoder(model_cache=args.model_cache)
    variants = ()
    timings = []
    if args.query_variants_file is not None:
        if args.query_zip is None:
            parser.error("--query-variants-file requires --query-zip and --query-name")
        variants = read_query_variants(args.query_variants_file, query_filename, query)
        rankings = []
        for variant_index, variant in enumerate(variants):
            encode_started = time.perf_counter()
            query_vector = encoder.encode(variant)
            encode_seconds = time.perf_counter() - encode_started
            search_started = time.perf_counter()
            rankings.append(index.search(query_vector, top_k=100))
            search_seconds = time.perf_counter() - search_started
            timings.append(
                {"variant_index": variant_index, "encode_seconds": encode_seconds, "search_seconds": search_seconds}
            )
        candidates = fuse_variant_rankings(rankings, top_k=args.top_k)
    else:
        encode_started = time.perf_counter()
        query_vector = encoder.encode(query)
        encode_seconds = time.perf_counter() - encode_started
        search_started = time.perf_counter()
        candidates = index.search(query_vector, top_k=args.top_k)
        search_seconds = time.perf_counter() - search_started
        timings.append({"variant_index": None, "encode_seconds": encode_seconds, "search_seconds": search_seconds})
    write_candidates(
        args.output_csv,
        json_path,
        candidates,
        query_filename=query_filename,
        query_text=query,
        query_variants=variants,
        timing=timings,
    )
    sys.stdout.write(
        json.dumps(
            {
                "results": len(candidates),
                "csv": str(args.output_csv),
                "json": str(json_path),
                "timings": timings,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
