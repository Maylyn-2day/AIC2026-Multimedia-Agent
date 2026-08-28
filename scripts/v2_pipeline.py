"""Multi-task batch submission orchestrator for AIC2026 V2 pipeline.

Processes all KIS / QA / TRAKE tasks from query.zip in one unattended run
and packages a competition-compliant submission.zip.

Usage (Kaggle)::

    python scripts/v2_pipeline.py \\
        --features /kaggle/input/clip-features-32 \\
        --mappings /kaggle/input/map-keyframes \\
        --keyframes /kaggle/input/keyframes \\
        --query-zip /kaggle/input/queries.zip \\
        --query-variants-file /kaggle/input/variants.json \\
        --model-cache /kaggle/input/cache \\
        --output-root /kaggle/working/v2-output \\
        --top-n 30 --alpha 0.6 --batch-size 4

Single-query CPU mode (no GPU required)::

    python scripts/v2_pipeline.py ... --query-name query-p1-01-kis.txt --skip-vlm

Smoke-test only (no model downloads)::

    python scripts/v2_pipeline.py ... --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools  # used in _validate_submission_zip
import json
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.logging import setup_logger  # noqa: E402
from backend.services.batch1_retrieval import (  # noqa: E402
    FusedSearchCandidate,
    OfficialClipShardIndex,
    SearchCandidate,
    fuse_variant_rankings,
)
from backend.services.clip_text_encoder import OpenAIClipTextEncoder  # noqa: E402
from backend.services.v2_trake_engine import (  # noqa: E402
    TRAKEEngine,
    format_trake_submission_row,
    sequence_to_dict,
)
from backend.services.v2_vlm_reranker import M2BatchReranker, M2ScoredCandidate  # noqa: E402

logger = setup_logger("v2_pipeline")


# ---------------------------------------------------------------------------
# Query classification helpers (reuse V1 functions)
# ---------------------------------------------------------------------------


def _classify_queries(query_zip: Path) -> dict[str, dict[str, str]]:
    """Return {task: {filename: text}} for KIS/QA/TRAKE."""
    from scripts.search_batch1 import read_classified_queries
    return read_classified_queries(query_zip)


def _load_variants(variants_path: Path, query_filename: str, query_text: str) -> list[str]:
    from scripts.search_batch1 import read_query_variants
    return list(read_query_variants(variants_path, query_filename, query_text))


def _parse_trake_events(text: str) -> tuple[str, ...]:
    from scripts.search_batch1 import parse_trake_events
    return parse_trake_events(text)


# ---------------------------------------------------------------------------
# Stage 1: CLIP retrieval
# ---------------------------------------------------------------------------


def stage1_retrieve(
    query_variants: list[str],
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    *,
    top_k: int = 100,
) -> list[FusedSearchCandidate]:
    """Multi-variant CLIP retrieval with RRF fusion."""
    rankings: list[list[SearchCandidate]] = [
        index.search(encoder.encode(v), top_k=100) for v in query_variants
    ]
    return fuse_variant_rankings(rankings, top_k=top_k)


def stage1_pool(
    query_variants: list[str],
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    *,
    pool_top_k: int = 300,
) -> list[SearchCandidate]:
    """Enlarged pool search for TRAKE temporal assembly."""
    if not query_variants:
        return []
    return index.search_pool(encoder.encode(query_variants[0]), top_k=pool_top_k)


# ---------------------------------------------------------------------------
# Score normalisation (Fix 1)
# ---------------------------------------------------------------------------


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    s_min, s_max = min(scores), max(scores)
    if s_max == s_min:
        return [0.5] * len(scores)
    return [(s - s_min) / (s_max - s_min) for s in scores]


# ---------------------------------------------------------------------------
# Fused result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FusedResult:
    rank: int
    video_id: str
    frame_id: int
    keyframe_id: str
    image_path: str
    final_score: float
    retrieval_score_raw: float
    retrieval_score_norm: float
    vlm_relevance_raw: float
    vlm_relevance_norm: float
    vlm_confidence: int
    vlm_answer: str
    vlm_reasoning: str


def stage3_fuse(
    scored_candidates: list[M2ScoredCandidate],
    *,
    alpha: float = 0.6,
) -> list[FusedResult]:
    """Min-Max normalised late fusion: final = (1-α)·R_norm + α·V_norm."""
    if not scored_candidates:
        return []
    retrieval_norm = _minmax_normalize([c.retrieval_score for c in scored_candidates])
    vlm_norm = _minmax_normalize([c.vlm_relevance for c in scored_candidates])
    results = [
        FusedResult(
            rank=0,
            video_id=c.video_id,
            frame_id=c.frame_id,
            keyframe_id=c.keyframe_id,
            image_path=str(c.image_path),
            final_score=(1.0 - alpha) * retrieval_norm[i] + alpha * vlm_norm[i],
            retrieval_score_raw=c.retrieval_score,
            retrieval_score_norm=retrieval_norm[i],
            vlm_relevance_raw=c.vlm_relevance,
            vlm_relevance_norm=vlm_norm[i],
            vlm_confidence=c.vlm_confidence,
            vlm_answer=c.vlm_answer,
            vlm_reasoning=c.vlm_reasoning,
        )
        for i, c in enumerate(scored_candidates)
    ]
    results.sort(key=lambda r: (-r.final_score, r.video_id, r.frame_id))
    for i, r in enumerate(results, 1):
        r.rank = i
    return results


# ---------------------------------------------------------------------------
# Submission CSV writers
# ---------------------------------------------------------------------------


def _write_kis_submission(results: list[FusedResult], path: Path) -> None:
    """Headerless: video_id,frame_id (top-100)."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in results[:100]:
            w.writerow([r.video_id, r.frame_id])


def _write_qa_submission(results: list[FusedResult], path: Path) -> None:
    """Headerless: video_id,frame_id,answer (top-1 with VLM answer)."""
    if not results:
        raise ValueError("QA result list is empty")
    top = results[0]
    answer = top.vlm_answer.strip()[:100] if top.vlm_answer.strip() else ""
    if not answer:
        raise ValueError(f"QA answer is blank for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([top.video_id, top.frame_id, answer])


def _write_trake_submission(sequences: list[Any], path: Path, n_events: int) -> None:
    """Headerless: video_id,frame_1,...,frame_N (top-1 sequence)."""
    if not sequences:
        raise ValueError(f"No valid TRAKE sequence for {path.name}")
    row = format_trake_submission_row(sequences[0])
    if len(row) != n_events + 1:
        raise ValueError(f"TRAKE row length mismatch: expected {n_events + 1}, got {len(row)}")
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def _write_provenance(results: list[FusedResult], path: Path, query: str) -> None:
    path.write_text(
        json.dumps({"query": query, "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test_v1(
    feature_root: Path,
    mapping_root: Path,
    keyframe_root: Path,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    shards = sorted(feature_root.glob("*.npy"))
    if not shards:
        logger.error("[Smoke] No .npy shards in %s", feature_root)
        return report
    for shard in shards:
        vid = shard.stem
        errors: list[str] = []
        feat = np.load(shard, mmap_mode="r")
        if feat.ndim != 2 or feat.shape[1] != 512:
            errors.append(f"shape={feat.shape}, expected (N,512)")
        map_path = mapping_root / f"{vid}.csv"
        if not map_path.is_file():
            errors.append(f"missing mapping: {map_path}")
            report[vid] = {"status": "FAIL", "rows": int(feat.shape[0]), "errors": errors}
            continue
        with map_path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != feat.shape[0]:
            errors.append(f"row mismatch: {feat.shape[0]} vs {len(rows)}")
        for idx in [0, len(rows) // 2, len(rows) - 1]:
            if idx < len(rows):
                n = int(rows[idx]["n"])
                img = keyframe_root / vid / f"{n:03d}.jpg"
                if not img.is_file():
                    errors.append(f"missing keyframe: {img}")
        report[vid] = {"status": "PASS" if not errors else "FAIL", "rows": int(feat.shape[0]), "errors": errors}
    total = sum(int(r["rows"]) for r in report.values())
    failed = sum(1 for r in report.values() if r["status"] == "FAIL")
    logger.info("[Smoke] %d videos, %d rows, %d failures.", len(report), total, failed)
    return report


# ---------------------------------------------------------------------------
# Submission packager & validator
# ---------------------------------------------------------------------------


def _package_submission(submission_dir: Path, final_zip: Path) -> dict[str, object]:
    """Package all CSV files in submission_dir into submission.zip."""
    files = sorted(submission_dir.glob("*.csv"))
    if not files:
        raise ValueError("No CSV files found in submission directory")
    final_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, f"submission/{p.name}")
    digest = hashlib.sha256(final_zip.read_bytes()).hexdigest()
    return {"sha256": digest, "size_bytes": final_zip.stat().st_size, "files": len(files)}


def _validate_submission_zip(final_zip: Path) -> dict[str, object]:
    """Validate archive structure and per-task CSV constraints."""
    errors: list[str] = []
    counts: dict[str, int] = {"kis": 0, "qa": 0, "trake": 0}
    with zipfile.ZipFile(final_zip) as archive:
        bad = archive.testzip()
        if bad:
            errors.append(f"corrupt entry: {bad}")
        for name in archive.namelist():
            if not name.startswith("submission/"):
                errors.append(f"unexpected path: {name}")
                continue
            stem = Path(name).stem
            task = stem.rsplit("-", 1)[-1] if "-" in stem else ""
            if task not in counts:
                errors.append(f"unrecognised task suffix: {name}")
                continue
            raw = archive.read(name).decode("utf-8", errors="strict")
            rows = list(csv.reader(raw.splitlines()))
            if not rows:
                errors.append(f"empty CSV: {name}")
                continue
            if rows[0][0] == "video_id":
                errors.append(f"header found in submission CSV: {name}")
            if task == "kis":
                if len(rows) > 100:
                    errors.append(f"KIS exceeds 100 rows: {name}")
                if any(len(r) != 2 for r in rows):
                    errors.append(f"KIS column count error: {name}")
            elif task == "qa":
                if len(rows) != 1 or len(rows[0]) != 3:
                    errors.append(f"QA must be 1 row × 3 cols: {name}")
                elif not rows[0][2].strip():
                    errors.append(f"QA answer blank: {name}")
            elif task == "trake":
                if len(rows) != 1 or len(rows[0]) < 3:
                    errors.append(f"TRAKE must be 1 row with ≥2 frames: {name}")
                else:
                    frame_vals = [int(x) for x in rows[0][1:]]
                    if any(a >= b for a, b in itertools.pairwise(frame_vals)):
                        errors.append(f"TRAKE frames not strictly increasing: {name}")
            counts[task] += 1
    return {"valid": not errors, "errors": errors, "counts": counts}


# ---------------------------------------------------------------------------
# Per-task pipeline runners
# ---------------------------------------------------------------------------


def run_kis_task(
    filename: str,
    query_text: str,
    variants: list[str],
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    reranker: M2BatchReranker | None,
    submission_dir: Path,
    provenance_dir: Path,
    *,
    top_k: int,
    top_n: int,
    alpha: float,
    batch_size: int,
    skip_vlm: bool,
) -> None:
    stem = Path(filename).stem
    candidates = stage1_retrieve(variants, index, encoder, top_k=top_k)

    if skip_vlm or reranker is None:
        scored = [
            M2ScoredCandidate(
                rank=c.rank, video_id=c.video_id, frame_id=c.frame_id,
                keyframe_id=c.keyframe_id, image_path=Path(c.image_path),
                retrieval_score=c.rrf_score, vlm_relevance=0.0,
                vlm_confidence=1, vlm_answer="", vlm_reasoning="vlm_skipped",
            )
            for c in candidates
        ]
        alpha_eff = 0.0
    else:
        scored = reranker.rerank(query_text, candidates, top_n=top_n, batch_size=batch_size)
        alpha_eff = alpha

    fused = stage3_fuse(scored, alpha=alpha_eff)
    _write_kis_submission(fused, submission_dir / f"{stem}.csv")
    _write_provenance(fused, provenance_dir / f"{stem}-v2.json", filename)
    logger.info("[KIS] %s → top=%s/%s (score=%.4f)", filename,
                fused[0].video_id if fused else "N/A",
                fused[0].frame_id if fused else -1,
                fused[0].final_score if fused else 0.0)


def run_qa_task(
    filename: str,
    query_text: str,
    variants: list[str],
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    reranker: M2BatchReranker | None,
    submission_dir: Path,
    provenance_dir: Path,
    *,
    top_k: int,
    top_n: int,
    alpha: float,
    batch_size: int,
    skip_vlm: bool,
) -> None:
    stem = Path(filename).stem
    candidates = stage1_retrieve(variants, index, encoder, top_k=top_k)

    if skip_vlm or reranker is None:
        logger.warning("[QA] %s: VLM skipped — answer will be blank placeholder.", filename)
        scored = [
            M2ScoredCandidate(
                rank=c.rank, video_id=c.video_id, frame_id=c.frame_id,
                keyframe_id=c.keyframe_id, image_path=Path(c.image_path),
                retrieval_score=c.rrf_score, vlm_relevance=0.0,
                vlm_confidence=1, vlm_answer="", vlm_reasoning="vlm_skipped",
            )
            for c in candidates
        ]
        alpha_eff = 0.0
    else:
        scored = reranker.rerank(query_text, candidates, top_n=top_n, batch_size=batch_size)
        alpha_eff = alpha

    fused = stage3_fuse(scored, alpha=alpha_eff)
    _write_qa_submission(fused, submission_dir / f"{stem}.csv")
    _write_provenance(fused, provenance_dir / f"{stem}-v2.json", filename)
    logger.info("[QA] %s → answer=%r", filename, fused[0].vlm_answer if fused else "")


def run_trake_task(
    filename: str,
    query_text: str,
    event_variants: list[list[str]],
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    submission_dir: Path,
    provenance_dir: Path,
    *,
    pool_top_k: int = 300,
    beam_width: int = 500,
) -> None:
    stem = Path(filename).stem
    n_events = len(event_variants)
    engine = TRAKEEngine(beam_width=beam_width, pool_top_k=pool_top_k)

    event_candidates: list[list[SearchCandidate]] = []
    for evt_idx, variants in enumerate(event_variants):
        rankings = [index.search_pool(encoder.encode(v), top_k=pool_top_k) for v in variants]
        # Flatten and keep unique (video_id, frame_id) keeping best score
        best: dict[tuple[str, int], SearchCandidate] = {}
        for ranking in rankings:
            for c in ranking:
                key = (c.video_id, c.frame_id)
                if key not in best or c.score > best[key].score:
                    best[key] = c
        event_candidates.append(list(best.values()))
        logger.info("[TRAKE] Event %d: %d unique candidates.", evt_idx, len(event_candidates[-1]))

    sequences = engine.assemble(event_candidates, top_k=20)
    _write_trake_submission(sequences, submission_dir / f"{stem}.csv", n_events)
    provenance = {
        "query_filename": filename,
        "query_text": query_text,
        "n_events": n_events,
        "sequences": [sequence_to_dict(s) for s in sequences],
    }
    (provenance_dir / f"{stem}-v2.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("[TRAKE] %s → seq_count=%d, top=%s %s",
                filename, len(sequences),
                sequences[0].video_id if sequences else "N/A",
                sequences[0].frame_ids if sequences else ())


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V2 multi-task pipeline: CLIP → VLM rerank → fusion → submission.zip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data paths
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--mappings", type=Path, required=True)
    p.add_argument("--keyframes", type=Path, required=True)
    p.add_argument("--query-zip", type=Path, required=True)
    p.add_argument("--query-variants-file", type=Path, required=True,
                   help="Variants JSON covering all tasks")
    p.add_argument("--model-cache", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    # Single-query mode
    p.add_argument("--query-name", type=str, default=None,
                   help="Process only this query file (optional; default: batch all)")
    # VLM
    p.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--no-4bit", action="store_true")
    # Tuning
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--pool-top-k", type=int, default=300)
    p.add_argument("--beam-width", type=int, default=500)
    # Modes
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--skip-vlm", action="store_true",
                   help="Stage 1 only (CPU; no GPU required). QA answers will be blank.")
    p.add_argument("--no-package", action="store_true",
                   help="Skip ZIP packaging (output CSVs only)")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    t_start = time.perf_counter()

    # ── Smoke test ──────────────────────────────────────────────────
    if args.smoke_test:
        report = smoke_test_v1(args.features, args.mappings, args.keyframes)
        args.output_root.mkdir(parents=True, exist_ok=True)
        out = args.output_root / "smoke_test.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        failed = sum(1 for r in report.values() if r["status"] == "FAIL")
        logger.info("[Smoke] Report written to %s", out)
        raise SystemExit(1 if failed else 0)

    # ── Load shared resources ───────────────────────────────────────
    logger.info("[Init] Loading CLIP index and encoder…")
    index = OfficialClipShardIndex(args.features, args.mappings, args.keyframes)
    encoder = OpenAIClipTextEncoder(model_cache=args.model_cache)
    logger.info("[Init] Index: %d videos, %d records.", index.video_count, index.record_count)

    # ── Classify all queries ────────────────────────────────────────
    classified = _classify_queries(args.query_zip)

    # Single-query override
    if args.query_name:
        task = next(
            (t for t, d in classified.items() if args.query_name in d),
            None,
        )
        if task is None:
            raise ValueError(f"Query {args.query_name!r} not found in ZIP")
        classified = {t: ({args.query_name: classified[t][args.query_name]} if t == task else {})
                      for t in classified}

    kis_queries = classified.get("kis", {})
    qa_queries = classified.get("qa", {})
    trake_queries = classified.get("trake", {})
    total = len(kis_queries) + len(qa_queries) + len(trake_queries)
    logger.info("[Init] Tasks: %d KIS, %d QA, %d TRAKE (%d total).",
                len(kis_queries), len(qa_queries), len(trake_queries), total)

    # ── Setup output dirs ───────────────────────────────────────────
    submission_dir = args.output_root / "submission"
    provenance_dir = args.output_root / "provenance"
    submission_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    # ── Load VLM reranker once ──────────────────────────────────────
    reranker: M2BatchReranker | None = None
    if not args.skip_vlm and (kis_queries or qa_queries):
        logger.info("[Init] Loading VLM reranker (%s)…", args.vlm_model)
        reranker = M2BatchReranker(
            model_id=args.vlm_model, device="cuda", use_4bit=not args.no_4bit,
        )
        reranker.load_model()

    # ── KIS tasks ───────────────────────────────────────────────────
    for filename, query_text in kis_queries.items():
        variants = _load_variants(args.query_variants_file, filename, query_text)
        run_kis_task(
            filename, query_text, variants, index, encoder, reranker,
            submission_dir, provenance_dir,
            top_k=args.top_k, top_n=args.top_n, alpha=args.alpha,
            batch_size=args.batch_size, skip_vlm=args.skip_vlm,
        )

    # ── QA tasks ────────────────────────────────────────────────────
    for filename, query_text in qa_queries.items():
        variants = _load_variants(args.query_variants_file, filename, query_text)
        run_qa_task(
            filename, query_text, variants, index, encoder, reranker,
            submission_dir, provenance_dir,
            top_k=args.top_k, top_n=args.top_n, alpha=args.alpha,
            batch_size=args.batch_size, skip_vlm=args.skip_vlm,
        )

    # ── Unload VLM before TRAKE (free VRAM) ─────────────────────────
    if reranker is not None:
        reranker.unload_model()
        reranker = None
        gc.collect()

    # ── TRAKE tasks ──────────────────────────────────────────────────
    for filename, query_text in trake_queries.items():
        events = _parse_trake_events(query_text)
        # Each event needs its own variants list from the variants JSON
        event_variants: list[list[str]] = []
        for _evt_text in events:
            evts = _load_variants(args.query_variants_file, filename, query_text)
            event_variants.append(evts)
        run_trake_task(
            filename, query_text, event_variants, index, encoder,
            submission_dir, provenance_dir,
            pool_top_k=args.pool_top_k, beam_width=args.beam_width,
        )

    # ── Package submission ZIP ───────────────────────────────────────
    if not args.no_package:
        final_zip = args.output_root / "submission.zip"
        pkg = _package_submission(submission_dir, final_zip)
        validation = _validate_submission_zip(final_zip)
        elapsed = time.perf_counter() - t_start
        summary = {
            "status": "success" if validation["valid"] else "invalid",
            "elapsed_seconds": round(elapsed, 2),
            "tasks": {"kis": len(kis_queries), "qa": len(qa_queries), "trake": len(trake_queries)},
            "package": pkg,
            "validation": validation,
            "submission_zip": str(final_zip),
        }
        logger.info("[Done] Pipeline complete in %.1fs. ZIP: %s", elapsed, final_zip)
        if validation["errors"]:
            for err in validation["errors"]:
                logger.error("[Validation] %s", err)
        sys.stdout.write(json.dumps(summary, ensure_ascii=True) + "\n")
    else:
        elapsed = time.perf_counter() - t_start
        logger.info("[Done] No-package mode complete in %.1fs.", elapsed)


if __name__ == "__main__":
    main()
