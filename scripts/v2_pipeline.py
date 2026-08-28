"""V2 3-stage Kaggle GPU pipeline: CLIP retrieval → batched VLM rerank → fusion.

Usage (Kaggle / local)::

    python scripts/v2_pipeline.py \\
        --features /kaggle/input/clip-features-32 \\
        --mappings /kaggle/input/map-keyframes \\
        --keyframes /kaggle/input/keyframes \\
        --query-zip /kaggle/input/queries.zip \\
        --query-variants-file /kaggle/input/variants.json \\
        --model-cache /kaggle/input/model-cache \\
        --output-root /kaggle/working/v2-output \\
        --top-n 30 --alpha 0.6 --batch-size 4

Stages:
    1. **Retrieval** — OpenAI CLIP ViT-B/32, cosine search, multi-variant RRF.
    2. **Reranking** — Qwen2.5-VL-7B (4-bit) with structured JSON scoring.
    3. **Fusion** — Min-Max normalised late fusion and submission formatting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
from backend.services.v2_vlm_reranker import M2BatchReranker, M2ScoredCandidate  # noqa: E402

logger = setup_logger("v2_pipeline")


# ---------------------------------------------------------------------------
# V1 Candidate Adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2CandidateRecord:
    """Intermediate record bridging V1 retrieval and M2 reranking."""

    rank: int
    video_id: str
    frame_id: int
    keyframe_id: str
    image_path: Path
    retrieval_score: float


def adapt_candidates(
    candidates: list[SearchCandidate] | list[FusedSearchCandidate],
    *,
    validate_exists: bool = True,
) -> list[M2CandidateRecord]:
    """Convert V1 candidates to M2-ready records with path validation.

    Args:
        candidates: Stage 1 output with pre-resolved ``image_path``.
        validate_exists: If True, skip candidates whose image is missing.

    Returns:
        Validated records preserving original rank order.

    Raises:
        ValueError: If all candidates fail validation.
    """
    records: list[M2CandidateRecord] = []
    skipped = 0
    for c in candidates:
        path = Path(c.image_path)
        if validate_exists and not path.is_file():
            skipped += 1
            continue
        score = c.rrf_score if isinstance(c, FusedSearchCandidate) else c.score
        records.append(M2CandidateRecord(
            rank=c.rank,
            video_id=c.video_id,
            frame_id=c.frame_id,
            keyframe_id=c.keyframe_id,
            image_path=path.resolve(),
            retrieval_score=float(score),
        ))
    if skipped:
        logger.warning("[Adapter] Skipped %d candidates with missing images.", skipped)
    if not records:
        raise ValueError("All candidates failed image path validation")
    return records


# ---------------------------------------------------------------------------
# Stage 1: CLIP Retrieval (reuses V1)
# ---------------------------------------------------------------------------


def stage1_retrieve(
    query_variants: list[str],
    index: OfficialClipShardIndex,
    encoder: OpenAIClipTextEncoder,
    *,
    top_k: int = 100,
) -> list[FusedSearchCandidate]:
    """Fast CLIP retrieval with multi-variant RRF fusion.

    Args:
        query_variants: English visual variants of the query.
        index: Preloaded shard index.
        encoder: CLIP text encoder.
        top_k: Number of candidates to retrieve.

    Returns:
        Fused candidates sorted by RRF score descending.
    """
    t0 = time.perf_counter()
    rankings: list[list[SearchCandidate]] = []
    for variant in query_variants:
        vec = encoder.encode(variant)
        rankings.append(index.search(vec, top_k=100))
    fused = fuse_variant_rankings(rankings, top_k=top_k)
    logger.info(
        "[Stage1] Retrieved %d candidates from %d variants in %.2fs.",
        len(fused), len(query_variants), time.perf_counter() - t0,
    )
    return fused


# ---------------------------------------------------------------------------
# Stage 3: Late Fusion with Min-Max Normalisation (Fix 1)
# ---------------------------------------------------------------------------


def _minmax_normalize(scores: list[float]) -> list[float]:
    """Normalise scores to [0, 1] via Min-Max scaling.

    If all scores are identical (s_max == s_min), returns 0.5 for all
    to avoid division by zero and provide a neutral midpoint.

    Args:
        scores: Raw score values.

    Returns:
        Normalised score values in [0.0, 1.0].
    """
    if not scores:
        return []
    s_min = min(scores)
    s_max = max(scores)
    if s_max == s_min:
        return [0.5] * len(scores)
    return [(s - s_min) / (s_max - s_min) for s in scores]


@dataclass(slots=True)
class FusedResult:
    """Final fused candidate for submission output."""

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
    """Min-Max normalised late fusion of retrieval and VLM scores.

    Formula: ``final = (1 - alpha) * norm_retrieval + alpha * norm_vlm``

    Both score distributions are independently normalised to [0, 1] before
    combination.  This prevents the VLM scores (which live in [0, 1]) from
    overwriting the retrieval scores (which live in [0, ~0.03] for RRF).

    Args:
        scored_candidates: M2-scored candidates from Stage 2.
        alpha: Weight for VLM signal.  0.6 = 60% VLM, 40% retrieval.

    Returns:
        Fused results sorted by ``final_score`` descending.
    """
    if not scored_candidates:
        return []

    retrieval_raw = [c.retrieval_score for c in scored_candidates]
    vlm_raw = [c.vlm_relevance for c in scored_candidates]

    retrieval_norm = _minmax_normalize(retrieval_raw)
    vlm_norm = _minmax_normalize(vlm_raw)

    results: list[FusedResult] = []
    for i, c in enumerate(scored_candidates):
        final = (1.0 - alpha) * retrieval_norm[i] + alpha * vlm_norm[i]
        results.append(FusedResult(
            rank=0,  # assigned after sorting
            video_id=c.video_id,
            frame_id=c.frame_id,
            keyframe_id=c.keyframe_id,
            image_path=str(c.image_path),
            final_score=final,
            retrieval_score_raw=c.retrieval_score,
            retrieval_score_norm=retrieval_norm[i],
            vlm_relevance_raw=c.vlm_relevance,
            vlm_relevance_norm=vlm_norm[i],
            vlm_confidence=c.vlm_confidence,
            vlm_answer=c.vlm_answer,
            vlm_reasoning=c.vlm_reasoning,
        ))

    results.sort(key=lambda r: (-r.final_score, r.video_id, r.frame_id))
    for i, r in enumerate(results, start=1):
        r.rank = i

    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test_v1(
    feature_root: Path,
    mapping_root: Path,
    keyframe_root: Path,
) -> dict[str, dict[str, object]]:
    """Validate V1 data contracts before M2 integration.

    Args:
        feature_root: Directory of per-video ``.npy`` feature shards.
        mapping_root: Directory of per-video mapping CSVs.
        keyframe_root: Root directory of keyframe images.

    Returns:
        ``{video_id: {status, rows, errors}}`` report.
    """
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
            errors.append(f"shape={feat.shape}, expected (N, 512)")

        map_path = mapping_root / f"{vid}.csv"
        if not map_path.is_file():
            errors.append(f"missing mapping: {map_path}")
            report[vid] = {"status": "FAIL", "rows": int(feat.shape[0]), "errors": errors}
            continue

        with map_path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        if len(rows) != feat.shape[0]:
            errors.append(f"row mismatch: {feat.shape[0]} features vs {len(rows)} mappings")

        # Spot-check images at boundaries
        for idx in [0, len(rows) // 2, len(rows) - 1]:
            if idx < len(rows):
                n = int(rows[idx]["n"])
                img = keyframe_root / vid / f"{n:03d}.jpg"
                if not img.is_file():
                    errors.append(f"missing keyframe: {img}")

        report[vid] = {
            "status": "PASS" if not errors else "FAIL",
            "rows": int(feat.shape[0]),
            "errors": errors,
        }

    total = sum(int(r["rows"]) for r in report.values())
    failed = sum(1 for r in report.values() if r["status"] == "FAIL")
    logger.info("[Smoke] %d videos, %d rows, %d failures.", len(report), total, failed)
    return report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_fused_results(
    results: list[FusedResult],
    output_dir: Path,
    query_name: str,
) -> dict[str, Path]:
    """Write fusion results as JSON, CSV, and headerless submission CSV.

    Args:
        results: Fused and sorted results.
        output_dir: Output directory (created if missing).
        query_name: Query stem for file naming.

    Returns:
        Dict of ``{format: path}`` for generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(query_name).stem

    # Full provenance JSON
    json_path = output_dir / f"{stem}-v2.json"
    json_path.write_text(
        json.dumps(
            {"query": query_name, "results": [asdict(r) for r in results]},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # Diagnostic CSV (with headers)
    csv_path = output_dir / f"{stem}-v2.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "rank", "video_id", "frame_id", "final_score",
            "retrieval_norm", "vlm_relevance", "vlm_confidence",
            "vlm_answer", "image_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "rank": r.rank,
                "video_id": r.video_id,
                "frame_id": r.frame_id,
                "final_score": f"{r.final_score:.6f}",
                "retrieval_norm": f"{r.retrieval_score_norm:.6f}",
                "vlm_relevance": f"{r.vlm_relevance_raw:.4f}",
                "vlm_confidence": r.vlm_confidence,
                "vlm_answer": r.vlm_answer,
                "image_path": r.image_path,
            })

    # Submission CSV (headerless: video_id,frame_id)
    sub_path = output_dir / f"{stem}.csv"
    with sub_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([r.video_id, r.frame_id])

    return {"json": json_path, "csv": csv_path, "submission": sub_path}


# ---------------------------------------------------------------------------
# Variant loading (reused from search_batch1)
# ---------------------------------------------------------------------------


def _load_variants(variants_path: Path, query_filename: str, query_text: str) -> list[str]:
    """Load query variants from the canonical JSON format."""
    from scripts.search_batch1 import read_query_variants
    return list(read_query_variants(variants_path, query_filename, query_text))


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the V2 pipeline CLI argument parser."""
    p = argparse.ArgumentParser(
        description="V2 3-stage pipeline: CLIP → VLM rerank → fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    p.add_argument("--features", type=Path, required=True, help="CLIP feature shard directory")
    p.add_argument("--mappings", type=Path, required=True, help="Mapping CSV directory")
    p.add_argument("--keyframes", type=Path, required=True, help="Keyframe images root")
    p.add_argument("--query-zip", type=Path, required=True, help="Query ZIP archive")
    p.add_argument("--query-name", type=str, required=True, help="Query filename inside ZIP")
    p.add_argument("--query-variants-file", type=Path, required=True, help="Variants JSON path")
    p.add_argument("--model-cache", type=Path, required=True, help="CLIP model cache directory")
    p.add_argument("--output-root", type=Path, required=True, help="Output directory")

    # VLM model
    p.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model ID")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    # Pipeline tuning
    p.add_argument("--top-k", type=int, default=100, help="Stage 1 retrieval candidates")
    p.add_argument("--top-n", type=int, default=30, help="Stage 2 VLM rerank candidates")
    p.add_argument("--alpha", type=float, default=0.6, help="VLM weight in late fusion [0, 1]")
    p.add_argument("--batch-size", type=int, default=4, help="Images per VLM forward pass")

    # Modes
    p.add_argument("--smoke-test", action="store_true", help="Run V1 data validation only")
    p.add_argument("--skip-vlm", action="store_true", help="Run Stage 1 only (no GPU needed)")

    return p


def main() -> None:
    """V2 pipeline entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    # ── Smoke test mode ───────────────────────────────────────────────
    if args.smoke_test:
        report = smoke_test_v1(args.features, args.mappings, args.keyframes)
        args.output_root.mkdir(parents=True, exist_ok=True)
        out = args.output_root / "smoke_test.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        failed = sum(1 for r in report.values() if r["status"] == "FAIL")
        logger.info("[Smoke] Report written to %s", out)
        raise SystemExit(1 if failed else 0)

    # ── Load V1 index and encoder ─────────────────────────────────────
    t_start = time.perf_counter()
    logger.info("[Init] Loading CLIP index and encoder…")
    index = OfficialClipShardIndex(args.features, args.mappings, args.keyframes)
    encoder = OpenAIClipTextEncoder(model_cache=args.model_cache)
    logger.info("[Init] Index: %d videos, %d records.", index.video_count, index.record_count)

    # ── Load query and variants ───────────────────────────────────────
    from scripts.search_batch1 import read_query_zip
    query_text = read_query_zip(args.query_zip, args.query_name)
    variants = _load_variants(args.query_variants_file, args.query_name, query_text)
    logger.info("[Init] Query: %r, %d variants.", args.query_name, len(variants))

    # ── Stage 1: CLIP Retrieval ───────────────────────────────────────
    stage1_candidates = stage1_retrieve(
        variants, index, encoder, top_k=args.top_k,
    )

    if args.skip_vlm:
        # Write Stage 1 results directly (no VLM)
        dummy_scored = [
            M2ScoredCandidate(
                rank=c.rank, video_id=c.video_id, frame_id=c.frame_id,
                keyframe_id=c.keyframe_id, image_path=Path(c.image_path),
                retrieval_score=c.rrf_score, vlm_relevance=0.0,
                vlm_confidence=1, vlm_answer="", vlm_reasoning="vlm skipped",
            )
            for c in stage1_candidates
        ]
        fused = stage3_fuse(dummy_scored, alpha=0.0)
        paths = write_fused_results(fused, args.output_root, args.query_name)
        logger.info("[Done] Stage 1 only. Output: %s", paths)
        return

    # ── Stage 2: Batched VLM Rerank ───────────────────────────────────
    logger.info("[Stage2] Loading VLM (%s)…", args.vlm_model)
    reranker = M2BatchReranker(
        model_id=args.vlm_model,
        device="cuda",
        use_4bit=not args.no_4bit,
    )
    reranker.load_model()

    t_s2 = time.perf_counter()
    scored = reranker.rerank(
        query_text,
        stage1_candidates,
        top_n=args.top_n,
        batch_size=args.batch_size,
    )
    logger.info(
        "[Stage2] Reranked %d candidates in %.2fs.",
        min(args.top_n, len(stage1_candidates)),
        time.perf_counter() - t_s2,
    )

    reranker.unload_model()

    # ── Stage 3: Normalised Late Fusion ───────────────────────────────
    fused = stage3_fuse(scored, alpha=args.alpha)
    paths = write_fused_results(fused, args.output_root, args.query_name)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[Done] V2 pipeline complete in %.1fs. Top result: %s/%d (score=%.4f). Files: %s",
        elapsed,
        fused[0].video_id if fused else "N/A",
        fused[0].frame_id if fused else -1,
        fused[0].final_score if fused else 0.0,
        {k: str(v) for k, v in paths.items()},
    )

    # Summary to stdout for Kaggle notebook capture
    sys.stdout.write(json.dumps({
        "status": "success",
        "query": args.query_name,
        "total_candidates": len(fused),
        "top_n_reranked": min(args.top_n, len(stage1_candidates)),
        "alpha": args.alpha,
        "elapsed_seconds": round(elapsed, 2),
        "output_files": {k: str(v) for k, v in paths.items()},
    }, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
