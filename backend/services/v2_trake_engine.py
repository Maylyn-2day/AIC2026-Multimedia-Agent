"""Canonical M3 TRAKE temporal engine for N-event sequential query alignment.

Algorithm
---------
Given per-event candidate lists ``[C_1, C_2, ..., C_N]`` (each from independent
CLIP retrieval), the engine assembles valid *same-video, monotonically ordered*
frame sequences and scores them with a penalised joint score:

    Score(S) = Σ score(E_i, frame_i)  -  λ · Penalty(Δt)

where the penalty term discourages implausibly short inter-event gaps.

Sequence assembly uses a **bounded beam search** that expands one event at a
time, pruning the beam to ``beam_width`` candidates after each step.  For small
event counts (N ≤ 4) and typical top-k ≤ 300 per event this is equivalent to
exact DP over the full space.

Tie-breaking is fully deterministic:
    1. Higher joint score.
    2. Smaller timestamp variance (prefer evenly-spaced events).
    3. Lower video ID (lexicographic).
    4. Lower frame IDs (element-wise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.core.logging import setup_logger

logger = setup_logger("v2_trake_engine")


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrakeEventFrame:
    """One event slot in a candidate sequence."""

    event_index: int
    video_id: str
    frame_id: int
    keyframe_id: str
    image_path: Path
    pts_time: float
    retrieval_score: float
    source_rank: int


@dataclass(frozen=True, slots=True)
class TrakeSequence:
    """A validated N-event sequence ready for submission formatting."""

    rank: int
    video_id: str
    joint_score: float
    timestamp_variance: float
    events: tuple[TrakeEventFrame, ...]

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(e.frame_id for e in self.events)

    @property
    def n_events(self) -> int:
        return len(self.events)


# ---------------------------------------------------------------------------
# Internal beam-search state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BeamState:
    """Partial sequence built incrementally during beam search."""

    video_id: str
    frames: list[int] = field(default_factory=list)
    pts_times: list[float] = field(default_factory=list)
    score: float = 0.0
    events: list[TrakeEventFrame] = field(default_factory=list)

    def sort_key(self) -> tuple[float, float, str, tuple[int, ...]]:
        """Descending score, ascending ts variance, then video/frame IDs."""
        var = _variance(self.pts_times) if len(self.pts_times) > 1 else 0.0
        return (-self.score, var, self.video_id, tuple(self.frames))


# ---------------------------------------------------------------------------
# TRAKE engine
# ---------------------------------------------------------------------------


class TRAKEEngine:
    """Canonical N-event temporal sequence assembler.

    Args:
        min_gap_seconds: Minimum acceptable inter-event timestamp gap (seconds).
            Sequences with a gap below this threshold are not discarded but
            incur a ``lambda_penalty`` per under-gap event transition.
        lambda_penalty: Score deduction per under-gap transition.
        beam_width: Maximum number of partial sequences retained per event step.
        pool_top_k: Maximum candidates per event consumed from retrieval.
    """

    def __init__(
        self,
        *,
        min_gap_seconds: float = 1.0,
        lambda_penalty: float = 0.005,
        beam_width: int = 500,
        pool_top_k: int = 300,
    ) -> None:
        self._min_gap = min_gap_seconds
        self._lambda = lambda_penalty
        self._beam_width = beam_width
        self._pool_top_k = pool_top_k

    # ── Public API ────────────────────────────────────────────────────

    def assemble(
        self,
        event_candidates: list[list[Any]],
        *,
        top_k: int = 20,
    ) -> list[TrakeSequence]:
        """Assemble valid sequences from per-event candidate lists.

        Args:
            event_candidates: List of N per-event candidate lists.  Each
                element is a list of objects with attributes ``video_id``,
                ``frame_id``, ``keyframe_id``, ``image_path``, and either
                ``rrf_score`` or ``score`` and ``rank``.  The candidates
                must carry a ``pts_time`` attribute; if absent, 0.0 is used
                (monotonicity is still enforced via ``frame_id``).
            top_k: Maximum number of sequences to return.

        Returns:
            Ranked list of ``TrakeSequence`` objects, or empty list when no
            valid same-video monotonic sequence exists.

        Raises:
            ValueError: If ``event_candidates`` is empty or any event list is
                empty.
        """
        n = len(event_candidates)
        if n == 0:
            raise ValueError("event_candidates must not be empty")
        for i, lst in enumerate(event_candidates):
            if not lst:
                raise ValueError(f"event {i} candidate list is empty")

        logger.info("[TRAKE] Assembling %d-event sequences (beam_width=%d).", n, self._beam_width)

        # Convert raw candidates to TrakeEventFrame lists per event
        event_frames: list[list[TrakeEventFrame]] = []
        for event_idx, candidates in enumerate(event_candidates):
            frames = [self._to_event_frame(event_idx, c) for c in candidates[: self._pool_top_k]]
            event_frames.append(frames)

        # Group per event by video_id for fast same-video filtering
        by_video_per_event: list[dict[str, list[TrakeEventFrame]]] = []
        for frames in event_frames:
            grouped: dict[str, list[TrakeEventFrame]] = {}
            for f in frames:
                grouped.setdefault(f.video_id, []).append(f)
            by_video_per_event.append(grouped)

        # Videos that appear in ALL events (necessary condition)
        common_videos: set[str] = set(by_video_per_event[0])
        for group in by_video_per_event[1:]:
            common_videos &= set(group)

        if not common_videos:
            logger.warning("[TRAKE] No video appears in all %d event candidate lists.", n)
            return []

        logger.info("[TRAKE] %d common videos across all events.", len(common_videos))

        # Beam search: initialise from event 0
        beam: list[_BeamState] = []
        for vid in sorted(common_videos):
            for ef in by_video_per_event[0][vid]:
                state = _BeamState(video_id=vid)
                state.frames.append(ef.frame_id)
                state.pts_times.append(ef.pts_time)
                state.score += self._frame_score(ef, penalty=0.0)
                state.events.append(ef)
                beam.append(state)

        # Expand beam one event at a time
        for event_idx in range(1, n):
            next_beam: list[_BeamState] = []
            for state in beam:
                vid = state.video_id
                last_frame = state.frames[-1]
                last_pts = state.pts_times[-1]
                candidates_for_vid = by_video_per_event[event_idx].get(vid, [])
                for ef in candidates_for_vid:
                    if ef.frame_id <= last_frame:
                        continue  # violates strict monotonic ordering
                    gap = ef.pts_time - last_pts
                    penalty = self._lambda if gap < self._min_gap else 0.0
                    new_state = _BeamState(
                        video_id=vid,
                        frames=list(state.frames) + [ef.frame_id],
                        pts_times=list(state.pts_times) + [ef.pts_time],
                        score=state.score + self._frame_score(ef, penalty=penalty),
                        events=list(state.events) + [ef],
                    )
                    next_beam.append(new_state)

            if not next_beam:
                logger.warning("[TRAKE] Beam collapsed at event %d — no valid extensions.", event_idx)
                return []

            # Prune beam
            next_beam.sort(key=lambda s: s.sort_key())
            beam = next_beam[: self._beam_width]

        # Convert final beam to TrakeSequence
        sequences: list[TrakeSequence] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for state in beam:
            identity = (state.video_id, tuple(state.frames))
            if identity in seen:
                continue
            seen.add(identity)
            var = _variance(state.pts_times) if len(state.pts_times) > 1 else 0.0
            sequences.append(TrakeSequence(
                rank=0,
                video_id=state.video_id,
                joint_score=state.score,
                timestamp_variance=var,
                events=tuple(state.events),
            ))

        # Final deterministic sort
        sequences.sort(key=lambda s: (-s.joint_score, s.timestamp_variance, s.video_id, s.frame_ids))
        for i, seq in enumerate(sequences[:top_k], start=1):
            # rank is frozen, so we must replace the object
            sequences[i - 1] = TrakeSequence(
                rank=i,
                video_id=seq.video_id,
                joint_score=seq.joint_score,
                timestamp_variance=seq.timestamp_variance,
                events=seq.events,
            )

        result = sequences[:top_k]
        logger.info("[TRAKE] Assembled %d valid sequences from %d common videos.", len(result), len(common_videos))
        return result

    # ── Helpers ──────────────────────────────────────────────────────

    def _frame_score(self, ef: TrakeEventFrame, *, penalty: float) -> float:
        return ef.retrieval_score - penalty

    @staticmethod
    def _to_event_frame(event_idx: int, candidate: Any) -> TrakeEventFrame:
        """Convert a V1 SearchCandidate / FusedSearchCandidate to TrakeEventFrame."""
        score = float(getattr(candidate, "rrf_score", None) or getattr(candidate, "score", 0.0))
        pts = float(getattr(candidate, "pts_time", 0.0))
        return TrakeEventFrame(
            event_index=event_idx,
            video_id=candidate.video_id,
            frame_id=candidate.frame_id,
            keyframe_id=candidate.keyframe_id,
            image_path=Path(candidate.image_path),
            pts_time=pts,
            retrieval_score=score,
            source_rank=candidate.rank,
        )


# ---------------------------------------------------------------------------
# Submission formatter
# ---------------------------------------------------------------------------


def format_trake_submission_row(sequence: TrakeSequence) -> list[str | int]:
    """Return a headerless CSV row: ``[video_id, frame_1, ..., frame_N]``."""
    return [sequence.video_id, *sequence.frame_ids]


# ---------------------------------------------------------------------------
# Provenance writer
# ---------------------------------------------------------------------------


def sequence_to_dict(seq: TrakeSequence) -> dict[str, object]:
    """Convert a TrakeSequence to a JSON-serialisable provenance dict."""
    return {
        "rank": seq.rank,
        "video_id": seq.video_id,
        "joint_score": seq.joint_score,
        "timestamp_variance": seq.timestamp_variance,
        "frame_ids": list(seq.frame_ids),
        "events": [
            {
                "event_index": e.event_index,
                "frame_id": e.frame_id,
                "keyframe_id": e.keyframe_id,
                "image_path": str(e.image_path),
                "pts_time": e.pts_time,
                "retrieval_score": e.retrieval_score,
                "source_rank": e.source_rank,
            }
            for e in seq.events
        ],
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _variance(values: list[float]) -> float:
    """Population variance of a list of floats."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n
