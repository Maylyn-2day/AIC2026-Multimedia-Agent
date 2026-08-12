"""Validate generated keyframe mappings against video frames and organizer samples."""

from __future__ import annotations

import bisect
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import cv2

from .video_preprocessor import l1_distance


def _read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validate_keyframes(
    video_directory: Path,
    generated_directory: Path,
    reference_mapping_directory: Path,
    video_ids: list[str],
) -> list[dict[str, Any]]:
    """Check frame accuracy and compare temporal coverage with organizer keyframes."""
    reports = []
    for video_id in video_ids:
        generated = _read_mapping(generated_directory / "map-keyframes" / f"{video_id}.csv")
        reference = _read_mapping(reference_mapping_directory / f"{video_id}.csv")
        if not generated or not reference:
            raise ValueError(f"Mappings must not be empty for {video_id}")

        generated_frames = [int(row["frame_idx"]) for row in generated]
        reference_frames = sorted(int(row["frame_idx"]) for row in reference)
        nearest_deltas = []
        for frame_id in generated_frames:
            position = bisect.bisect_left(reference_frames, frame_id)
            neighbors = reference_frames[max(0, position - 1) : position + 1]
            nearest_deltas.append(min(abs(frame_id - reference_frame) for reference_frame in neighbors))

        sample_indexes = sorted({0, len(generated) // 2, len(generated) - 1})
        sample_targets = {generated_frames[index]: index for index in sample_indexes}
        sample_distances: list[float] = []
        capture = cv2.VideoCapture(str(video_directory / f"{video_id}.mp4"))
        if not capture.isOpened():
            raise OSError(f"Cannot open video {video_id}")
        frame_id = 0
        try:
            while sample_targets:
                success, frame = capture.read()
                if not success:
                    raise OSError(f"Cannot read validation frame {min(sample_targets)} from {video_id}")
                if frame_id in sample_targets:
                    row = generated[sample_targets.pop(frame_id)]
                    image = cv2.imread(str(generated_directory / "keyframes" / video_id / f"{int(row['n']):04d}.jpg"))
                    if image is None:
                        raise FileNotFoundError(f"Missing generated keyframe {video_id}/{row['n']}")
                    sample_distances.append(l1_distance(frame, image))
                frame_id += 1
        finally:
            capture.release()

        report = {
            "video_id": video_id,
            "generated_keyframes": len(generated),
            "reference_keyframes": len(reference),
            "count_ratio": round(len(generated) / len(reference), 3),
            "nearest_reference_median_seconds": round(statistics.median(nearest_deltas) / float(generated[0]["fps"]), 3),
            "saved_frame_max_l1": round(max(sample_distances), 6),
            "saved_frame_mapping_valid": max(sample_distances) < 0.02,
        }
        reports.append(report)

    report_path = generated_directory / "validation-report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports
