"""Normalize organizer Faster R-CNN JSON into searchable JSONL records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def normalize_objects(
    object_directory: Path,
    mapping_directory: Path,
    output_directory: Path,
    video_ids: list[str],
    minimum_score: float = 0.1,
) -> list[dict[str, Any]]:
    """Write one record per keyframe with filtered labels, scores, and boxes."""
    output_directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for video_id in video_ids:
        with (mapping_directory / f"{video_id}.csv").open(encoding="utf-8-sig", newline="") as file:
            mappings = list(csv.DictReader(file))
        object_files = sorted((object_directory / video_id).glob("*.json"))
        if len(object_files) != len(mappings):
            raise ValueError(f"{video_id}: objects={len(object_files)}, mappings={len(mappings)} must match")

        detection_count = 0
        output_path = output_directory / f"{video_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for object_path, mapping in zip(object_files, mappings, strict=True):
                source = json.loads(object_path.read_text(encoding="utf-8"))
                fields = (
                    source["detection_scores"],
                    source["detection_class_names"],
                    source["detection_class_entities"],
                    source["detection_class_labels"],
                    source["detection_boxes"],
                )
                if len({len(field) for field in fields}) != 1:
                    raise ValueError(f"Detection arrays have different lengths: {object_path}")
                detections = [
                    {
                        "entity": entity,
                        "class_name": class_name,
                        "class_id": label,
                        "score": float(score),
                        "box": [float(value) for value in box],
                    }
                    for score, class_name, entity, label, box in zip(*fields, strict=True)
                    if float(score) >= minimum_score
                ]
                detection_count += len(detections)
                record = {
                    "video_id": video_id,
                    "keyframe_id": object_path.stem,
                    "frame_id": int(float(mapping["frame_idx"])),
                    "timestamp": float(mapping["pts_time"]),
                    "labels": sorted({detection["entity"] for detection in detections}),
                    "detections": detections,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        reports.append({"video_id": video_id, "keyframes": len(object_files), "detections": detection_count})

    (output_directory / "manifest.json").write_text(
        json.dumps({"minimum_score": minimum_score, "videos": reports}, indent=2),
        encoding="utf-8",
    )
    return reports
