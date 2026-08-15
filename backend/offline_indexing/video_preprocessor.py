"""Extract representative scene keyframes from raw videos."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video


def l1_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return normalized mean L1 distance between two BGR frames."""
    def thumbnail(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    return float(np.mean(np.abs(thumbnail(first) - thumbnail(second))))


def preprocess_video(
    video_path: Path,
    output_directory: Path,
    scene_threshold: float = 27.0,
    dedup_threshold: float = 0.04,
    minimum_scene_frames: int = 15,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Detect shots, keep non-duplicate midpoint frames, and write their mapping."""
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    video_id = video_path.stem
    keyframe_directory = output_directory / "keyframes" / video_id
    mapping_directory = output_directory / "map-keyframes"
    keyframe_directory.mkdir(parents=True, exist_ok=True)
    mapping_directory.mkdir(parents=True, exist_ok=True)
    for old_keyframe in keyframe_directory.glob("*.jpg"):
        old_keyframe.unlink()

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=scene_threshold, min_scene_len=minimum_scene_frames))
    scene_manager.detect_scenes(video=video)
    scenes = scene_manager.get_scene_list(start_in_scene=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError(f"Invalid FPS for {video_path}: {fps}")

    # ponytail: midpoint can retain transition graphics; use a scene medoid if retrieval benchmarks regress.
    targets = [((start.frame_num + end.frame_num - 1) // 2, start.frame_num, end.frame_num - 1) for start, end in scenes]
    rows: list[dict[str, int | float | str]] = []
    previous_frame: np.ndarray | None = None
    target_index = 0
    frame_id = 0
    try:
        # Sequential decoding avoids unreliable random H.264 seeks on organizer videos.
        while target_index < len(targets):
            success, frame = capture.read()
            if not success:
                raise OSError(f"Cannot read target frame {targets[target_index][0]} from {video_path}")
            target_frame, scene_start, scene_end = targets[target_index]
            if frame_id < target_frame:
                frame_id += 1
                continue
            distance = None if previous_frame is None else l1_distance(previous_frame, frame)
            if distance is not None and distance < dedup_threshold:
                target_index += 1
                frame_id += 1
                continue
            keyframe_id = f"{len(rows) + 1:04d}"
            image_path = keyframe_directory / f"{keyframe_id}.jpg"
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise OSError(f"Cannot write {image_path}")
            rows.append(
                {
                    "n": len(rows) + 1,
                    "pts_time": frame_id / fps,
                    "fps": fps,
                    "frame_idx": frame_id,
                    "scene_start": scene_start,
                    "scene_end": scene_end,
                    "l1_distance": "" if distance is None else round(distance, 6),
                }
            )
            previous_frame = frame
            target_index += 1
            frame_id += 1
    finally:
        capture.release()

    mapping_path = mapping_directory / f"{video_id}.csv"
    fieldnames = ["n", "pts_time", "fps", "frame_idx", "scene_start", "scene_end", "l1_distance"]
    with mapping_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "video_id": video_id,
        "detected_scenes": len(scenes),
        "keyframes": len(rows),
        "removed_duplicates": len(scenes) - len(rows),
        "fps": fps,
        "mapping_path": str(mapping_path),
    }


def preprocess_videos(
    video_directory: Path,
    output_directory: Path,
    video_ids: list[str],
    skip_existing: bool = False,
    workers: int = 1,
    **settings: Any,
) -> list[dict[str, Any]]:
    """Process selected MP4 files and persist a reproducible run report."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    reports = []
    pending = []
    for video_id in video_ids:
        mapping_path = output_directory / "map-keyframes" / f"{video_id}.csv"
        image_count = len(list((output_directory / "keyframes" / video_id).glob("*.jpg")))
        if skip_existing and mapping_path.is_file():
            with mapping_path.open(encoding="utf-8", newline="") as file:
                mapping_count = sum(1 for _ in csv.DictReader(file))
            if mapping_count > 0 and mapping_count == image_count:
                report = {"video_id": video_id, "keyframes": image_count, "status": "skipped"}
                reports.append(report)
                print(json.dumps(report), flush=True)
                continue
        pending.append(video_id)

    def record(report: dict[str, Any]) -> None:
        report["status"] = "completed"
        reports.append(report)
        print(json.dumps(report), flush=True)

    if workers == 1:
        for video_id in pending:
            record(preprocess_video(video_directory / f"{video_id}.mp4", output_directory, **settings))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(preprocess_video, video_directory / f"{video_id}.mp4", output_directory, **settings): video_id
                for video_id in pending
            }
            for future in as_completed(futures):
                record(future.result())
    reports.sort(key=lambda report: str(report["video_id"]))
    report_path = output_directory / "preprocessing-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"settings": settings, "skip_existing": skip_existing, "workers": workers, "videos": reports}, indent=2),
        encoding="utf-8",
    )
    return reports
