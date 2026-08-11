"""Normalize organizer keyframes, mappings, features, objects, and metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_DIRECTORY_NAMES = {
    "videos": ("Videos", "videos", "Video", "video"),
    "keyframes": ("Keyframes", "keyframes"),
    "features": ("clip", "CLIP features", "CLIP_features", "clip-features", "clip-features-32", "clip_features"),
    "objects": ("Objects", "objects"),
    "metadata": ("Metadata", "metadata", "media-info"),
    "mapping": ("Map-keyframes", "map-keyframes", "FrameMaps", "frame-maps"),
}


def _find_artifact_directory(dataset: Path, artifact_type: str, required: bool = True) -> Path | None:
    path = next(
        (dataset / name for name in ARTIFACT_DIRECTORY_NAMES[artifact_type] if (dataset / name).is_dir()),
        None,
    )
    if required and path is None:
        expected = ARTIFACT_DIRECTORY_NAMES[artifact_type]
        raise FileNotFoundError(f"Missing {artifact_type} directory under {dataset}: {expected}")
    return path


def _portable_path(path: Path | None, dataset: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(dataset.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _find_artifact_file(directory: Path | None, stem: str, suffix: str) -> Path | None:
    if directory is None:
        return None
    direct = directory / f"{stem}{suffix}"
    if direct.is_file():
        return direct
    nested = directory / stem
    matches = sorted(nested.glob(f"*{suffix}")) if nested.is_dir() else []
    return matches[0] if matches else None


def _read_frame_mapping(mapping_directory: Path, video_id: str) -> list[dict[str, float | int | None]]:
    path = _find_artifact_file(mapping_directory, video_id, ".csv")
    if path is None:
        raise FileNotFoundError(f"Missing frame mapping CSV for {video_id}")

    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty frame mapping: {path}")

    mapping = []
    for row in rows:
        frame_text = next((row.get(key) for key in ("frame_id", "frame_idx", "frame") if row.get(key)), None)
        if frame_text is None:
            raise ValueError(f"{path} needs a frame_id, frame_idx, or frame column")
        frame_id = int(float(frame_text))
        time_text = next((row.get(key) for key in ("timestamp", "pts_time", "time") if row.get(key)), None)
        fps_text = row.get("fps")
        timestamp = float(time_text) if time_text is not None else frame_id / float(fps_text) if fps_text else None
        mapping.append({"frame_id": frame_id, "timestamp": timestamp})
    return mapping


def build_index(dataset: Path, output: Path, video_prefix: str | None = None) -> dict[str, int]:
    """Validate BTC artifacts and create a portable hand-off index."""
    dataset, output = dataset.resolve(), output.resolve()
    keyframe_directory = _find_artifact_directory(dataset, "keyframes")
    feature_directory = _find_artifact_directory(dataset, "features")
    mapping_directory = _find_artifact_directory(dataset, "mapping")
    object_directory = _find_artifact_directory(dataset, "objects", required=False)
    metadata_directory = _find_artifact_directory(dataset, "metadata", required=False)
    video_directory = _find_artifact_directory(dataset, "videos", required=False)

    videos: list[dict[str, Any]] = []
    batches: list[tuple[str, list[Path], list[dict[str, float | int | None]], np.ndarray]] = []
    feature_dimension: int | None = None
    total_frames = 0

    for video_keyframes in sorted(
        path
        for path in keyframe_directory.iterdir()
        if path.is_dir() and (video_prefix is None or path.name.startswith(video_prefix))
    ):
        video_id = video_keyframes.name
        images = sorted(
            path for path in video_keyframes.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        feature_path = _find_artifact_file(feature_directory, video_id, ".npy")
        if not images or feature_path is None:
            raise FileNotFoundError(f"{video_id} needs keyframes and a CLIP .npy file")
        feature_matrix = np.load(feature_path, mmap_mode="r")
        frame_mapping = _read_frame_mapping(mapping_directory, video_id)
        if feature_matrix.ndim != 2 or len(images) != len(feature_matrix) or len(images) != len(frame_mapping):
            raise ValueError(
                f"{video_id}: keyframes={len(images)}, features={len(feature_matrix)}, "
                f"mapping={len(frame_mapping)} must match"
            )
        feature_dimension = feature_dimension or int(feature_matrix.shape[1])
        if feature_matrix.shape[1] != feature_dimension:
            raise ValueError(f"{video_id}: feature dimension {feature_matrix.shape[1]} != {feature_dimension}")
        batches.append((video_id, images, frame_mapping, feature_matrix))
        total_frames += len(images)

        metadata_path = _find_artifact_file(metadata_directory, video_id, ".json")
        metadata = None
        if metadata_path:
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
        video_path = _find_artifact_file(video_directory, video_id, ".mp4")
        videos.append({"video_id": video_id, "video_path": _portable_path(video_path, dataset), "metadata": metadata})

    if not batches:
        raise ValueError(f"No video keyframe folders found in {keyframe_directory}")

    output.mkdir(parents=True, exist_ok=True)
    combined_features = np.lib.format.open_memmap(
        output / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, feature_dimension),
    )
    feature_row = 0
    with (output / "records.jsonl").open("w", encoding="utf-8") as records_file:
        for video_id, images, frame_mapping, feature_matrix in batches:
            combined_features[feature_row : feature_row + len(feature_matrix)] = feature_matrix
            for image, frame in zip(images, frame_mapping, strict=True):
                video_objects = (
                    object_directory / video_id
                    if object_directory and (object_directory / video_id).is_dir()
                    else object_directory
                )
                object_path = _find_artifact_file(video_objects, image.stem, ".json")
                record = {
                    "feature_row": feature_row,
                    "video_id": video_id,
                    "keyframe_id": image.stem,
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "image_path": _portable_path(image, dataset),
                    "objects_path": _portable_path(object_path, dataset),
                }
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                feature_row += 1
    combined_features.flush()

    with (output / "videos.jsonl").open("w", encoding="utf-8") as file:
        for video in videos:
            file.write(json.dumps(video, ensure_ascii=False) + "\n")

    manifest = {"videos": len(videos), "frames": total_frames, "feature_dim": int(feature_dimension)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
