"""Extract global and optional dense SigLIP2 visual features from keyframes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.services.embedding import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED_TAG,
    SigLIP2Encoder,
)

MODEL_NAME = DEFAULT_MODEL_NAME
PRETRAINED_TAG = DEFAULT_PRETRAINED_TAG


def extract_siglip2_features(
    keyframe_directory: Path,
    output_directory: Path,
    video_ids: list[str],
    batch_size: int = 1,
    device: str | None = None,
    save_dense: bool = False,
    weights: Path | None = None,
) -> list[dict[str, Any]]:
    """Write one normalized global matrix per video and dense features on demand."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if weights is None:
        raise ValueError("--weights is required; pass the downloaded open_clip_model.safetensors file")
    weights_path = weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    encoder = SigLIP2Encoder(weights_path, device=device)
    output_directory.mkdir(parents=True, exist_ok=True)
    reports = []

    for video_id in video_ids:
        images = sorted((keyframe_directory / video_id).glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No keyframes found for {video_id}")
        global_path = output_directory / "global" / f"{video_id}.npy"
        dense_path = output_directory / "dense" / f"{video_id}.npy"
        global_path.parent.mkdir(parents=True, exist_ok=True)
        if save_dense:
            dense_path.parent.mkdir(parents=True, exist_ok=True)

        global_batches = []
        dense_batches = []
        for start in range(0, len(images), batch_size):
            batch_paths = images[start : start + batch_size]
            batch_images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    batch_images.append(image.convert("RGB"))
            global_features, dense_features = encoder.encode_batch(batch_images, return_dense=save_dense)
            global_batches.append(global_features.astype(np.float16))
            if dense_features is not None:
                dense_batches.append(dense_features.astype(np.float16))

        global_matrix = np.concatenate(global_batches)
        np.save(global_path, global_matrix)
        dense_shape = None
        if save_dense:
            dense_matrix = np.concatenate(dense_batches)
            np.save(dense_path, dense_matrix)
            dense_shape = list(dense_matrix.shape)
        reports.append(
            {
                "video_id": video_id,
                "keyframes": len(images),
                "global_shape": list(global_matrix.shape),
                "dense_shape": dense_shape,
            }
        )

    manifest = {
        "model": MODEL_NAME,
        "pretrained": PRETRAINED_TAG,
        "weights": str(weights.resolve()) if weights else "Hugging Face cache",
        "image_size": 384,
        "normalized": True,
        "dtype": "float16",
        "device": encoder.device,
        "videos": reports,
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return reports
