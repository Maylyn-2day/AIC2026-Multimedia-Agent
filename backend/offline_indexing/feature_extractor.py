"""Extract global and optional dense SigLIP2 visual features from keyframes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image


MODEL_NAME = "ViT-gopt-16-SigLIP2-384"
PRETRAINED_TAG = "webli"


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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(weights.resolve()) if weights else PRETRAINED_TAG,
        device=device,
        precision="fp16" if device.startswith("cuda") else "fp32",
    )
    model.eval()
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
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                if save_dense:
                    result = model.visual.forward_intermediates(batch, indices=[-1], output_fmt="NCHW")
                    global_features = result["image_features"]
                    dense_batches.append(result["image_intermediates"][0].cpu().to(torch.float16).numpy())
                else:
                    global_features = model.encode_image(batch)
                global_features = torch.nn.functional.normalize(global_features, dim=-1)
                global_batches.append(global_features.cpu().to(torch.float16).numpy())

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
        "device": device,
        "videos": reports,
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return reports
