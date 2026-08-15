"""Extract global and optional dense SigLIP2 visual features from keyframes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image
from open_clip.model import _build_vision_tower
from open_clip.transform import PreprocessCfg, image_transform_v2
from safetensors import safe_open


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
    if weights is None:
        raise ValueError("--weights is required; pass the downloaded open_clip_model.safetensors file")
    weights_path = weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    config = open_clip.get_model_config(MODEL_NAME)
    compute_dtype = torch.float16 if device.startswith("cuda") else torch.bfloat16
    visual = _build_vision_tower(config["embed_dim"], config["vision_cfg"]).to(dtype=compute_dtype)
    with safe_open(weights_path, framework="pt", device="cpu") as checkpoint, torch.no_grad():
        missing = []
        for name, target in visual.state_dict().items():
            key = f"visual.{name}"
            if key not in checkpoint.keys():
                missing.append(key)
                continue
            target.copy_(checkpoint.get_tensor(key).to(dtype=target.dtype))
    if missing:
        raise ValueError(f"Missing visual weights: {missing[:5]}")
    visual = visual.to(device).eval()
    pretrained = open_clip.get_pretrained_cfg(MODEL_NAME, PRETRAINED_TAG)
    preprocess = image_transform_v2(
        PreprocessCfg(
            size=384,
            mean=pretrained["mean"],
            std=pretrained["std"],
            interpolation=pretrained["interpolation"],
            resize_mode=pretrained["resize_mode"],
        ),
        is_train=False,
    )
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
            batch = torch.stack(tensors).to(device=device, dtype=compute_dtype)
            with torch.inference_mode():
                if save_dense:
                    result = visual.forward_intermediates(batch, indices=[-1], output_fmt="NCHW")
                    global_features = result["image_features"]
                    dense_batches.append(result["image_intermediates"][0].cpu().to(torch.float16).numpy())
                else:
                    global_features = visual(batch)
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
