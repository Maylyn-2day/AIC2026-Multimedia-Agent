"""Shared SigLIP 2 image encoder for offline indexing and online queries."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image

DEFAULT_MODEL_NAME = "ViT-gopt-16-SigLIP2-384"
DEFAULT_PRETRAINED_TAG = "webli"
SIGLIP2_EMBEDDING_DIM = 1536
CLIP_B32_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_B32_PRETRAINED = "openai"
CLIP_B32_EMBEDDING_DIM = 512


class CLIPB32Encoder:
    """Lazy OpenAI CLIP ViT-B/32 encoder for organizer-provided features."""

    def __init__(
        self,
        model_name: str = CLIP_B32_MODEL_NAME,
        pretrained: str = CLIP_B32_PRETRAINED,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._requested_device = device
        self._device = ""
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._load_lock = threading.Lock()

    @property
    def embedding_dim(self) -> int:
        """Dimension of organizer CLIP ViT-B/32 vectors."""
        return CLIP_B32_EMBEDDING_DIM

    def __call__(self, image: Image, /) -> np.ndarray:
        """Encode one PIL image into a normalized 512-d vector."""
        return self.encode_images([image])[0]

    def encode_images(self, images: Sequence[Image]) -> np.ndarray:
        """Encode a non-empty batch of PIL images."""
        if not images:
            raise ValueError("images must not be empty")
        self._ensure_loaded()

        import torch

        batch = torch.stack([self._preprocess(image.convert("RGB")) for image in images])
        with torch.inference_mode():
            vectors = self._model.encode_image(batch.to(self._device))
            vectors = torch.nn.functional.normalize(vectors, dim=-1).float().cpu().numpy()
        self._validate_dimension(vectors)
        return vectors

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text queries in the same space as organizer image features."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        self._ensure_loaded()

        import torch

        tokens = self._tokenizer(list(texts)).to(self._device)
        with torch.inference_mode():
            vectors = self._model.encode_text(tokens)
            vectors = torch.nn.functional.normalize(vectors, dim=-1).float().cpu().numpy()
        self._validate_dimension(vectors)
        return vectors

    def _validate_dimension(self, vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"{self._model_name} returned {vectors.shape[1]} dimensions; "
                f"expected {self.embedding_dim}"
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import open_clip
            import torch

            self._device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, _, self._preprocess = open_clip.create_model_and_transforms(
                self._model_name,
                pretrained=self._pretrained,
            )
            self._model = model.to(self._device).eval()
            self._tokenizer = open_clip.get_tokenizer(self._model_name)


class SigLIP2Encoder:
    """Lazy OpenCLIP SigLIP 2 encoder shared by indexing and search."""

    def __init__(
        self,
        weights: str | Path,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED_TAG,
        device: str | None = None,
    ) -> None:
        self._weights = Path(weights).expanduser()
        self._model_name = model_name
        self._pretrained = pretrained
        self._requested_device = device
        self._device = ""
        self._dtype: Any = None
        self._visual: Any = None
        self._preprocess: Any = None
        # ponytail: one model-wide lock is enough; revisit only if concurrent cold starts matter.
        self._load_lock = threading.Lock()

    @property
    def embedding_dim(self) -> int:
        """Dimension of global vectors written to Qdrant."""
        return SIGLIP2_EMBEDDING_DIM

    @property
    def device(self) -> str:
        """Resolved device after loading, or the requested device beforehand."""
        return self._device or self._requested_device or "auto"

    def __call__(self, image: Image, /) -> np.ndarray:
        """Encode one PIL image into a unit-normalized float32 vector."""
        vectors, _ = self.encode_batch([image])
        return vectors[0]

    def encode_batch(
        self,
        images: Sequence[Image],
        return_dense: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Encode PIL images and optionally return the final dense feature map."""
        if not images:
            raise ValueError("images must not be empty")
        self._ensure_loaded()

        import torch

        batch = torch.stack([self._preprocess(image.convert("RGB")) for image in images])
        batch = batch.to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            dense: np.ndarray | None = None
            if return_dense:
                result = self._visual.forward_intermediates(batch, indices=[-1], output_fmt="NCHW")
                global_features = result["image_features"]
                dense = result["image_intermediates"][0].float().cpu().numpy()
            else:
                global_features = self._visual(batch)
            global_features = torch.nn.functional.normalize(global_features, dim=-1)
            vectors = global_features.float().cpu().numpy()

        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"{self._model_name} returned {vectors.shape[1]} dimensions; "
                f"expected {self.embedding_dim}"
            )
        return vectors, dense

    def _ensure_loaded(self) -> None:
        if self._visual is not None:
            return
        with self._load_lock:
            if self._visual is not None:
                return
            self._load()

    def _load(self) -> None:
        import open_clip
        import torch
        from open_clip.model import _build_vision_tower
        from open_clip.transform import PreprocessCfg, image_transform_v2
        from safetensors import safe_open

        weights = self._weights.resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"SigLIP 2 weights not found: {weights}")

        config = open_clip.get_model_config(self._model_name)
        if config is None or config["embed_dim"] != self.embedding_dim:
            raise ValueError(f"Unsupported OpenCLIP model configuration: {self._model_name}")

        self._device = self._requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        visual = _build_vision_tower(config["embed_dim"], config["vision_cfg"]).to(dtype=self._dtype)
        with safe_open(weights, framework="pt", device="cpu") as checkpoint, torch.no_grad():
            missing = []
            keys = set(checkpoint.keys())
            for name, target in visual.state_dict().items():
                key = f"visual.{name}"
                if key not in keys:
                    missing.append(key)
                    continue
                target.copy_(checkpoint.get_tensor(key).to(dtype=target.dtype))
        if missing:
            raise ValueError(f"Missing SigLIP 2 visual weights: {missing[:5]}")

        pretrained = open_clip.get_pretrained_cfg(self._model_name, self._pretrained)
        self._preprocess = image_transform_v2(
            PreprocessCfg(
                size=384,
                mean=pretrained["mean"],
                std=pretrained["std"],
                interpolation=pretrained["interpolation"],
                resize_mode=pretrained["resize_mode"],
            ),
            is_train=False,
        )
        self._visual = visual.to(self._device).eval()


@lru_cache
def get_clip_b32_encoder() -> CLIPB32Encoder:
    """Return the process-wide CLIP encoder used by online search."""
    from backend.core.config import get_settings

    settings = get_settings()
    if settings.qdrant_vector_size != CLIP_B32_EMBEDDING_DIM:
        raise ValueError(
            f"QDRANT_VECTOR_SIZE must be {CLIP_B32_EMBEDDING_DIM} for {settings.openclip_model_id}"
        )
    return CLIPB32Encoder(
        model_name=settings.openclip_model_id,
        pretrained=settings.openclip_pretrained,
        device=settings.openclip_device or None,
    )
