from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from backend.services.embedding import (
    CLIP_B32_EMBEDDING_DIM,
    CLIPB32Encoder,
    SIGLIP2_EMBEDDING_DIM,
    SigLIP2Encoder,
)


class _FakeVisual:
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        vectors = torch.zeros((len(batch), SIGLIP2_EMBEDDING_DIM), dtype=torch.float32)
        vectors[:, 0] = 3
        vectors[:, 1] = 4
        return vectors


class _FakeCLIP:
    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return self._vectors(len(batch))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._vectors(len(tokens))

    @staticmethod
    def _vectors(count: int) -> torch.Tensor:
        vectors = torch.zeros((count, CLIP_B32_EMBEDDING_DIM), dtype=torch.float32)
        vectors[:, 0] = 3
        vectors[:, 1] = 4
        return vectors


def _loaded_test_encoder() -> SigLIP2Encoder:
    encoder = SigLIP2Encoder("unused.safetensors", device="cpu")
    encoder._device = "cpu"
    encoder._dtype = torch.float32
    encoder._preprocess = lambda image: torch.zeros((3, 384, 384), dtype=torch.float32)
    encoder._visual = _FakeVisual()
    return encoder


def _loaded_clip_encoder() -> CLIPB32Encoder:
    encoder = CLIPB32Encoder(device="cpu")
    encoder._device = "cpu"
    encoder._model = _FakeCLIP()
    encoder._preprocess = lambda image: torch.zeros((3, 224, 224), dtype=torch.float32)
    encoder._tokenizer = lambda texts: torch.zeros((len(texts), 77), dtype=torch.int64)
    return encoder


def test_clip_b32_encoder_matches_organizer_dimension() -> None:
    encoder = _loaded_clip_encoder()
    image_vector = encoder(Image.new("RGB", (16, 16)))
    text_vector = encoder.encode_texts(["a red car"])[0]

    assert image_vector.shape == (CLIP_B32_EMBEDDING_DIM,)
    assert text_vector.shape == (CLIP_B32_EMBEDDING_DIM,)
    assert np.linalg.norm(image_vector) == pytest.approx(1.0)
    assert np.linalg.norm(text_vector) == pytest.approx(1.0)


def test_siglip2_encoder_returns_normalized_1536_vector() -> None:
    vector = _loaded_test_encoder()(Image.new("RGB", (16, 16)))

    assert vector.shape == (SIGLIP2_EMBEDDING_DIM,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_siglip2_encoder_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _loaded_test_encoder().encode_batch([])
