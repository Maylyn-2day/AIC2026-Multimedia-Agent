from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from backend.services.clip_text_encoder import OpenAIClipTextEncoder, _OfficialOpenAIClipBackend


class FakeBackend:
    def encode_text(self, text: str) -> np.ndarray:
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = len(text)
        return vector


class FakeTokens:
    def to(self, device: str) -> FakeTokens:
        assert device == "cpu"
        return self


class FakeClipModule:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def tokenize(self, texts: list[str], *, truncate: bool) -> FakeTokens:
        self.calls.append((texts.copy(), truncate))
        return FakeTokens()


class FakeTensor:
    def __getitem__(self, index: int) -> FakeTensor:
        assert index == 0
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return np.ones(512, dtype=np.float32)


class FakeModel:
    def eval(self) -> None:
        pass

    def encode_text(self, tokens: FakeTokens) -> FakeTensor:
        return FakeTensor()


class FakeInferenceMode:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        pass


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorchModule:
    cuda = FakeCuda()

    @staticmethod
    def inference_mode() -> FakeInferenceMode:
        return FakeInferenceMode()


def test_encoder_is_lazy_normalized_and_finite() -> None:
    loads = 0

    def load() -> FakeBackend:
        nonlocal loads
        loads += 1
        return FakeBackend()

    encoder = OpenAIClipTextEncoder(load)
    assert loads == 0
    result = encoder.encode("  query  ")
    assert loads == 1
    assert result.shape == (512,)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1.0)


@pytest.mark.parametrize("query", ["một người đang nói", "đây là truy vấn tiếng Việt rất dài " * 100])
def test_official_backend_uses_deterministic_token_truncation(query: str) -> None:
    clip_module = FakeClipModule()
    backend = _OfficialOpenAIClipBackend(
        None,
        clip_module=clip_module,
        torch_module=FakeTorchModule(),
        model=FakeModel(),
    )
    original = query[:]
    result = OpenAIClipTextEncoder(lambda: backend).encode(query)
    assert query == original
    assert clip_module.calls == [([query.strip()], True)]
    assert result.shape == (512,)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1.0)


@pytest.mark.parametrize("text", ["", " ", "\n"])
def test_encoder_rejects_blank(text: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        OpenAIClipTextEncoder(FakeBackend).encode(text)


def test_concurrent_load_happens_once() -> None:
    loads = 0

    def load() -> FakeBackend:
        nonlocal loads
        loads += 1
        return FakeBackend()

    encoder = OpenAIClipTextEncoder(load)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(encoder.encode, ["query"] * 32))
    assert loads == 1


def test_multiple_query_variants_load_model_once() -> None:
    loads = 0

    def load() -> FakeBackend:
        nonlocal loads
        loads += 1
        return FakeBackend()

    encoder = OpenAIClipTextEncoder(load)
    variants = ("spacecraft launch", "four astronauts", "polar aurora")
    results = [encoder.encode(variant) for variant in variants]
    assert loads == 1
    assert len(results) == len(variants)
