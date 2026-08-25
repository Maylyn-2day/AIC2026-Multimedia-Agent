"""Lazy official OpenAI CLIP ViT-B/32 text encoding."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class TextEncoder(Protocol):
    def encode(self, text: str) -> NDArray[np.float32]: ...


class ClipBackend(Protocol):
    def encode_text(self, text: str) -> NDArray[np.floating]: ...


def _normalize(vector: NDArray[np.floating]) -> NDArray[np.float32]:
    result = np.asarray(vector, dtype=np.float32).reshape(-1)
    if result.shape != (512,) or not np.isfinite(result).all():
        raise ValueError("OpenAI CLIP output must contain 512 finite values")
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("OpenAI CLIP output must be non-zero")
    return np.asarray(result / norm, dtype=np.float32)


class _OfficialOpenAIClipBackend:
    """OpenAI CLIP adapter using its deterministic 77-token truncation."""

    def __init__(
        self,
        model_cache: Path | None,
        *,
        clip_module: Any | None = None,
        torch_module: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if clip_module is None:
            import clip as clip_module
        if torch_module is None:
            import torch as torch_module

        self._clip = clip_module
        self._torch = torch_module
        self.device = "cuda" if torch_module.cuda.is_available() else "cpu"
        if model is None:
            model, _ = clip_module.load(
                "ViT-B/32",
                device=self.device,
                download_root=None if model_cache is None else str(model_cache),
            )
        self.model = model
        self.model.eval()

    def encode_text(self, text: str) -> NDArray[np.floating]:
        # Official OpenAI CLIP has a 77-token context. Its tokenizer performs
        # deterministic token-level truncation without slicing Unicode text.
        tokens = self._clip.tokenize([text], truncate=True).to(self.device)
        with self._torch.inference_mode():
            return self.model.encode_text(tokens)[0].float().cpu().numpy()


class OpenAIClipTextEncoder:
    """Thread-safe lazy wrapper around official OpenAI CLIP ``ViT-B/32``."""

    def __init__(
        self,
        backend_loader: Callable[[], ClipBackend] | None = None,
        *,
        model_cache: Path | None = None,
    ) -> None:
        self._backend_loader = backend_loader or (lambda: _OfficialOpenAIClipBackend(model_cache))
        self._backend: ClipBackend | None = None
        self._lock = threading.Lock()

    def _get_backend(self) -> ClipBackend:
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    self._backend = self._backend_loader()
        return self._backend

    def encode(self, text: str) -> NDArray[np.float32]:
        if not isinstance(text, str) or not (query := text.strip()):
            raise ValueError("text must not be blank")
        return _normalize(self._get_backend().encode_text(query))
