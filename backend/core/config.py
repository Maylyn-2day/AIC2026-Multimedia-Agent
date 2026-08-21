"""
Application Configuration via Pydantic Settings.

Loads environment variables from `.env` file and provides typed,
validated configuration for all system components: database URIs,
model identifiers, VRAM budgets, and RRF hyperparameters.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized, type-safe application configuration.

    All values are loaded from environment variables (or `.env` file).
    Defaults are tuned for local development on a single laptop GPU.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "AIC2026-Multimedia-Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ── FastAPI Backend ──────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_workers: int = 1
    cors_origins: list[str] = Field(default=["http://localhost:8501"])

    # ── Qdrant (Vector Database) ─────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection_name: str = "aic2026_keyframes"
    qdrant_vector_size: int = 512

    # ── Elasticsearch (Sparse Search) ────────────────────────
    es_host: str = "localhost"
    es_port: int = 9200
    es_index_ocr: str = "aic2026_ocr"
    es_index_asr: str = "aic2026_asr"
    es_index_objects: str = "aic2026_objects"
    es_index_metadata: str = "aic2026_metadata"

    # ── Model Paths ──────────────────────────────────────────
    siglip2_model_id: str = "ViT-gopt-16-SigLIP2-384"
    siglip2_pretrained: str = "webli"
    siglip2_weights_path: str = ""
    siglip2_device: str = ""
    openclip_model_id: str = "ViT-B-32-quickgelu"
    openclip_pretrained: str = "openai"
    openclip_device: str = ""
    qwen_vl_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"

    # ── LLM Agent ────────────────────────────────────────────
    llm_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    openai_api_key: str = ""
    openai_model: str = "o1-mini"

    # ── Data Paths ───────────────────────────────────────────
    data_raw_dir: str = "./data/raw"
    data_processed_dir: str = "./data/processed"
    data_index_dir: str = "./data/index"
    data_mock_dir: str = "./data/mock"
    keyframes_dir: str = "./data/raw/keyframes"

    # ── VRAM Budget ──────────────────────────────────────────
    max_vram_gb: float = 8.0
    rerank_top_k: int = 50
    deep_reasoning_top_k: int = 5

    # ── RRF Fusion ───────────────────────────────────────────
    rrf_k: int = 60

    @property
    def qdrant_url(self) -> str:
        """Construct the Qdrant REST API URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def es_url(self) -> str:
        """Construct the Elasticsearch URL."""
        return f"http://{self.es_host}:{self.es_port}"


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached singleton of the application settings.

    Uses ``lru_cache`` to ensure the `.env` file is read only once
    per process lifetime, avoiding redundant I/O on every request.
    """
    return Settings()
