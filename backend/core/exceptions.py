"""
Custom Exception Hierarchy for AIC2026.

Provides domain-specific exceptions that map to HTTP error codes,
enabling clean error handling in FastAPI route handlers.
"""

from __future__ import annotations


class AIC2026BaseError(Exception):
    """Base exception for all AIC2026 application errors."""

    def __init__(self, message: str = "An internal error occurred.", status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotLoadedError(AIC2026BaseError):
    """Raised when a required AI model is not loaded into memory."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=f"Model '{model_name}' is not loaded. Check VRAM budget or model path.",
            status_code=503,
        )


class VRAMOverflowError(AIC2026BaseError):
    """Raised when an operation would exceed the configured VRAM budget."""

    def __init__(self, required_gb: float, available_gb: float) -> None:
        super().__init__(
            message=f"VRAM overflow: {required_gb:.1f}GB required but only {available_gb:.1f}GB available.",
            status_code=503,
        )


class DatabaseConnectionError(AIC2026BaseError):
    """Raised when Qdrant or Elasticsearch is unreachable."""

    def __init__(self, service: str) -> None:
        super().__init__(
            message=f"Failed to connect to {service}. Check Docker containers.",
            status_code=503,
        )


class InvalidQueryError(AIC2026BaseError):
    """Raised when a user query fails validation beyond Pydantic checks."""

    def __init__(self, detail: str) -> None:
        super().__init__(message=f"Invalid query: {detail}", status_code=400)


class SubmissionFormatError(AIC2026BaseError):
    """Raised when submission payload does not match BTC required format."""

    def __init__(self, detail: str) -> None:
        super().__init__(message=f"Submission format error: {detail}", status_code=400)
