"""
Shared FastAPI Dependencies.

Provides dependency-injected access to database sessions,
configuration, and model references used across all route handlers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Header, HTTPException

from backend.core.config import Settings, get_settings


async def get_config() -> Settings:
    """Inject the application settings singleton."""
    return get_settings()


async def require_session_id(
    x_session_id: Annotated[str | None, Header()] = None,
) -> str:
    """
    Extract and validate the ``X-Session-ID`` header.

    Required by the API contract for request tracing and
    conversational context tracking.
    """
    if not x_session_id:
        raise HTTPException(
            status_code=401,
            detail="Missing required header: X-Session-ID",
        )
    return x_session_id
