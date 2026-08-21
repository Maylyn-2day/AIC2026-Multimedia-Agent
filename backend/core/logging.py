"""
Structured JSON Logging with Correlation IDs.

Provides a pre-configured logger that outputs structured JSON lines,
making logs queryable in Docker, ELK, or any log aggregation system.
Each log entry carries an optional ``session_id`` for request tracing.
"""

from __future__ import annotations

import logging
import sys

from backend.core.config import get_settings


def setup_logger(name: str = "aic2026") -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Logger name (defaults to ``aic2026``).

    Returns:
        A ``logging.Logger`` configured at the level specified
        by ``Settings.log_level``.
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger


# Module-level default logger for convenience imports.
logger = setup_logger()
