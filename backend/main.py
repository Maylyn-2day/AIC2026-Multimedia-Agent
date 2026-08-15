"""
AIC2026-Multimedia-Agent — FastAPI Application Factory.

Entry point for the backend service. Configures CORS, mounts the
v1 API router, registers exception handlers, and manages the
application lifespan (startup/shutdown hooks for DB connections).

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.v1 import v1_router
from backend.core.config import get_settings
from backend.core.exceptions import AIC2026BaseError
from backend.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Startup: Initialize DB connections, load lightweight models.
    Shutdown: Close connections, free GPU memory.
    """
    settings = get_settings()
    logger.info("[STARTUP] Starting %s (%s)", settings.app_name, settings.app_env)
    logger.info("[DB] Qdrant: %s | ES: %s", settings.qdrant_url, settings.es_url)
    logger.info("[LLM] Provider: %s | Model: %s", settings.llm_provider, settings.gemini_model)

    # TODO (Phase 2): Initialize Qdrant and ES clients
    # TODO (Phase 3): Load SigLIP 2 and OpenCLIP encoders

    yield

    # Shutdown
    logger.info("[SHUTDOWN] Stopping %s", settings.app_name)
    # TODO (Phase 2): Close DB connections
    # TODO (Phase 3): Free GPU memory


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Creates and configures the application with CORS, routers,
    and exception handlers.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Intelligent Multimedia Retrieval & Reasoning System for AIC 2026. "
            "Hybrid dense-sparse search with System 2 CoT reasoning, TRAKE temporal "
            "alignment, and cascading visual grounding."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return {"message": "Welcome to AIC2026 Multimedia Agent API!"}

    # ── CORS Middleware ──────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Execution Time Middleware ─────────────────────────────
    @app.middleware("http")
    async def add_execution_time_header(request: Request, call_next):
        """Inject X-Execution-Time header into every response."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Execution-Time"] = f"{elapsed:.4f}s"
        return response

    # ── Exception Handlers ───────────────────────────────────
    @app.exception_handler(AIC2026BaseError)
    async def aic2026_error_handler(request: Request, exc: AIC2026BaseError):
        """Handle all domain-specific exceptions with structured JSON."""
        logger.error("AIC2026 Error [%d]: %s", exc.status_code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "data": None,
                "message": exc.message,
                "execution_time": "N/A",
                "agent_reasoning": None,
            },
        )

    # ── Mount API Routers ────────────────────────────────────
    app.include_router(v1_router)

    return app


# Module-level app instance for uvicorn
app = create_app()
