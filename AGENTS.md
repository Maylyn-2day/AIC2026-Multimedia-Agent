# AGENTS.md — AI Coding Agent Conventions

## Repository Overview

This is the **AIC2026-Multimedia-Agent** repository — an agentic AI multimedia retrieval system for the Ho Chi Minh City AI Challenge 2026 (AIC 2026).

## Architecture

- **Backend:** FastAPI (Python 3.11+) in `backend/` — RESTful API with 7 endpoints under `/v1/`
- **Frontend:** Streamlit unified single-page dashboard in `frontend/`
- **Vector DB:** Qdrant (standardized, no Milvus/Faiss)
- **Sparse DB:** Elasticsearch for OCR, ASR, Objects, Metadata
- **LLM Agent:** Gemini 2.0 Flash (primary) with OpenAI adapter fallback
- **Visual Encoder:** `google/siglip2-so400m-patch14-384` (primary) + OpenCLIP ViT-L/14 (fallback)

## Code Conventions

1. **Python version:** 3.11+ required.
2. **Type hints:** All function signatures must include type annotations.
3. **Pydantic:** All API request/response bodies use Pydantic v2 models in `backend/schemas/`.
4. **Async-first:** All FastAPI route handlers and service functions should be `async def`.
5. **Imports:** Use absolute imports from package root (`from backend.schemas.common import BaseResponse`).
6. **Linting:** Code must pass `ruff check` with the project's `pyproject.toml` configuration.
7. **Testing:** All new endpoints and services must have corresponding tests in `tests/`.
8. **Logging:** Use `backend.core.logging` — never use bare `print()` statements.
9. **Config:** All environment variables accessed via `backend.core.config.Settings`, never `os.getenv()` directly.
10. **Error handling:** Raise exceptions from `backend.core.exceptions` — never return raw error dicts.

## API Design Rules

- All endpoints versioned under `/v1/`.
- Request headers must include `Content-Type: application/json` and `X-Session-ID`.
- All responses follow the `BaseResponse` schema with `status`, `data`, `message`, `execution_time`.
- Error responses use standard HTTP codes (400, 401, 404, 500, 503) per `docs/api_contract.md`.

## File Ownership (M1–M5)

Refer to `docs/team_roles.md` for detailed ownership. In brief:
- **M1:** Agent, fusion, submission, chatbot sidebar
- **M2:** Reranker, VLM, sketch, keyframe cards
- **M3:** Temporal engine, timeline viewer
- **M4:** Offline pipeline, embedding, data ingestion scripts
- **M5:** DB clients, search services, filters, SOM grid, benchmarks
