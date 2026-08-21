# AIC2026-Multimedia-Agent — Comprehensive Technical Audit & Implementation Plan

> **Objective:** Generate a production-grade README.md, propose a complete file tree, audit repository completeness, and deliver a concrete 4-phase build roadmap — all synthesized from the technical documentation in `docs/`.

---

## 1. Deep Analysis & Synthesis of `docs/`

### 1.1 Documents Audited

| Document | Lines | Key Content |
|:---|:---:|:---|
| [api_contract.md](file:///d:/tlinh/AIchallenge/AIC2026-Multimedia-Agent/docs/api_contract.md) | 133 | 7 RESTful endpoints, error codes, RRF formula, VRAM strategy, System 2 Agent, submission format |
| [team_roles.md](file:///d:/tlinh/AIchallenge/AIC2026-Multimedia-Agent/docs/team_roles.md) | 75 | 5 team members, per-member AI/Backend + Frontend tasks, tech stack assignments |
| [Kien_Truc_He_Thong_Retrieval.md](file:///d:/tlinh/AIchallenge/AIC2026-Multimedia-Agent/docs/Kien_Truc_He_Thong_Retrieval.md) | 154 | Full system architecture, offline/online pipeline, model selection table, scoring formulas, 4-phase roadmap |

### 1.2 Key Technical Requirements Extracted

| Requirement | Specification |
|:---|:---|
| **API Framework** | FastAPI with `/v1/` versioning, auto-generated Swagger/ReDoc |
| **7 Endpoints** | `GET /v1/health`, `POST /v1/db/query`, `POST /v1/rerank/early-fusion`, `POST /v1/query/image-example`, `POST /v1/query/sketch`, `POST /v1/temporal/align`, `POST /v1/submission/submit` |
| **Latency Budgets** | Health <50ms, Query <400ms, Rerank <600ms, Image <100ms, Sketch <300ms, Temporal <200ms, Submit <100ms |
| **VRAM Strategy** | Cascading 3-stage: SigLIP2+BM25 → Grounding DINO → Qwen2.5-VL (7B, Top-5 only) |
| **RRF Algorithm** | `RRF_Score(d) = Σ 1/(60 + r_m(d))` across Dense (Qdrant) + Sparse (ES BM25) |
| **System 2 Agent** | CoT reasoning via **Gemini 2.0 Flash** (primary) with OpenAI adapter fallback, task routing (KIS/VQA/TRAKE) |
| **STAR/TRAKE** | Multi-stage temporal alignment: Q_past, Q_current, Q_future with index ordering constraint |
| **Scoring** | `Final Score = (1/5) Σ R@k` for k ∈ {1, 5, 20, 50, 100}; mandatory 100-result submission |
| **Vector DB** | **Qdrant** (standardized, HNSW index), ANN query <10ms |
| **Visual Encoder** | **OpenAI CLIP `ViT-B-32`** (512-d organizer baseline) + experimental SigLIP2 offline |
| **Sparse DB** | Elasticsearch for OCR, ASR, Objects JSON, YouTube Metadata |
| **UI Framework** | Streamlit **unified single-page dashboard** (Chatbot sidebar, SOM Grid center, Timeline/Sketch modals) |
| **Sprint Duration** | **10 days** strict schedule (Phase 1: D1–2, Phase 2: D3–5, Phase 3: D6–8, Phase 4: D9–10) |
| **Team Size** | 5 members with distinct ownership areas |

---

## 2. Proposed Changes

### 2.1 README.md Generation

#### [MODIFY] [README.md](file:///d:/tlinh/AIchallenge/AIC2026-Multimedia-Agent/README.md)

Replace the single-line placeholder with a production-grade README containing:
- Project title with emoji branding + status badges (Build, License, Python, FastAPI, Docker)
- One-paragraph executive summary
- Mermaid system architecture diagram showing Offline Pipeline → Dual DB → Online Retrieval → Agent → UI
- **Key Features** section with categorized capabilities
- **Target Metrics** table (latency budgets per endpoint)
- **API Endpoints** summary table (all 7 endpoints with method, path, description, latency)
- **Tech Stack** table (models, databases, frameworks)
- **Quickstart Guide** with Prerequisites, Clone, Environment Setup, Docker Compose, Run Backend/Frontend
- **Project Structure** (abbreviated tree)
- **Team Roles** table (5 members mapped to responsibility areas)
- **Development Phases** overview
- **Scoring Formula** section
- **Contributing** guidelines stub
- **License** and **Acknowledgments** (AIC 2026, LSC, VBS references)

### 2.2 Complete Directory Tree

#### [NEW] All Python files, configs, schemas, routes, services, and tests

The proposed tree below covers every single file needed for a production-ready system. Each file includes a 1-line technical comment and owner assignment (M1–M5 = Member 1–5).

```
AIC2026-Multimedia-Agent/
│
├── README.md                           # Production-grade project documentation (ALL)
├── LICENSE                             # MIT License (ALL)
├── .gitignore                          # Git exclusion rules (ALL) [EXISTS]
├── .env.example                        # Environment variable template with placeholder keys (M5)
├── docker-compose.yml                  # Orchestrates FastAPI + Qdrant + Elasticsearch + Streamlit (M5)
├── Makefile                            # Developer shortcuts: make dev, make test, make ingest (ALL)
├── pyproject.toml                      # PEP 621 project metadata, tool configs (ruff, pytest) (ALL)
├── AGENTS.md                           # AI coding agent instructions and repo conventions (ALL)
│
├── backend/
│   ├── requirements.txt                # Backend Python dependencies (FastAPI, torch, etc.) (M5)
│   ├── main.py                         # FastAPI app factory, CORS, lifespan, router mounts (M1)
│   │
│   ├── api/
│   │   ├── __init__.py                 # API package init (M1)
│   │   ├── deps.py                     # Shared FastAPI dependencies (DB sessions, model refs) (M5)
│   │   └── v1/
│   │       ├── __init__.py             # V1 router aggregator (M1)
│   │       ├── health.py               # GET /v1/health — heartbeat for Qdrant, ES, models (M5)
│   │       ├── query.py                # POST /v1/db/query — hybrid search (vector + BM25 + RRF) (M5)
│   │       ├── rerank.py               # POST /v1/rerank/early-fusion — Qwen2.5-VL VQA reranking (M2)
│   │       ├── image_query.py          # POST /v1/query/image-example — image-to-image search (M2)
│   │       ├── sketch.py               # POST /v1/query/sketch — sketch-to-image via ControlNet (M2)
│   │       ├── temporal.py             # POST /v1/temporal/align — TRAKE temporal alignment (M3)
│   │       └── submission.py           # POST /v1/submission/submit — competition result packaging (M1)
│   │
│   ├── schemas/
│   │   ├── __init__.py                 # Schemas package init (M1)
│   │   ├── common.py                   # BaseResponse, PaginationMeta, ErrorDetail Pydantic models (M1)
│   │   ├── query.py                    # HybridQueryRequest, QueryFilters, QueryResult schemas (M5)
│   │   ├── rerank.py                   # RerankRequest, VQAResponse, GroundingResult schemas (M2)
│   │   ├── image.py                    # ImageQueryRequest, SketchRequest schemas (M2)
│   │   ├── temporal.py                 # TemporalAlignRequest, TemporalSequence schemas (M3)
│   │   └── submission.py              # SubmissionPayload, SubmissionResult, TaskType enum (M1)
│   │
│   ├── services/
│   │   ├── __init__.py                 # Services package init (M1)
│   │   ├── vector_search.py            # Qdrant ANN search + mean-centering calibration (M5)
│   │   ├── sparse_search.py            # Elasticsearch BM25 queries with Boolean filters (M5)
│   │   ├── fusion.py                   # RRF (Reciprocal Rank Fusion) score computation engine (M1)
│   │   ├── reranker.py                 # Grounding DINO / OWL-ViT visual grounding service (M2)
│   │   ├── vlm_service.py             # Qwen2.5-VL inference for OCR, VQA, deep reasoning (M2)
│   │   ├── sketch_service.py           # ControlNet + SDXL-Turbo sketch-to-image pipeline (M2)
│   │   ├── temporal_engine.py          # Multi-stage temporal alignment (TRAKE) algorithm (M3)
│   │   ├── agent.py                    # System 2 CoT Agent — query routing, expansion, planning (M1)
│   │   ├── submission_service.py       # Competition result formatting and validation (M1)
│   │   └── embedding.py               # SigLIP 2 (so400m-patch14-384) + OpenCLIP fallback encoding (M4)
│   │
│   ├── core/
│   │   ├── __init__.py                 # Core package init (M1)
│   │   ├── config.py                   # Pydantic Settings: env vars, model paths, DB URIs (M5)
│   │   ├── logging.py                  # Structured JSON logging with correlation IDs (M1)
│   │   └── exceptions.py              # Custom exception hierarchy (VRAMOverflow, ModelNotLoaded) (M1)
│   │
│   └── db/
│       ├── __init__.py                 # Database package init (M5)
│       ├── qdrant_client.py            # Qdrant connection pool and collection management (M5)
│       ├── elasticsearch_client.py     # Elasticsearch connection and index management (M5)
│       └── session.py                  # Async context managers for DB lifecycle (M5)
│
├── frontend/
│   ├── requirements.txt                # Frontend dependencies (streamlit, Pillow, requests) (M1)
│   ├── app.py                          # Unified single-page dashboard: layout orchestrator (M1)
│   │                                   #   ┌─────────────────────────────────────────────┐
│   │                                   #   │ [LEFT SIDEBAR]    │  [CENTER MAIN AREA]     │
│   │                                   #   │ • Chatbot (M1)    │  • SOM Grid (M5)        │
│   │                                   #   │ • Filters (M5)    │  • Video-Grouped Grid   │
│   │                                   #   │ • Submit (M1)     │  • Keyframe Cards (M2)  │
│   │                                   #   │                   │  [MODALS / EXPANDERS]   │
│   │                                   #   │                   │  • Timeline Viewer (M3) │
│   │                                   #   │                   │  • Sketch Board (M2)    │
│   │                                   #   └─────────────────────────────────────────────┘
│   │
│   ├── components/
│   │   ├── __init__.py                 # Components package init (M1)
│   │   ├── chatbot_sidebar.py          # Agent chat + CoT trace in st.sidebar (M1)
│   │   ├── filter_sidebar.py           # Date/time, channel, object, OCR/ASR filters (M5)
│   │   ├── submission_sidebar.py       # KIS/VQA/TRAKE result packaging + submit btn (M1)
│   │   ├── som_grid.py                 # SOM 2D cluster + Video-Grouped Grid Layout (M5)
│   │   ├── keyframe_card.py            # Single keyframe display with bbox overlay (M2)
│   │   ├── timeline_modal.py           # Timeline Expansion Viewer ±30s (st.expander) (M3)
│   │   ├── sketch_modal.py             # Canvas sketch input + image drag-drop (M2)
│   │   └── video_strip.py             # Horizontal video timeline strip component (M3)
│   │
│   └── utils/
│       ├── __init__.py                 # Utils package init (M1)
│       ├── api_client.py               # HTTP client wrapper for backend API calls (M1)
│       └── image_utils.py             # Base64 encode/decode, thumbnail generation (M2)
│
├── configs/
│   ├── settings.yaml                   # Master configuration: model paths, DB URIs, thresholds (M5)
│   ├── prompts/
│   │   ├── system_prompt.txt           # System 2 Agent base system prompt template (M1)
│   │   ├── query_expansion.txt         # Query expansion prompt (Vi↔En translation) (M1)
│   │   └── trake_decomposition.txt    # TRAKE Q_past/Q_current/Q_future decomposition prompt (M3)
│   └── model_registry.yaml            # Model name → HF repo, VRAM budget, load priority (M4)
│
├── data/
│   ├── raw/                            # Raw BTC-provided data: keyframes/, videos/, objects/ (M4)
│   │   └── .gitkeep
│   ├── processed/                      # Extracted features: .npy vectors, OCR/ASR JSON (M4)
│   │   └── .gitkeep
│   ├── index/                          # Pre-built Qdrant snapshots and ES index dumps (M5)
│   │   └── .gitkeep
│   └── mock/                          # Mock JSON responses for Phase 1 frontend development (M1)
│       ├── health_response.json        # Mock /v1/health response (M1)
│       ├── query_response.json         # Mock /v1/db/query response with sample keyframes (M1)
│       ├── rerank_response.json        # Mock /v1/rerank/early-fusion response (M1)
│       └── temporal_response.json     # Mock /v1/temporal/align response (M1)
│
├── docker/
│   ├── Dockerfile.backend              # Multi-stage build: Python 3.11 + CUDA + FastAPI (M5)
│   ├── Dockerfile.frontend             # Streamlit container image (M5)
│   └── docker-compose.override.yml    # Dev overrides: volume mounts, hot-reload, GPU passthrough (M5)
│
├── models/
│   ├── .gitkeep                        # Placeholder — model weights downloaded at runtime (M4)
│   └── download_models.py            # Script to download SigLIP2, Qwen2.5-VL, CLIP from HF (M4)
│
├── scripts/
│   ├── ingest_keyframes.py             # Batch extract + encode keyframes → Qdrant vectors (M4)
│   ├── ingest_metadata.py              # Load OCR/ASR/Objects JSON → Elasticsearch indices (M5)
│   ├── extract_ocr.py                  # Run Qwen2.5-VL OCR on keyframe dataset (M4)
│   ├── extract_asr.py                  # Run Whisper/PhoWhisper ASR on video audio tracks (M4)
│   ├── benchmark.py                    # Compute Recall@K, MAP, latency on BTC sample data (M5)
│   └── seed_mock_data.py             # Generate mock JSON fixtures for frontend dev (M1)
│
├── tests/
│   ├── conftest.py                     # Pytest fixtures: test client, mock DB, sample data (ALL)
│   ├── test_health.py                  # Health endpoint integration test (M5)
│   ├── test_query.py                   # Hybrid query endpoint unit + integration tests (M5)
│   ├── test_rerank.py                  # Reranking pipeline test with mock VLM (M2)
│   ├── test_temporal.py                # TRAKE temporal alignment algorithm correctness (M3)
│   ├── test_submission.py              # Submission format validation tests (M1)
│   ├── test_fusion.py                  # RRF score computation unit tests (M1)
│   └── test_agent.py                  # Agent routing + CoT trace tests (M1)
│
└── docs/
    ├── api_contract.md                 # RESTful API specification (7 endpoints) [EXISTS]
    ├── team_roles.md                   # Team member assignments & responsibilities [EXISTS]
    ├── Kien_Truc_He_Thong_Retrieval.md # System architecture & retrieval pipeline [EXISTS]
    └── .gitkeep                        # [EXISTS]
```

### 2.3 Repository Completeness Audit — Missing Root Files

| File | Status | Purpose |
|:---|:---:|:---|
| `README.md` | ⚠️ Stub only | Needs full replacement |
| `.env.example` | ❌ Missing | Template for environment variables (API keys, DB URIs, model paths) |
| `docker-compose.yml` | ❌ Missing | Service orchestration (backend, frontend, qdrant, elasticsearch) |
| `pyproject.toml` | ❌ Missing | PEP 621 project metadata + tool configs (ruff, pytest, mypy) |
| `Makefile` | ❌ Missing | Developer shortcuts (`make dev`, `make test`, `make ingest`, `make lint`) |
| `AGENTS.md` | ❌ Missing | AI coding agent conventions and repo-specific rules |
| `backend/requirements.txt` | ❌ Missing | Backend Python dependencies |
| `frontend/requirements.txt` | ❌ Missing | Frontend Python dependencies |
| `docker/Dockerfile.backend` | ❌ Missing | Backend container image |
| `docker/Dockerfile.frontend` | ❌ Missing | Frontend container image |
| `.gitignore` | ✅ Exists | Comprehensive Python gitignore already in place |
| `LICENSE` | ✅ Exists | MIT License |

---

## 3. Execution Plan — What Will Be Created Now

Given the scope of this request, I will generate the following files:

1. **`README.md`** — Full production-grade README (the primary deliverable)
2. **Implementation plan artifact** (this document) — covering analysis, tree, audit, and roadmap

> [!IMPORTANT]
> The README.md will be written **directly to the repository root**. All other files in the proposed tree (Python modules, configs, Docker files, etc.) are **documented here for future phased implementation** and will NOT be created in this step.

---

## 4. Technical Implementation Roadmap (4 Phases — 10-Day Sprint)

> [!IMPORTANT]
> **Schedule:** Compressed to a strict **10-day sprint** per `team_roles.md` constraints.

### Phase 1 — Mock Data & FastAPI Skeleton & Docker (Days 1–2)

**Goal:** Fully functional API skeleton with mock responses; Docker infrastructure up; unified dashboard shell renders with mock data.

| Step | Task | Owner | Deliverable |
|:---:|:---|:---:|:---|
| 1.1 | Create `backend/main.py` with FastAPI app factory, CORS, lifespan hooks | M1 | Running FastAPI server |
| 1.2 | Define all Pydantic schemas in `backend/schemas/` | M1 | Type-safe request/response contracts |
| 1.3 | Implement 7 route stubs in `backend/api/v1/` returning mock JSON | M1+M5 | Swagger UI with all endpoints |
| 1.4 | Generate mock JSON fixtures in `data/mock/` | M1 | Frontend test data |
| 1.5 | Create `backend/core/config.py` with Pydantic Settings | M5 | `.env.example` + config loading |
| 1.6 | Write `docker-compose.yml` for **Qdrant** + Elasticsearch + FastAPI | M5 | `docker compose up` works |
| 1.7 | Build `frontend/app.py` as unified single-page dashboard shell | M1 | Streamlit dashboard launches |
| 1.8 | Wire `components/som_grid.py` + `chatbot_sidebar.py` with mock data | M1+M5 | Dashboard layout visible |
| 1.9 | Write initial `conftest.py` + `test_health.py` | ALL | CI-ready test suite |

**Exit Criteria:** `docker compose up` starts all services; all 7 endpoints return mock 200 responses; Streamlit renders unified dashboard with sidebar chatbot + center grid using mock keyframes.

---

### Phase 2 — DB Ingestion & Hybrid Search RRF (Days 3–5)

**Goal:** Real data flows through the pipeline; hybrid search (Qdrant vectors + ES BM25 + RRF) operational.

| Step | Task | Owner | Deliverable |
|:---:|:---|:---:|:---|
| 2.1 | Build `scripts/ingest_keyframes.py`: extract semantic keyframes (AutoShot + L1 filter) | M4 | Deduplicated keyframe set |
| 2.2 | Ingest organizer CLIP ViT-B/32 vectors; benchmark SigLIP2 separately | M4 | `data/processed/*.npy` |
| 2.3 | Implement `backend/db/qdrant_client.py`: create collection, upsert vectors | M5 | Vectors searchable in Qdrant |
| 2.4 | Build `scripts/ingest_metadata.py`: load Objects JSON + YouTube metadata → ES | M5 | ES indices populated |
| 2.5 | Run `scripts/extract_ocr.py` (Qwen2.5-VL) + `scripts/extract_asr.py` (Whisper) | M4 | OCR/ASR text in ES |
| 2.6 | Implement `backend/services/vector_search.py` with mean-centering (Qdrant-only) | M5 | Dense retrieval working |
| 2.7 | Implement `backend/services/sparse_search.py` with Boolean filters | M5 | Sparse retrieval working |
| 2.8 | Implement `backend/services/fusion.py` with RRF (k=60) | M1 | Hybrid search endpoint live |
| 2.9 | Replace mock `/v1/db/query` with real hybrid search pipeline | M5 | End-to-end query works |
| 2.10 | Wire `components/filter_sidebar.py` into dashboard with live filters | M5 | Real filter-driven search |
| 2.11 | Run `scripts/benchmark.py` for Recall@K baseline | M5 | Baseline metrics documented |

**Exit Criteria:** `/v1/db/query` returns real keyframe results from Qdrant+ES with RRF fusion; Recall@100 > 0.5 on BTC sample data; search latency <400ms.

---

### Phase 3 — System 2 Agent & Visual Grounding & TRAKE (Days 6–8)

**Goal:** System 2 Agent routes tasks via Gemini 2.0 Flash; visual grounding and TRAKE temporal alignment functional; dashboard components wired to live endpoints.

| Step | Task | Owner | Deliverable |
|:---:|:---|:---:|:---|
| 3.1 | Implement `backend/services/agent.py`: CoT router with **Gemini 2.0 Flash** (+ OpenAI adapter) | M1 | Agent classifies KIS/VQA/TRAKE |
| 3.2 | Build query expansion (Vi↔En) in agent with `configs/prompts/` templates | M1 | Expanded queries improve recall |
| 3.3 | Implement conversational KIS with buffer memory | M1 | Multi-turn search sessions |
| 3.4 | Build `backend/services/reranker.py`: Grounding DINO visual grounding | M2 | Bbox verification on Top-50 |
| 3.5 | Build `backend/services/vlm_service.py`: Qwen2.5-VL for VQA + deep reasoning | M2 | `/v1/rerank/early-fusion` live |
| 3.6 | Build `backend/services/sketch_service.py`: ControlNet sketch-to-image | M2 | `/v1/query/sketch` live |
| 3.7 | Implement `/v1/query/image-example` with SigLIP 2 re-query | M2 | Image-to-image search works |
| 3.8 | Build `backend/services/temporal_engine.py`: TRAKE multi-stage alignment | M3 | `/v1/temporal/align` live |
| 3.9 | Implement TRAKE scoring: `S_final = w_c·Score(r_c) + w_p·Score(r_p) + w_n·Score(r_n)` | M3 | Temporal sequences ranked |
| 3.10 | Wire `components/chatbot_sidebar.py` to live Agent API with CoT trace | M1 | Agent reasoning visible in sidebar |
| 3.11 | Wire `components/sketch_modal.py` with canvas + bbox overlay | M2 | Sketch search in dashboard modal |
| 3.12 | Wire `components/timeline_modal.py` with ±30s expansion | M3 | TRAKE results in dashboard expander |
| 3.13 | Write `test_rerank.py`, `test_temporal.py`, `test_agent.py` | M1+M2+M3 | Core algorithm tests pass |

**Exit Criteria:** Agent correctly routes KIS/VQA/TRAKE queries via Gemini 2.0 Flash; Qwen2.5-VL reranking improves Top-5 precision by ≥15%; TRAKE returns temporally ordered frame sequences; VRAM stays under budget.

---

### Phase 4 — Streamlit Dashboard Integration & E2E Testing (Days 9–10)

**Goal:** Competition-ready unified dashboard; full E2E testing; rehearsal drills.

| Step | Task | Owner | Deliverable |
|:---:|:---|:---:|:---|
| 4.1 | Polish unified dashboard: ensure all components render cohesively | M1 | Seamless single-page UX |
| 4.2 | Finalize `components/som_grid.py` with Video-Grouped Grid + SOM clustering | M5 | Professional results display |
| 4.3 | Implement keyboard shortcuts for rapid submission | M1 | Speed-optimized workflow |
| 4.4 | Build `/v1/submission/submit` with BTC format validation | M1 | Competition submission works |
| 4.5 | Implement "Fill 100" strategy: always submit 100 ranked results | M1 | R@50–R@100 maximized |
| 4.6 | Full E2E test suite: query → retrieve → rerank → display → submit | ALL | All pipelines validated |
| 4.7 | Performance profiling: latency per endpoint, VRAM peak measurement | M5 | All budgets met |
| 4.8 | Competition rehearsal: simulate 10 timed query sessions | ALL | Team workflow optimized |
| 4.9 | Documentation finalization: update README metrics, API docs | ALL | Repo presentation-ready |

**Exit Criteria:** Full query-to-submission pipeline completes in <2s; all 7 endpoints meet latency budgets; VRAM peak <8GB on laptop GPU; team completes rehearsal queries within competition time limits.

---

## Resolved Design Decisions

> [!NOTE]
> **Vector DB → Qdrant (100%):** Standardized on Qdrant for all vector storage and ANN search. Lighter footprint for local/laptop development, excellent Docker support, and native HNSW indexing. All references to Milvus/Faiss removed from codebase.

> [!NOTE]
> **LLM Provider → Gemini 2.0 Flash (Primary):** `backend/services/agent.py` implements a provider adapter pattern with Gemini 2.0 Flash as the default. OpenAI o1-mini is supported as a config-driven fallback via `configs/settings.yaml` → `agent.llm_provider: "gemini" | "openai"`.

> [!NOTE]
> **Competition baseline → OpenAI CLIP `ViT-B-32`:** Online image/text queries use normalized 512-dimensional vectors matching the organizer features. OpenCLIP `ViT-gopt-16-SigLIP2-384` remains an isolated 1536-dimensional offline experiment and must use a separate Qdrant collection if enabled later.

---

## Verification Plan

### Automated Tests
- `pytest tests/` — all unit and integration tests pass
- Latency benchmarks via `scripts/benchmark.py`
- Docker Compose smoke test: `docker compose up --build` succeeds

### Manual Verification
- Swagger UI at `http://localhost:8000/docs` shows all 7 endpoints
- Streamlit dashboard at `http://localhost:8501` renders unified single-page layout (sidebar + grid + modals)
- VRAM monitoring during inference (`nvidia-smi`)
