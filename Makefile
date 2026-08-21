# ============================================================
# AIC2026-Multimedia-Agent — Makefile
# ============================================================
# Developer shortcuts for common tasks.
# Usage: make <target>
# ============================================================

.PHONY: help dev backend frontend test lint format infra down ingest clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ──────────────────────────────────────────────

dev: infra backend frontend ## Start full dev stack (DBs + Backend + Frontend)

backend: ## Start FastAPI backend with hot-reload
	cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

frontend: ## Start Streamlit frontend
	cd frontend && streamlit run app.py --server.port 8501

# ── Infrastructure ───────────────────────────────────────────

infra: ## Start Qdrant + Elasticsearch via Docker
	docker compose up -d qdrant es

down: ## Teardown all Docker services
	docker compose down -v

# ── Quality ──────────────────────────────────────────────────

test: ## Run pytest test suite
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=backend --cov-report=html --cov-report=term-missing

lint: ## Run ruff linter
	ruff check backend/ frontend/ scripts/ tests/

format: ## Auto-format code with ruff
	ruff format backend/ frontend/ scripts/ tests/

# ── Data Pipeline ────────────────────────────────────────────

ingest: ## Run full data ingestion pipeline
	python scripts/ingest_keyframes.py --data-dir data/raw/keyframes/
	python scripts/ingest_metadata.py --data-dir data/raw/

benchmark: ## Run Recall@K benchmark
	python scripts/benchmark.py --eval-set data/raw/eval/

mock: ## Generate mock JSON fixtures
	python scripts/seed_mock_data.py

# ── Cleanup ──────────────────────────────────────────────────

clean: ## Remove cached and generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
