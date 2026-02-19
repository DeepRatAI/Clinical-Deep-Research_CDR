# CDR Makefile — Clinical Deep Research
# ============================================================================
# Usage:
#   make setup        — Create venv + install all dependencies
#   make sync         — Install deps from uv.lock (reproducible, CI)
#   make test         — Run backend tests
#   make test-ui      — Run frontend tests
#   make eval         — Run evaluation harness
#   make demo         — Generate offline sample outputs (no API keys)
#   make demo-online  — Run 5 real pipeline queries (needs API keys)
#   make figures      — Generate evaluation charts
#   make lint         — Lint Python + check types
#   make format       — Auto-format code
#   make run          — Start API server (production)
#   make dev          — Start API server (dev mode with reload)
#   make clean        — Remove build artifacts
#   make help         — Show this help
# ============================================================================

.PHONY: help setup install sync test test-ui test-all eval lint format run dev \
        server clean docker-build docker-up docker-down check-env \
        test-coherence test-gates validate-run gate-report \
        demo demo-online figures

SHELL := /bin/bash
PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := PYTHONPATH=src $(VENV)/bin/pytest
UVICORN := PYTHONPATH=src $(VENV)/bin/uvicorn
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
BLACK := $(VENV)/bin/black

# Detect if we're already inside a venv
ifdef VIRTUAL_ENV
  PIP := pip
  PYTEST := PYTHONPATH=src pytest
  UVICORN := PYTHONPATH=src uvicorn
  RUFF := ruff
  MYPY := mypy
  BLACK := black
endif

# Default target
help:
	@echo ""
	@echo "CDR — Clinical Deep Research  (v0.1 Open Alpha)"
	@echo "================================================"
	@echo ""
	@echo "  Setup"
	@echo "    make setup          Create venv + install all deps"
	@echo "    make sync           Install from uv.lock (frozen, CI)"
	@echo "    make install        Install deps into current env"
	@echo "    make check-env      Verify .env is configured"
	@echo ""
	@echo "  Testing"
	@echo "    make test           Run backend tests (pytest)"
	@echo "    make test-ui        Run frontend tests (vitest)"
	@echo "    make test-all       Run backend + frontend tests"
	@echo "    make test-coherence Run DoD3 coherence harness"
	@echo "    make test-gates     Run evidence gates tests"
	@echo ""
	@echo "  Evaluation"
	@echo "    make eval           Run evaluation on golden set"
	@echo ""
	@echo "  Demo & Assets"
	@echo "    make demo           Offline demo (sample_report.json → .md)"
	@echo "    make demo-online    5 real pipeline runs (needs API keys)"
	@echo "    make figures        Generate evaluation charts (latency)"
	@echo ""
	@echo "  Quality"
	@echo "    make lint           Lint (ruff + mypy)"
	@echo "    make format         Auto-format (black + ruff fix)"
	@echo ""
	@echo "  Run"
	@echo "    make run            Start API server (port 8000)"
	@echo "    make dev            Start API server (dev + reload)"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build   Build Docker images"
	@echo "    make docker-up      Start all services"
	@echo "    make docker-down    Stop all services"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean          Remove build artifacts"
	@echo ""

# ============================================================================
# Setup
# ============================================================================

setup: $(VENV)/bin/activate install
	@echo ""
	@echo "✅  Setup complete."
	@echo "    Activate your venv:  source $(VENV)/bin/activate"
	@echo "    Copy env file:       cp .env.example .env"
	@echo "    Start server:        make run"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

sync:
	@command -v uv >/dev/null 2>&1 || { echo "❌  uv not found. Install: pip install uv"; exit 1; }
	uv sync --frozen
	@echo "✅  Dependencies installed from uv.lock (frozen)"

check-env:
	@if [ ! -f .env ]; then \
		echo "❌  .env not found. Copy the example:"; \
		echo "    cp .env.example .env"; \
		exit 1; \
	fi
	@echo "✅  .env exists"
	@grep -q "HF_TOKEN=hf_" .env 2>/dev/null && echo "✅  HF_TOKEN configured" \
		|| echo "⚠️   HF_TOKEN not set (required for LLM calls)"

# ============================================================================
# Testing
# ============================================================================

test:
	$(PYTEST) tests/ -v --tb=short $(ARGS)

test-ui:
	cd ui && npm ci --silent && npm test -- --run

test-all: test test-ui
	@echo "✅  All tests passed (backend + frontend)"

test-coherence:
	@echo "=============================================="
	@echo "DoD3 Coherence Validation Harness"
	@echo "=============================================="
	PYTHONPATH=src $(PYTHON) -m cdr.evaluation.semantic_harness
	@echo ""

test-gates:
	$(PYTEST) tests/test_evidence_gates.py tests/test_dod3_gates.py -v

# ============================================================================
# Evaluation
# ============================================================================

eval:
	@echo "=============================================="
	@echo "CDR Evaluation — Golden Set (5 questions)"
	@echo "=============================================="
	PYTHONPATH=src $(PYTHON) -m eval.eval_runner \
		--dataset eval/datasets/golden_set_toy.json \
		--output eval/results/
	@echo ""
	@echo "Results in eval/results/"

# ============================================================================
# Quality
# ============================================================================

lint:
	$(RUFF) check src/ tests/
	$(MYPY) src/

format:
	$(BLACK) src/ tests/
	$(RUFF) check --fix src/ tests/

# ============================================================================
# Run
# ============================================================================

run: check-env
	$(UVICORN) cdr.api.routes:app --host 0.0.0.0 --port 8000

dev:
	$(UVICORN) cdr.api.routes:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Alias for backwards compatibility
server: run

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo "⏳  Waiting for health checks..."
	@sleep 5
	@docker compose ps
	@echo ""
	@echo "API:  http://localhost:8000/docs"
	@echo "UI:   http://localhost:5173"

docker-down:
	docker compose down

# ============================================================================
# Demo & Assets
# ============================================================================

demo:
	@echo "=============================================="
	@echo "CDR Demo — Offline Sample Generation"
	@echo "=============================================="
	PYTHONPATH=src $(PYTHON) scripts/generate_demo.py
	@echo ""
	@echo "Outputs:"
	@echo "  examples/output/sample_report.json"
	@echo "  examples/output/sample_report.md"

demo-online: check-env
	@echo "=============================================="
	@echo "CDR Demo — Online Pipeline (5 queries)"
	@echo "=============================================="
	PYTHONPATH=src $(PYTHON) scripts/run_online_demo.py
	@echo ""
	@echo "Outputs: examples/output/online/run_01..05/"

figures:
	@echo "=============================================="
	@echo "CDR Figures — Evaluation Charts"
	@echo "=============================================="
	PYTHONPATH=src $(PYTHON) scripts/generate_figures.py
	@echo ""
	@echo "Output: eval/results/fig_latency.png"

# ============================================================================
# Maintenance
# ============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache/ htmlcov/ .coverage 2>/dev/null || true
	@echo "✅  Clean"
