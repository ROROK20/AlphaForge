# AlphaForge — developer convenience targets.
# Run `make` or `make help` to see available commands.

.DEFAULT_GOAL := help

PYTHON ?= python3

.PHONY: help install test lint api snapshot frontend frontend-build demo demo-local clean

help: ## Show this help message
	@echo "AlphaForge — make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install backend package with dev extras (editable)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test: ## Run the backend test suite
	$(PYTHON) -m pytest tests/

lint: ## Lint the backend package with ruff
	ruff check alphaforge/ --ignore E501,E402

api: ## Run the FastAPI engine API (http://localhost:8003)
	$(PYTHON) -m alphaforge.api.app

snapshot: ## Re-record the dashboard's offline engine snapshot (after engine changes)
	$(PYTHON) scripts/record_engine_snapshot.py

frontend: ## Install deps and run the web demo (http://localhost:3001)
	cd frontend && npm install && npm run dev

frontend-build: ## Lint and build the web frontend for production
	cd frontend && npm install && npm run lint && npm run build

demo: ## One-command full stack via Docker (frontend:3001 → backend:8003)
	docker compose up --build

demo-local: ## Full stack without Docker: backend (8003) + frontend dev (3001)
	@echo "Starting AlphaForge API on :8003 and frontend on :3001 …"
	@echo "Stop with Ctrl-C (kills both)."
	@( $(PYTHON) -m alphaforge.api.app & echo $$! > /tmp/alphaforge-api.pid ) ; \
	  trap 'kill `cat /tmp/alphaforge-api.pid` 2>/dev/null' EXIT ; \
	  cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8003 npm run dev

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .coverage htmlcov coverage.xml build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
