# Developer entry points. Python side uses the repo venv; Node via npm on PATH or nvm.
PY    := .venv/bin/python
NPM   ?= $(shell command -v npm 2>/dev/null || echo $(HOME)/.nvm/versions/node/v22.23.2/bin/npm)

.PHONY: test test-py test-fe api-types frontend dev start

test: test-py test-fe

test-py:
	$(PY) -m pytest -q

test-fe:
	@if [ -d frontend/node_modules ]; then cd frontend && $(NPM) test; else echo "frontend not installed (make frontend)"; fi

# Regenerate the OpenAPI snapshot + the TypeScript client types from the FastAPI app.
api-types:
	$(PY) -m vorch openapi > frontend/openapi.json
	@if [ -d frontend/node_modules ]; then cd frontend && $(NPM) run gen:api; else echo "frontend/openapi.json updated; run 'make frontend' then 'make api-types' for TS types"; fi

frontend:
	cd frontend && $(NPM) ci && $(NPM) run build

# Bring the world up (adopts anything already running), then print the one thing it cannot
# start for you. `vorch status` afterwards says who holds the GPU.
dev:
	$(PY) -m vorch up
	@echo
	@echo "UI (the one part vorch cannot start for itself):"
	@echo "    cd frontend && npm run dev     ->  http://localhost:3000"

start:
	cd frontend && $(NPM) run start
