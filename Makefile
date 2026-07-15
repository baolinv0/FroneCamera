PYTHON ?= python3.12

.PHONY: install test lint format-check frontend api worker docker-up docker-down

install:
	$(PYTHON) -m pip install -e ".[dev,pdf]"

test:
	pytest -q

lint:
	ruff check src tests scripts
	mypy src/portrait_eval

format-check:
	ruff format --check src tests scripts

frontend:
	cd web && npm install --no-package-lock && npm test -- --run && npm run build

api:
	portrait-eval-api --host 127.0.0.1 --port 7860

worker:
	portrait-eval-worker

docker-up:
	docker compose up --build

docker-down:
	docker compose down
