.PHONY: help install dev-install test lint format clean run-dev run-prod

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

dev-install:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-asyncio black flake8 mypy pre-commit

test:  ## Run tests
	python -m pytest tests/ -v

lint:  ## Run linting
	python -m flake8 src/
	python -m mypy src/

format:  ## Format code
	python -m black src/

clean:  ## Clean temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

run-dev:  ## Run development server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:  ## Run production server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

load-data:  ## Load sample data (requires tw.csv and moloco.csv)
	python -m src.main --tw tw.csv --moloco moloco.csv

setup-env:  ## Setup environment
	python -m venv .venv
	source .venv/Scripts/activate && pip install -r requirements.txt

db-init:  ## Initialize database (requires PostgreSQL)
	python -c "from src.db.setup import ensure_schema; ensure_schema()"