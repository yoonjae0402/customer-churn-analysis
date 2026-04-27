# Makefile for Customer Churn Analysis Project
# Usage: make <target>

.PHONY: help install install-dev test train serve demo lint format clean docker-build docker-run setup-hooks

# Default target
help:
	@echo "Available commands:"
	@echo "  make train        - Train the random forest model pipeline"
	@echo "  make serve        - Start FastAPI server (src/api/main.py)"
	@echo "  make test         - Run test suite"
	@echo "  make demo         - Launch Streamlit ROI dashboard"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make setup-hooks  - Install pre-commit hooks"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Remove cached files and build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"

# ── Core workflow ────────────────────────────────────────────────────────────

train:
	python3 -m src.train --model random_forest

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

demo:
	streamlit run dashboard.py

# ── Installation ─────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install pre-commit

setup-hooks:
	pre-commit install

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	flake8 src/ tests/ --count --show-source --statistics
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -f app.log 2>/dev/null || true

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -f Dockerfile -t customer-churn-app .

docker-run:
	@echo "Run with: docker run -p 8000:8000 -e GEMINI_API_KEY='your_key' customer-churn-app"

# ── Notebooks ─────────────────────────────────────────────────────────────────

notebooks:
	jupyter notebook notebooks/
