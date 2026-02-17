# Makefile for Customer Churn Analysis Project
# Usage: make <target>

.PHONY: help install install-dev test test-cov lint format clean run-api docker-build docker-run setup-model setup-hooks

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make setup-hooks  - Install pre-commit hooks"
	@echo "  make test         - Run test suite"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make lint         - Run linting checks (flake8, mypy)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Remove cached files and build artifacts"
	@echo "  make run-api      - Start FastAPI server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make setup-model  - Train and save the model pipeline"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install pre-commit

setup-hooks:
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov=api --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ api/ tests/ --count --show-source --statistics
	mypy src/ api/ --ignore-missing-imports

format:
	black src/ api/ tests/
	isort src/ api/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -f app.log 2>/dev/null || true

# Running the application
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Model setup
setup-model:
	python3 -m src.train --model random_forest

# Docker
docker-build:
	docker build -f Dockerfile -t customer-churn-app .

docker-run:
	@echo "Run with: docker run -p 8000:8000 -e GEMINI_API_KEY='your_key' customer-churn-app"

# Notebooks
notebooks:
	jupyter notebook notebooks/
