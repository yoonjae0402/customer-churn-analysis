# Makefile for Customer Churn Analysis Project
# Usage: make <target>

.PHONY: help install install-dev test lint format clean run-api run-cli docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Remove cached files and build artifacts"
	@echo "  make run-api      - Start FastAPI server"
	@echo "  make run-cli      - Show CLI usage example"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make setup-model  - Generate production model artifacts"

# Installation
install:
	pip install -r app/requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 app/ src/ tests/ --max-line-length=120
	mypy app/ src/ --ignore-missing-imports

format:
	black app/ src/ tests/ --line-length=120
	isort app/ src/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -f app/app.log 2>/dev/null || true

# Running the application
run-api:
	cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-cli:
	@echo "Example CLI command:"
	@echo "cd app && python3 main.py --cli \\"
	@echo "    --gender Female \\"
	@echo "    --senior_citizen 0 \\"
	@echo "    --partner Yes \\"
	@echo "    --dependents No \\"
	@echo "    --tenure 12 \\"
	@echo "    --phone_service Yes \\"
	@echo "    --multiple_lines No \\"
	@echo "    --internet_service 'Fiber optic' \\"
	@echo "    --online_security No \\"
	@echo "    --online_backup Yes \\"
	@echo "    --device_protection No \\"
	@echo "    --tech_support No \\"
	@echo "    --streaming_tv Yes \\"
	@echo "    --streaming_movies No \\"
	@echo "    --contract Month-to-month \\"
	@echo "    --paperless_billing Yes \\"
	@echo "    --payment_method 'Electronic check' \\"
	@echo "    --monthly_charges 89.90"

# Model setup
setup-model:
	cd app && python3 save_pipeline.py

# Docker
docker-build:
	docker build -f app/Dockerfile -t customer-churn-app .

docker-run:
	@echo "Run with: docker run -p 8000:8000 -e GEMINI_API_KEY='your_key' customer-churn-app"

# Notebooks
notebooks:
	jupyter notebook notebooks/
