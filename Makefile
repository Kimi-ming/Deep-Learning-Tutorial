.PHONY: help install install-dev format lint test clean

help:
	@echo "Deep Learning Tutorial - Makefile Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Run ruff linter"
	@echo "  make test         - Run pytest tests"
	@echo "  make clean        - Remove cache and build files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

format:
	@echo "Formatting code with black..."
	black .

lint:
	@echo "Running ruff linter..."
	ruff check .

test:
	@echo "Running tests with pytest..."
	pytest

clean:
	@echo "Cleaning cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "Clean complete!"
