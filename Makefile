# Makefile for GEM strategy project

.PHONY: help install install-dev test test-cov lint format type-check clean build docs run-example

# Default target
help:
	@echo "GEM Strategy - Make Commands"
	@echo "============================"
	@echo "install          Install package in production mode"
	@echo "install-dev      Install package in development mode with all dependencies"
	@echo "test             Run tests with pytest"
	@echo "test-cov         Run tests with coverage report"
	@echo "lint             Run linting (flake8)"
	@echo "format           Format code (black + isort)"
	@echo "type-check       Run type checking (mypy)"
	@echo "clean            Clean up build artifacts and cache"
	@echo "build            Build distribution packages"
	@echo "docs             Generate documentation"
	@echo "run-example      Run example backtest"
	@echo "setup-hooks      Install pre-commit hooks"
	@echo "security         Run security checks (bandit)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=gem --cov-report=term-missing --cov-report=html

test-watch:
	pytest-watch tests/ -- -v

# Code Quality
lint:
	flake8 gem/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	black gem/ tests/
	isort gem/ tests/ --profile=black

type-check:
	mypy gem/ --ignore-missing-imports

security:
	bandit -r gem/ -f json -o bandit-report.json || true
	@echo "Security report saved to bandit-report.json"

# Development Tools
setup-hooks:
	pre-commit install
	@echo "Pre-commit hooks installed"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Build
build: clean
	python -m build

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "README.md and notebook tutorial available"

# Examples
run-example:
	@echo "Running example backtest..."
	gem backtest --start-date 2020-01-01 --show-metrics --create-charts

demo:
	@echo "Running GEM strategy demo..."
	gem allocate
	@echo ""
	gem compare --asset SPY --asset VEU --asset AGG

# All quality checks
check-all: lint type-check test security
	@echo "All quality checks completed!"

# CI/CD pipeline simulation
ci: install-dev check-all
	@echo "CI pipeline completed successfully!"