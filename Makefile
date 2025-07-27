# Makefile for Model Card Generator
# Provides standardized commands for development, testing, and deployment

.PHONY: help install install-dev clean test test-unit test-integration test-coverage \
        lint format type-check security docs docs-serve build docker-build \
        docker-test docker-dev docker-clean release pre-commit setup-dev

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Configuration
# ============================================================================

PYTHON := python3
PIP := pip3
PROJECT_NAME := modelcard-as-code-generator
VERSION := $(shell python setup.py --version 2>/dev/null || echo "1.0.0")
DOCKER_IMAGE := terragonlabs/modelcard-generator
DOCKER_TAG := $(VERSION)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# Help target
# ============================================================================

help: ## Show this help message
	@echo "$(BLUE)Model Card Generator - Development Commands$(NC)"
	@echo "$(BLUE)===========================================$(NC)"
	@echo
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make install-dev     # Install development dependencies"
	@echo "  make test            # Run all tests"
	@echo "  make docker-build    # Build Docker image"
	@echo "  make release         # Create release"

# ============================================================================
# Installation targets
# ============================================================================

install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,test,docs,integrations]"
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

setup-dev: install-dev ## Complete development setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	mkdir -p output examples tests/fixtures scripts
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from template"; fi
	@echo "$(GREEN)Development setup complete!$(NC)"

# ============================================================================
# Cleaning targets
# ============================================================================

clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "$(GREEN)Clean complete!$(NC)"

clean-docker: ## Clean Docker images and volumes
	@echo "$(GREEN)Cleaning Docker resources...$(NC)"
	docker-compose down --volumes --remove-orphans
	docker system prune -f
	docker volume prune -f

# ============================================================================
# Testing targets
# ============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/unit/ -v -m "not slow"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest --cov=modelcard_generator --cov-report=html --cov-report=term --cov-report=xml
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-fast: ## Run fast tests only (exclude slow tests)
	@echo "$(GREEN)Running fast tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "not slow and not network"

test-all: ## Run all tests including slow and network tests
	@echo "$(GREEN)Running all tests (including slow and network tests)...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short

# ============================================================================
# Code quality targets
# ============================================================================

lint: ## Run all linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	flake8 src/ tests/
	ruff check src/ tests/
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)Code formatting complete!$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(GREEN)Checking code formatting...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checking...$(NC)"
	mypy src/ tests/

security: ## Run security scans
	@echo "$(GREEN)Running security scans...$(NC)"
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "$(GREEN)Security scans complete!$(NC)"

quality-check: lint type-check security ## Run all quality checks
	@echo "$(GREEN)All quality checks complete!$(NC)"

# ============================================================================
# Pre-commit targets
# ============================================================================

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

pre-commit-install: ## Install pre-commit hooks
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	pre-commit install

# ============================================================================
# Documentation targets
# ============================================================================

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	mkdocs build
	@echo "$(GREEN)Documentation built in site/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(GREEN)Deploying documentation...$(NC)"
	mkdocs gh-deploy

# ============================================================================
# Build targets
# ============================================================================

build: clean ## Build Python package
	@echo "$(GREEN)Building Python package...$(NC)"
	$(PYTHON) -m build --wheel --no-isolation
	@echo "$(GREEN)Package built in dist/$(NC)"

build-check: build ## Build and check package
	@echo "$(GREEN)Checking built package...$(NC)"
	twine check dist/*

# ============================================================================
# Docker targets
# ============================================================================

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build \
		--build-arg BUILD_DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ) \
		--build-arg VERSION=$(VERSION) \
		--build-arg VCS_REF=$(shell git rev-parse --short HEAD) \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) \
		-t $(DOCKER_IMAGE):latest \
		.
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(NC)"
	docker build \
		--target development \
		--build-arg BUILD_DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ) \
		--build-arg VERSION=$(VERSION)-dev \
		--build-arg VCS_REF=$(shell git rev-parse --short HEAD) \
		-t $(DOCKER_IMAGE):dev \
		.

docker-test: ## Run tests in Docker container
	@echo "$(GREEN)Running tests in Docker...$(NC)"
	docker-compose --profile test up --build --abort-on-container-exit test

docker-dev: ## Start development environment with Docker Compose
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker-compose --profile dev up --build -d

docker-full: ## Start full environment with all services
	@echo "$(GREEN)Starting full environment...$(NC)"
	docker-compose --profile full up --build -d

docker-logs: ## Show Docker Compose logs
	docker-compose logs -f

docker-stop: ## Stop Docker Compose services
	docker-compose down

docker-clean: clean-docker docker-stop ## Clean Docker resources

# ============================================================================
# Release targets
# ============================================================================

bump-patch: ## Bump patch version
	@echo "$(GREEN)Bumping patch version...$(NC)"
	bump2version patch

bump-minor: ## Bump minor version
	@echo "$(GREEN)Bumping minor version...$(NC)"
	bump2version minor

bump-major: ## Bump major version
	@echo "$(GREEN)Bumping major version...$(NC)"
	bump2version major

release-check: ## Check if ready for release
	@echo "$(GREEN)Checking release readiness...$(NC)"
	@echo "Current version: $(VERSION)"
	@echo "Git status:"
	@git status --porcelain
	@echo "Last commits:"
	@git log --oneline -5
	@echo "$(GREEN)Release check complete!$(NC)"

release: quality-check test-coverage build-check ## Create release
	@echo "$(GREEN)Creating release...$(NC)"
	@read -p "Version to release (current: $(VERSION)): " version; \
	if [ -n "$$version" ]; then \
		bump2version --new-version $$version patch; \
	fi
	git push origin main --tags
	@echo "$(GREEN)Release complete!$(NC)"

# ============================================================================
# Utility targets
# ============================================================================

show-version: ## Show current version
	@echo "Current version: $(VERSION)"

show-env: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'unknown')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

deps-update: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,test,docs,integrations]"

deps-list: ## List installed dependencies
	@echo "$(GREEN)Installed dependencies:$(NC)"
	$(PIP) list

check-deps: ## Check for dependency security issues
	@echo "$(GREEN)Checking dependencies for security issues...$(NC)"
	safety check

# ============================================================================
# CI/CD targets
# ============================================================================

ci-test: ## Run tests for CI (with specific formatting)
	@echo "$(GREEN)Running CI tests...$(NC)"
	$(PYTHON) -m pytest tests/ \
		--cov=modelcard_generator \
		--cov-report=xml \
		--cov-report=term \
		--junitxml=test-results.xml \
		-v

ci-quality: ## Run quality checks for CI
	@echo "$(GREEN)Running CI quality checks...$(NC)"
	flake8 src/ tests/ --format=json --output-file=flake8-report.json || true
	bandit -r src/ -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

ci-build: ## Build for CI
	@echo "$(GREEN)Building for CI...$(NC)"
	$(PYTHON) -m build --wheel --no-isolation
	twine check dist/*

ci-all: ci-quality ci-test ci-build ## Run all CI checks

# ============================================================================
# Example and demo targets
# ============================================================================

example-basic: ## Run basic example
	@echo "$(GREEN)Running basic example...$(NC)"
	$(PYTHON) -c "print('Basic example would run here')"

example-advanced: ## Run advanced example
	@echo "$(GREEN)Running advanced example...$(NC)"
	$(PYTHON) -c "print('Advanced example would run here')"

demo: ## Run demonstration
	@echo "$(GREEN)Running demonstration...$(NC)"
	@echo "This would demonstrate the Model Card Generator capabilities"

# ============================================================================
# Development workflow targets
# ============================================================================

dev-setup: setup-dev pre-commit-install ## Complete development setup
	@echo "$(GREEN)Development environment fully configured!$(NC)"

dev-test: format lint test-fast ## Quick development test cycle
	@echo "$(GREEN)Development test cycle complete!$(NC)"

dev-full: format lint type-check test-coverage ## Full development check
	@echo "$(GREEN)Full development check complete!$(NC)"

# ============================================================================
# Debugging targets
# ============================================================================

debug-env: show-env ## Show debugging environment info
	@echo "$(GREEN)Additional debug info:$(NC)"
	@echo "Working directory: $(PWD)"
	@echo "PATH: $(PATH)"
	@echo "Python path: $(shell which $(PYTHON))"
	@echo "Make version: $(shell make --version | head -1)"

debug-deps: ## Debug dependency issues
	@echo "$(GREEN)Debugging dependencies...$(NC)"
	$(PIP) check
	$(PIP) list --outdated

# ============================================================================
# Performance targets
# ============================================================================

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	@echo "Benchmarks would run here"

profile: ## Run profiling
	@echo "$(GREEN)Running profiling...$(NC)"
	@echo "Profiling would run here"