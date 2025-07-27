# Multi-stage Docker build for Model Card Generator
# Optimized for security, size, and performance

# ============================================================================
# Build stage - Dependencies and compilation
# ============================================================================
FROM python:3.11-slim-bookworm as builder

# Build arguments
ARG BUILD_DATE
ARG VERSION="1.0.0"
ARG VCS_REF

# Labels for metadata
LABEL org.opencontainers.image.title="Model Card Generator" \
      org.opencontainers.image.description="Automated generation of Model Cards as executable, versioned artifacts" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/terragonlabs/modelcard-as-code-generator" \
      org.opencontainers.image.url="https://github.com/terragonlabs/modelcard-as-code-generator" \
      org.opencontainers.image.documentation="https://docs.terragonlabs.com/modelcard-generator" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for building
RUN groupadd --gid 1000 mcg && \
    useradd --uid 1000 --gid mcg --shell /bin/bash --create-home mcg

# Set working directory
WORKDIR /app

# Copy dependency files
COPY --chown=mcg:mcg pyproject.toml README.md ./
COPY --chown=mcg:mcg src/ ./src/

# Install dependencies and build wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir build && \
    python -m build --wheel --no-isolation

# ============================================================================
# Security scanning stage (optional)
# ============================================================================
FROM aquasec/trivy:latest as security-scan

# Copy built wheel for scanning
COPY --from=builder /app/dist/*.whl /tmp/

# Run security scan (this will be used in CI/CD)
RUN trivy fs --exit-code 0 --no-progress --format json --output /tmp/scan-results.json /tmp/ || true

# ============================================================================
# Runtime stage - Minimal production image
# ============================================================================
FROM python:3.11-slim-bookworm as runtime

# Build arguments
ARG BUILD_DATE
ARG VERSION="1.0.0"
ARG VCS_REF

# Labels for metadata
LABEL org.opencontainers.image.title="Model Card Generator" \
      org.opencontainers.image.description="Automated generation of Model Cards as executable, versioned artifacts" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/terragonlabs/modelcard-as-code-generator" \
      org.opencontainers.image.url="https://github.com/terragonlabs/modelcard-as-code-generator" \
      org.opencontainers.image.documentation="https://docs.terragonlabs.com/modelcard-generator" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 mcg && \
    useradd --uid 1000 --gid mcg --shell /bin/bash --create-home mcg

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the application
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir *.whl && \
    rm -f *.whl

# Create directories with proper permissions
RUN mkdir -p /app/output /app/cache /app/templates /app/logs && \
    chown -R mcg:mcg /app

# Copy templates and schemas (if they exist)
COPY --chown=mcg:mcg --from=builder /app/src/modelcard_generator/templates/ /app/templates/ 2>/dev/null || true
COPY --chown=mcg:mcg --from=builder /app/src/modelcard_generator/schemas/ /app/schemas/ 2>/dev/null || true

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MCG_ENVIRONMENT=production \
    MCG_LOG_LEVEL=INFO \
    MCG_OUTPUT_DIR=/app/output \
    MCG_CACHE_DIR=/app/cache \
    MCG_CUSTOM_TEMPLATES_DIR=/app/templates

# Switch to non-root user
USER mcg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD mcg --version || exit 1

# Expose port for potential HTTP server (future feature)
EXPOSE 8080

# Default command
CMD ["mcg", "--help"]

# ============================================================================
# Development stage - With dev dependencies
# ============================================================================
FROM runtime as development

# Switch back to root for installing dev dependencies
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

# Install git (needed for pre-commit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code for development
COPY --chown=mcg:mcg . /app/src/

# Switch back to non-root user
USER mcg

# Set development environment
ENV MCG_ENVIRONMENT=development \
    MCG_LOG_LEVEL=DEBUG \
    MCG_DEBUG=true

# Default command for development
CMD ["bash"]

# ============================================================================
# CI/CD stage - With testing and analysis tools
# ============================================================================
FROM development as cicd

# Switch to root for installing CI tools
USER root

# Install additional CI/CD tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    coverage \
    pytest-xdist \
    hypothesis

# Install additional system tools for CI
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy all source files and configs
COPY --chown=mcg:mcg . /app/

# Switch back to non-root user
USER mcg

# Set CI environment
ENV MCG_ENVIRONMENT=ci \
    MCG_LOG_LEVEL=INFO \
    CI=true

# Default command for CI
CMD ["python", "-m", "pytest", "--cov=modelcard_generator", "--cov-report=xml", "--cov-report=html"]

# ============================================================================
# Documentation stage - With docs building tools
# ============================================================================
FROM python:3.11-slim-bookworm as docs

# Install documentation dependencies
RUN pip install --no-cache-dir \
    mkdocs \
    mkdocs-material \
    mkdocs-mermaid2-plugin \
    mkdocstrings[python]

# Create non-root user
RUN groupadd --gid 1000 docs && \
    useradd --uid 1000 --gid docs --shell /bin/bash --create-home docs

WORKDIR /app

# Copy documentation files
COPY --chown=docs:docs docs/ ./docs/
COPY --chown=docs:docs mkdocs.yml ./
COPY --chown=docs:docs README.md ./

# Switch to non-root user
USER docs

# Expose docs port
EXPOSE 8000

# Default command for documentation
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]

# ============================================================================
# Utility stage - CLI-focused minimal image
# ============================================================================
FROM runtime as cli

# Remove unnecessary packages to minimize size
USER root
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Switch back to non-root user
USER mcg

# Override default command for CLI usage
ENTRYPOINT ["mcg"]
CMD ["--help"]