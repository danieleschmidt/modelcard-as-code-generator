# Multi-stage production-ready Dockerfile for ModelCard Generator
FROM python:3.12-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r modelcard && useradd -r -g modelcard -u 1000 modelcard

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY README.md /app/

# Install the application in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /var/log/modelcard /var/cache/modelcard /var/output/modelcards && \
    chown -R modelcard:modelcard /var/log/modelcard /var/cache/modelcard /var/output/modelcards /app

# Copy health check script
COPY deployment/docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER modelcard

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MODELCARD_LOGGING_LEVEL=INFO \
    MODELCARD_CACHE_ENABLED=true

# Command to run the application
CMD ["python", "-m", "modelcard_generator.cli.main", "serve", "--host", "0.0.0.0", "--port", "8080"]