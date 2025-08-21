# Production Dockerfile for Model Card Generator
FROM python:3.11-slim

# Set production environment
ENV ENVIRONMENT=production
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .[all]

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY deployment/ ./deployment/

# Create non-root user for security
RUN groupadd -r mcg && useradd -r -g mcg -d /app -s /bin/bash mcg
RUN chown -R mcg:mcg /app
USER mcg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "modelcard_generator.api", "--host", "0.0.0.0", "--port", "8080"]
