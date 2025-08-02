# Docker Deployment Guide

## Overview

The Model Card Generator provides comprehensive Docker support with multiple build targets optimized for different use cases. This guide covers all deployment scenarios from development to production.

## Quick Start

### Basic Usage

```bash
# Pull the latest image
docker pull terragonlabs/modelcard-generator:latest

# Generate a model card
docker run --rm -v $(pwd):/data terragonlabs/modelcard-generator:latest \
  mcg generate /data/eval_results.json --output /data/MODEL_CARD.md
```

### Using Docker Compose

```bash
# Start the complete development environment
docker-compose --profile dev up -d

# Start with MLOps integrations
docker-compose --profile mlops up -d

# Start with monitoring
docker-compose --profile monitoring up -d
```

## Build Targets

### 1. Runtime (Production)

Minimal production image optimized for size and security.

```bash
# Build
./scripts/build.sh -t runtime

# Use
docker run --rm terragonlabs/modelcard-generator:latest mcg --version
```

**Features:**
- Python 3.11 slim base
- Non-root user execution
- Health checks included
- Minimal attack surface
- Size: ~200MB

### 2. Development

Full development environment with all tools and source code.

```bash
# Build
./scripts/build.sh -t development

# Use
docker run --rm -it -v $(pwd):/app/src \
  terragonlabs/modelcard-generator:development bash
```

**Features:**
- Development dependencies included
- Source code mounted
- Pre-commit hooks available
- Interactive debugging
- Size: ~500MB

### 3. CI/CD

Testing and analysis environment for continuous integration.

```bash
# Build
./scripts/build.sh -t cicd -T

# Use in CI
docker run --rm -v $(pwd):/app \
  terragonlabs/modelcard-generator:cicd \
  pytest --cov=src --cov-report=xml
```

**Features:**
- Testing frameworks
- Security scanning tools
- Coverage reporting
- Code quality tools
- Size: ~600MB

### 4. Documentation

Documentation serving with MkDocs.

```bash
# Build
./scripts/build.sh -t docs

# Serve docs
docker run --rm -p 8000:8000 \
  terragonlabs/modelcard-generator:docs
```

**Features:**
- MkDocs with Material theme
- Auto-reloading
- Plugin ecosystem
- Size: ~300MB

### 5. CLI

Ultra-minimal CLI-only image.

```bash
# Build
./scripts/build.sh -t cli

# Use as CLI tool
docker run --rm terragonlabs/modelcard-generator:cli --help
```

**Features:**
- CLI-focused
- Minimal dependencies
- Fastest startup
- Size: ~150MB

## Build Script Usage

### Basic Building

```bash
# Build default runtime image
./scripts/build.sh

# Build specific target
./scripts/build.sh -t development

# Build with version tag
./scripts/build.sh -v 1.2.3

# Build and push to registry
./scripts/build.sh -v 1.2.3 -p -r registry.company.com
```

### Advanced Options

```bash
# Multi-platform build
./scripts/build.sh --platform linux/amd64,linux/arm64

# Build without cache
./scripts/build.sh --no-cache

# Custom build arguments
./scripts/build.sh --build-arg CUSTOM_ARG=value

# Build and test
./scripts/build.sh -t cicd -T
```

## Docker Compose Profiles

### Development Profile

```bash
docker-compose --profile dev up -d
```

**Services:**
- `dev`: Development container with source mounted
- Volumes for persistent data
- Hot-reload capability

### MLOps Profile

```bash
docker-compose --profile mlops up -d
```

**Services:**
- `mlflow`: MLflow tracking server
- `app`: Main application
- Shared network for integration

### Monitoring Profile

```bash
docker-compose --profile monitoring up -d
```

**Services:**
- `prometheus`: Metrics collection
- `grafana`: Visualization dashboard
- Pre-configured dashboards

### Full Stack

```bash
docker-compose --profile full up -d
```

**Services:**
- All services enabled
- Complete development environment
- All integrations available

## Environment Configuration

### Environment Variables

```bash
# Core configuration
MCG_ENVIRONMENT=production
MCG_LOG_LEVEL=INFO
MCG_OUTPUT_DIR=/app/output
MCG_CACHE_DIR=/app/cache

# ML Platform integrations
MLFLOW_TRACKING_URI=http://mlflow:5000
WANDB_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here

# Security settings
MCG_ENABLE_SECRET_SCANNING=true
MCG_AUDIT_LOG_FILE=/app/logs/audit.log

# Performance tuning
MCG_WORKER_PROCESSES=4
MCG_MEMORY_LIMIT_MB=2048
```

### Volume Mounts

```bash
# Input/output data
-v $(pwd)/data:/app/data:ro
-v $(pwd)/output:/app/output

# Custom templates
-v $(pwd)/templates:/app/templates:ro

# Configuration
-v $(pwd)/.env:/app/.env:ro

# Persistent cache
-v mcg-cache:/app/cache
```

## Production Deployment

### Using Docker Swarm

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  mcg:
    image: terragonlabs/modelcard-generator:1.0.0
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
    environment:
      - MCG_ENVIRONMENT=production
      - MCG_LOG_LEVEL=INFO
    volumes:
      - production-data:/app/data:ro
      - production-output:/app/output
    networks:
      - mcg-production

networks:
  mcg-production:
    external: true

volumes:
  production-data:
    external: true
  production-output:
    external: true
```

Deploy:
```bash
docker stack deploy -c docker-compose.prod.yml mcg-stack
```

### Using Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelcard-generator
  namespace: mcg
spec:
  replicas: 3
  selector:
    matchLabels:
      app: modelcard-generator
  template:
    metadata:
      labels:
        app: modelcard-generator
    spec:
      containers:
      - name: mcg
        image: terragonlabs/modelcard-generator:1.0.0
        env:
        - name: MCG_ENVIRONMENT
          value: "production"
        - name: MCG_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: config
          mountPath: /app/.env
          subPath: .env
        - name: output
          mountPath: /app/output
        livenessProbe:
          exec:
            command:
            - mcg
            - --version
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - mcg
            - --version
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: mcg-config
      - name: output
        persistentVolumeClaim:
          claimName: mcg-output
```

## Security Considerations

### Image Security

1. **Non-root execution**: All containers run as non-root user `mcg`
2. **Minimal base images**: Using slim variants to reduce attack surface
3. **Security scanning**: Integrated Trivy scanning in build process
4. **Read-only filesystems**: Where possible, use read-only mounts

### Build Security

```bash
# Enable content trust
export DOCKER_CONTENT_TRUST=1

# Scan before push
./scripts/build.sh -t runtime
trivy image terragonlabs/modelcard-generator:latest

# Sign images (if using Docker Notary)
docker trust sign terragonlabs/modelcard-generator:latest
```

### Runtime Security

```bash
# Run with security options
docker run --rm \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /app/cache \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  terragonlabs/modelcard-generator:latest mcg --version
```

## Performance Optimization

### Multi-stage Builds

The Dockerfile uses multi-stage builds to:
- Separate build dependencies from runtime
- Minimize final image size
- Enable parallel building
- Improve caching efficiency

### Build Cache Optimization

```bash
# Use BuildKit for better caching
export DOCKER_BUILDKIT=1

# Cache mount for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

### Resource Limits

```yaml
# docker-compose.yml
services:
  mcg:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
```

## Monitoring and Logging

### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD mcg --version || exit 1
```

### Logging

```bash
# Configure logging driver
docker run --rm \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  terragonlabs/modelcard-generator:latest
```

### Metrics Collection

```bash
# Start with Prometheus monitoring
docker-compose --profile monitoring up -d

# Access Grafana dashboard
open http://localhost:3000
```

## Troubleshooting

### Common Issues

**Build Failures:**
```bash
# Clean build cache
docker builder prune -f

# Build without cache
./scripts/build.sh --no-cache

# Check disk space
docker system df
```

**Runtime Issues:**
```bash
# Check container logs
docker logs mcg-app

# Interactive debugging
docker run --rm -it \
  terragonlabs/modelcard-generator:development bash

# Check health
docker exec mcg-app mcg --version
```

**Permission Issues:**
```bash
# Fix volume permissions
docker run --rm -v $(pwd):/data alpine chown -R 1000:1000 /data

# Use correct user mapping
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd):/data terragonlabs/modelcard-generator:latest
```

### Debug Mode

```bash
# Enable debug logging
docker run --rm \
  -e MCG_LOG_LEVEL=DEBUG \
  -e MCG_DEBUG=true \
  terragonlabs/modelcard-generator:latest mcg generate data.json
```

## Registry Management

### Private Registry

```bash
# Login to private registry
docker login registry.company.com

# Build and push
./scripts/build.sh -v 1.0.0 -p -r registry.company.com

# Pull from private registry
docker pull registry.company.com/terragonlabs/modelcard-generator:1.0.0
```

### Harbor Integration

```bash
# Configure Harbor robot account
export HARBOR_USERNAME=robot$project+mcg
export HARBOR_PASSWORD=your_secret

# Login and push
echo $HARBOR_PASSWORD | docker login --username $HARBOR_USERNAME --password-stdin harbor.company.com
./scripts/build.sh -v 1.0.0 -p -r harbor.company.com/ml-tools
```

## Best Practices

### Image Tagging

```bash
# Semantic versioning
./scripts/build.sh -v 1.2.3

# Environment tags
./scripts/build.sh -v 1.2.3-staging
./scripts/build.sh -v 1.2.3-prod

# Feature branches
./scripts/build.sh -v feature-new-templates
```

### Layer Caching

1. Order Dockerfile instructions from least to most frequently changing
2. Use specific package versions
3. Leverage multi-stage builds
4. Use .dockerignore effectively

### Resource Management

```bash
# Regular cleanup
docker system prune -f

# Remove unused images
docker image prune -f

# Monitor resource usage
docker stats
```

## Integration Examples

### CI/CD Pipeline

```yaml
# .github/workflows/docker.yml
name: Docker Build and Push

on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and test
        run: |
          ./scripts/build.sh -t cicd -T
          
      - name: Build and push production
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          ./scripts/build.sh -t runtime -v ${GITHUB_REF#refs/tags/} -p
```

### Local Development

```bash
# Start development environment
docker-compose --profile dev up -d

# Attach to development container
docker exec -it mcg-dev bash

# Run tests
docker exec mcg-dev pytest

# Watch for changes
docker exec mcg-dev python tests/test_runners.py watch
```