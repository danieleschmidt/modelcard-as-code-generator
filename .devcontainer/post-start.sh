#!/bin/bash

# Post-start script for Model Card Generator development container
# This script runs every time the container starts

set -e

echo "🚀 Starting Model Card Generator development services..."

# Wait for dependent services to be ready
echo "⏳ Waiting for services to be ready..."

# Wait for Redis
echo "Waiting for Redis..."
while ! redis-cli -h redis ping 2>/dev/null; do
    sleep 1
done
echo "✅ Redis is ready"

# Wait for MLflow (with timeout)
echo "Waiting for MLflow..."
counter=0
while ! curl -s http://mlflow:5000/health > /dev/null 2>&1; do
    sleep 2
    counter=$((counter + 1))
    if [ $counter -gt 30 ]; then
        echo "⚠️ MLflow startup timeout, continuing without it"
        break
    fi
done
if [ $counter -le 30 ]; then
    echo "✅ MLflow is ready"
fi

# Check if .env file exists and source it
if [ -f "/workspace/.env" ]; then
    echo "⚙️ Loading environment variables from .env"
    export $(grep -v '^#' /workspace/.env | xargs -d '\n')
fi

# Update pip and install any new dependencies
echo "📦 Checking for dependency updates..."
pip install --upgrade pip > /dev/null 2>&1

# Check if pyproject.toml has changed and reinstall if needed
if [ -f "/workspace/pyproject.toml" ]; then
    pip install -e ".[dev,test,docs,integrations,all]" --quiet > /dev/null 2>&1
fi

# Run any pre-commit autoupdate if hooks are installed
if [ -f "/workspace/.pre-commit-config.yaml" ] && command -v pre-commit >/dev/null 2>&1; then
    echo "🪝 Updating pre-commit hooks..."
    pre-commit autoupdate --quiet || true
fi

# Clear any stale cache files
echo "🧹 Cleaning cache files..."
find /workspace -name "*.pyc" -delete 2>/dev/null || true
find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Start development services in background if not running
echo "🔄 Starting development services..."

# Start a simple file watcher for auto-reloading (optional)
if [ "$MCG_HOT_RELOAD" = "true" ]; then
    echo "🔥 Hot reload enabled"
    # You can add file watching logic here if needed
fi

# Create runtime directories if they don't exist
mkdir -p /workspace/output 2>/dev/null || true
mkdir -p /workspace/cache 2>/dev/null || true
mkdir -p /workspace/logs 2>/dev/null || true

# Set permissions
chown -R mcg:mcg /workspace/output 2>/dev/null || true
chown -R mcg:mcg /workspace/cache 2>/dev/null || true
chown -R mcg:mcg /workspace/logs 2>/dev/null || true

# Check if we're in a git repository and setup git hooks
if [ -d "/workspace/.git" ]; then
    echo "🔗 Setting up git hooks..."
    if [ -f "/workspace/.pre-commit-config.yaml" ] && command -v pre-commit >/dev/null 2>&1; then
        pre-commit install --install-hooks --quiet || true
        pre-commit install --hook-type commit-msg --quiet || true
    fi
fi

# Log startup information
cat > /workspace/logs/dev-startup.log << EOF
Development Environment Started: $(date)
Python Version: $(python --version)
Pip Version: $(pip --version)
Git Version: $(git --version)
User: $(whoami)
Workspace: /workspace
Environment: ${MCG_ENVIRONMENT:-development}
EOF

# Show status
echo "✅ Development environment is ready!"
echo "📋 Check logs/dev-startup.log for startup details"
echo "💡 Run 'cat .devcontainer/welcome.txt' for helpful commands"

# Optional: Start background processes
if [ "$MCG_ENABLE_METRICS" = "true" ]; then
    echo "📈 Metrics collection enabled"
fi

if [ "$MCG_ENABLE_PROFILING" = "true" ]; then
    echo "🔍 Profiling enabled"
fi

echo "🎉 Ready for development!"
