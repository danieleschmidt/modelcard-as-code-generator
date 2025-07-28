#!/bin/bash

# Post-create script for Model Card Generator development container
# This script runs after the container is created

set -e

echo "🚀 Setting up Model Card Generator development environment..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/output
mkdir -p /workspace/cache
mkdir -p /workspace/logs
mkdir -p /workspace/profiling
mkdir -p /workspace/notebooks
mkdir -p /workspace/data
mkdir -p /workspace/plugins
mkdir -p /workspace/schemas
mkdir -p /workspace/backups

# Set proper permissions
echo "🔒 Setting permissions..."
chown -R mcg:mcg /workspace/output
chown -R mcg:mcg /workspace/cache
chown -R mcg:mcg /workspace/logs
chown -R mcg:mcg /workspace/profiling
chown -R mcg:mcg /workspace/notebooks
chown -R mcg:mcg /workspace/data

# Install the package in development mode
echo "📦 Installing package in development mode..."
cd /workspace
pip install -e ".[dev,test,docs,integrations,all]"

# Install additional development tools
echo "🔧 Installing additional development tools..."
pip install \
    pre-commit \
    commitizen \
    git-cliff \
    invoke \
    jupyterlab \
    ipykernel \
    nbstripout \
    pip-tools \
    pipdeptree

# Setup git configuration for the container
echo "🔧 Setting up git configuration..."
git config --global user.name "MCG Developer"
git config --global user.email "dev@modelcard-generator.local"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input
git config --global core.editor "code --wait"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
if [ -f "/workspace/.pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️ No .pre-commit-config.yaml found, skipping pre-commit setup"
fi

# Setup Jupyter kernel
echo "🔬 Setting up Jupyter kernel..."
python -m ipykernel install --user --name mcg-dev --display-name "Model Card Generator (Dev)"

# Setup nbstripout for Jupyter notebooks
echo "🧹 Setting up nbstripout for clean notebook commits..."
nbstripout --install

# Create a .env file from .env.example if it doesn't exist
echo "⚙️ Setting up environment configuration..."
if [ ! -f "/workspace/.env" ] && [ -f "/workspace/.env.example" ]; then
    cp /workspace/.env.example /workspace/.env
    echo "✅ Created .env from .env.example"
fi

# Generate development documentation
echo "📚 Setting up documentation..."
if [ -f "/workspace/mkdocs.yml" ]; then
    mkdocs build --quiet
    echo "✅ Documentation built"
fi

# Create development aliases
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Model Card Generator development aliases
alias mcg-dev='python -m modelcard_generator.cli'
alias mcg-test='pytest -v'
alias mcg-test-cov='pytest --cov=modelcard_generator --cov-report=html'
alias mcg-lint='ruff check src/ tests/'
alias mcg-format='black src/ tests/ && isort src/ tests/'
alias mcg-type='mypy src/'
alias mcg-docs='mkdocs serve'
alias mcg-notebook='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias mcg-clean='find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true'

# Git aliases for better workflow
alias gst='git status'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gpl='git pull'
alias gps='git push'
alias gad='git add'
alias gcm='git commit -m'
alias gca='git commit --amend'
alias glo='git log --oneline'
alias gdf='git diff'
alias gdc='git diff --cached'

# Python aliases
alias py='python'
alias pip-upgrade='pip install --upgrade pip setuptools wheel'
alias pip-list='pip list --format=columns'
alias pip-outdated='pip list --outdated'

EOF

# Source the new aliases
source ~/.bashrc

# Create development scripts
echo "📜 Creating development scripts..."
mkdir -p /workspace/scripts/dev

# Create quick test script
cat > /workspace/scripts/dev/quick-test.sh << 'EOF'
#!/bin/bash
# Quick test script for development
set -e

echo "🧪 Running quick tests..."
pytest tests/unit/ -v --tb=short

echo "✅ Quick tests completed!"
EOF

# Create full test script
cat > /workspace/scripts/dev/full-test.sh << 'EOF'
#!/bin/bash
# Full test suite for development
set -e

echo "🧪 Running full test suite..."

echo "📝 Code formatting check..."
black --check src/ tests/
isort --check-only src/ tests/

echo "🔍 Linting..."
ruff check src/ tests/

echo "🏷️ Type checking..."
mypy src/

echo "🔒 Security check..."
bandit -r src/
safety check

echo "🧪 Unit tests..."
pytest tests/unit/ -v --cov=modelcard_generator

echo "🔗 Integration tests..."
pytest tests/integration/ -v

echo "✅ All tests passed!"
EOF

# Create benchmark script
cat > /workspace/scripts/dev/benchmark.sh << 'EOF'
#!/bin/bash
# Benchmark script for performance testing
set -e

echo "📊 Running performance benchmarks..."
pytest tests/performance/ -v --benchmark-only

echo "✅ Benchmarks completed!"
EOF

# Make scripts executable
chmod +x /workspace/scripts/dev/*.sh

# Create welcome message
echo "📋 Creating welcome message..."
cat > /workspace/.devcontainer/welcome.txt << 'EOF'
🎉 Welcome to the Model Card Generator Development Environment!

🚀 Quick Start:
   mcg-dev --help              # See available commands
   mcg-test                    # Run unit tests
   mcg-docs                    # Start documentation server
   mcg-notebook                # Start Jupyter Lab

🔧 Development Tools:
   mcg-lint                    # Run linting
   mcg-format                  # Format code
   mcg-type                    # Type checking
   scripts/dev/quick-test.sh   # Quick test suite
   scripts/dev/full-test.sh    # Full test suite

📚 Documentation:
   - API docs: http://localhost:8000/docs
   - Development guide: docs/DEVELOPMENT.md
   - Deployment guide: docs/DEPLOYMENT.md

🐛 Debugging:
   - VS Code is configured for Python debugging
   - Jupyter Lab available at http://localhost:8888
   - Use token: mcg-dev-token

🌐 Services:
   - MLflow UI: http://localhost:5000
   - Redis: localhost:6379
   - Prometheus: http://localhost:9091
   - Grafana: http://localhost:3000 (admin/admin)

Happy coding! 🎯
EOF

# Display welcome message
echo "✅ Development environment setup completed!"
echo ""
cat /workspace/.devcontainer/welcome.txt
echo ""
echo "💡 Tip: Run 'cat .devcontainer/welcome.txt' to see this message again."

# Final permissions check
chown -R mcg:mcg /home/mcg/.cache
chown -R mcg:mcg /workspace/.pytest_cache 2>/dev/null || true
chown -R mcg:mcg /workspace/.mypy_cache 2>/dev/null || true

echo "🎉 Setup complete! Ready for development."
