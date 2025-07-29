#!/bin/bash
set -e

echo "🚀 Setting up Model Card Generator development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -e ".[dev,test,docs,integrations,all]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p {logs,temp,output,reports}

# Set up git configuration
echo "🔧 Configuring git..."
git config core.autocrlf input
git config pull.rebase false

# Initialize database/cache if needed
echo "🗄️ Initializing development database..."
# Add any database initialization here

# Download sample data for testing
echo "📊 Setting up sample data..."
mkdir -p tests/fixtures/sample_models
# Add commands to download or generate sample data

# Verify installation
echo "✅ Verifying installation..."
mcg --version
pytest --version
mkdocs --version

# Run initial tests
echo "🧪 Running smoke tests..."
python -c "import modelcard_generator; print('✅ Package imports successfully')"

# Start background services
echo "🚀 Starting development services..."
# Start any background services like databases, caches, etc.

echo "🎉 Development environment setup complete!"
echo ""
echo "📚 Quick start commands:"
echo "  mcg --help                    # View CLI help"
echo "  pytest tests/                 # Run tests"
echo "  mkdocs serve                  # Start documentation server"
echo "  pre-commit run --all-files    # Run all pre-commit checks"
echo ""
echo "🔗 Useful ports:"
echo "  http://localhost:8000         # Documentation server"
echo "  http://localhost:9090         # Prometheus (if running)"
echo ""
echo "Happy coding! 🚀"