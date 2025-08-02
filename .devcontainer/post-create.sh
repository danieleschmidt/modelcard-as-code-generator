#!/bin/bash

# Post-create script for Model Card Generator development environment
set -e

echo "🚀 Setting up Model Card Generator development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    curl \
    wget \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    jq \
    tree \
    htop

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
echo "📚 Installing Model Card Generator in development mode..."
pip install -e ".[dev,test,docs,integrations]"

# Install pre-commit hooks
echo "🎣 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {logs,output,cache,examples/output}

# Set up environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📄 Creating .env file from template..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your configuration"
fi

# Set up git configuration
echo "🔧 Configuring git..."
git config --global --add safe.directory ${PWD}

echo ""
echo "🎉 Model Card Generator development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run quality checks with 'make lint'"
echo "3. Run tests with 'make test'"
echo "4. Start developing! 🚀"
echo ""