#!/bin/bash
set -e

echo "🚀 Setting up Model Card Generator development environment..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install additional development tools
sudo apt-get install -y \
    git-lfs \
    jq \
    tree \
    htop \
    curl \
    wget \
    unzip \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# Install Poetry for dependency management
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/vscode/.local/bin:$PATH"
echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.bashrc

# Install pre-commit for git hooks
pip install pre-commit

# Install development dependencies if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo "📦 Installing project dependencies with Poetry..."
    poetry install --with dev,test,docs
else
    echo "📦 Installing common development packages..."
    pip install --upgrade pip
    pip install \
        black \
        flake8 \
        isort \
        mypy \
        pytest \
        pytest-cov \
        pytest-mock \
        hypothesis \
        ruff \
        bandit \
        safety
fi

# Setup git hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Create useful aliases
cat >> ~/.bashrc << 'EOF'

# Model Card Generator aliases
alias mcg='python -m modelcard_generator'
alias test='pytest -v'
alias test-cov='pytest --cov=modelcard_generator --cov-report=html'
alias lint='flake8 . && black --check . && isort --check-only .'
alias format='black . && isort .'
alias type-check='mypy .'
alias security='bandit -r . && safety check'
alias docs='mkdocs serve'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gd='git diff'

# Docker aliases
alias dps='docker ps'
alias di='docker images'
alias dc='docker-compose'

# Useful functions
function new-branch() {
    git checkout -b "$1"
    git push -u origin "$1"
}

function clean-branches() {
    git branch --merged | grep -v "\*\|main\|master" | xargs -n 1 git branch -d
}

EOF

# Create development directories
mkdir -p \
    docs/{guides,api,examples} \
    tests/{unit,integration,fixtures} \
    scripts \
    .github/{workflows,ISSUE_TEMPLATE,PULL_REQUEST_TEMPLATE} \
    examples \
    benchmarks

# Install GitHub CLI extensions (if needed)
if command -v gh &> /dev/null; then
    echo "🔧 Installing useful GitHub CLI extensions..."
    gh extension install github/gh-copilot || true
    gh extension install dlvhdr/gh-dash || true
fi

# Setup VSCode settings for the project
mkdir -p .vscode
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: Test Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "CLI: Generate Model Card",
            "type": "python",
            "request": "launch",
            "module": "modelcard_generator.cli",
            "args": ["generate", "--help"],
            "console": "integratedTerminal"
        }
    ]
}
EOF

cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "pytest",
            "args": ["-v", "--cov=modelcard_generator"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "Lint Code",
            "type": "shell",
            "command": "flake8",
            "args": ["."],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "Format Code",
            "type": "shell",
            "command": "black",
            "args": ["."],
            "group": "build"
        },
        {
            "label": "Type Check",
            "type": "shell",
            "command": "mypy",
            "args": ["."],
            "group": "build"
        },
        {
            "label": "Build Docs",
            "type": "shell",
            "command": "mkdocs",
            "args": ["build"],
            "group": "build"
        },
        {
            "label": "Serve Docs",
            "type": "shell",
            "command": "mkdocs",
            "args": ["serve"],
            "group": "build",
            "isBackground": true
        }
    ]
}
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  test       - Run tests with coverage"
echo "  lint       - Check code quality"
echo "  format     - Format code with black and isort"
echo "  docs       - Serve documentation locally"
echo "  mcg --help - Show Model Card Generator CLI help"
echo ""
echo "📚 Documentation will be available at http://localhost:8000 when serving"
echo "🔧 Remember to reload your shell or run 'source ~/.bashrc' to use aliases"