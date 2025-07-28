#!/bin/bash

# Prepare release script for Model Card Generator
# This script prepares the project for a new release

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Error: Version argument required"
    echo "Usage: $0 <version>"
    exit 1
fi

echo "🚀 Preparing release $VERSION..."

# Update version in pyproject.toml
echo "📝 Updating version in pyproject.toml..."
if command -v sed >/dev/null 2>&1; then
    sed -i.bak "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    rm pyproject.toml.bak 2>/dev/null || true
else
    echo "Warning: sed not available, manual version update required"
fi

# Update version in Python package
echo "🐍 Updating Python package version..."
mkdir -p src/modelcard_generator
cat > src/modelcard_generator/_version.py << EOF
"""Version information for Model Card Generator."""

__version__ = "$VERSION"
__version_info__ = tuple(int(x) for x in "$VERSION".split("."))
EOF

# Update package.json if it exists (for semantic-release)
if [ -f "package.json" ]; then
    echo "📦 Updating package.json version..."
    if command -v jq >/dev/null 2>&1; then
        jq ".version = \"$VERSION\"" package.json > package.json.tmp
        mv package.json.tmp package.json
    else
        echo "Warning: jq not available, package.json not updated"
    fi
fi

# Build the package
echo "🏗️ Building package..."
python -m build --clean

# Run security scan
echo "🔒 Running security scan..."
mkdir -p reports
bandit -r src/ -f json -o reports/security-report.json || echo "Security scan completed with warnings"
safety check --json --output reports/safety-report.json || echo "Safety check completed with warnings"

# Generate SBOM (Software Bill of Materials)
echo "📜 Generating SBOM..."
if command -v pip-licenses >/dev/null 2>&1; then
    pip-licenses --format=json --output-file=sbom.json
else
    echo "Warning: pip-licenses not available, SBOM not generated"
fi

# Run tests one more time
echo "✅ Running final test suite..."
pytest --tb=short -q

# Generate test coverage report
echo "📈 Generating coverage report..."
pytest --cov=src/modelcard_generator --cov-report=xml:reports/coverage-report.xml --tb=no -q

# Build documentation
echo "📚 Building documentation..."
if [ -f "mkdocs.yml" ]; then
    mkdocs build
    cd site && tar -czf ../docs/site.tar.gz . && cd ..
fi

# Validate package
echo "✓ Validating package..."
twine check dist/*

# Display release summary
echo ""
echo "\033[32m✓ Release $VERSION prepared successfully!\033[0m"
echo ""
echo "Release artifacts:"
echo "• Python package: $(ls dist/*.whl 2>/dev/null | head -1)"
echo "• Source distribution: $(ls dist/*.tar.gz 2>/dev/null | head -1)"
echo "• Security report: reports/security-report.json"
echo "• Coverage report: reports/coverage-report.xml"
echo "• SBOM: sbom.json"
if [ -f "docs/site.tar.gz" ]; then
    echo "• Documentation: docs/site.tar.gz"
fi
echo ""
echo "Next steps:"
echo "1. Review the generated artifacts"
echo "2. The semantic-release process will handle the rest"
echo ""
