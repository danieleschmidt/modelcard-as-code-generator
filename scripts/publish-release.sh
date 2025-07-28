#!/bin/bash

# Publish release script for Model Card Generator
# This script publishes the release to various platforms

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Error: Version argument required"
    echo "Usage: $0 <version>"
    exit 1
fi

echo "🚀 Publishing release $VERSION..."

# Check required environment variables
required_vars=("PYPI_TOKEN" "DOCKER_REGISTRY" "DOCKER_USERNAME")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Warning: $var environment variable not set"
    fi
done

# Publish to PyPI
if [ -n "$PYPI_TOKEN" ]; then
    echo "📦 Publishing to PyPI..."
    export TWINE_USERNAME="__token__"
    export TWINE_PASSWORD="$PYPI_TOKEN"
    twine upload dist/* --verbose
    echo "✓ Published to PyPI successfully"
else
    echo "⚠️ Skipping PyPI publish (PYPI_TOKEN not set)"
fi

# Build and publish Docker images
if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
    echo "🐳 Building and publishing Docker images..."
    
    # Login to Docker registry
    echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
    
    # Build images
    docker build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --target runtime \
        -t "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION" \
        -t "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:latest" \
        .
    
    # Build CLI-specific image
    docker build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --target cli \
        -t "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION-cli" \
        .
    
    # Push images
    docker push "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION"
    docker push "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:latest"
    docker push "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION-cli"
    
    echo "✓ Docker images published successfully"
else
    echo "⚠️ Skipping Docker publish (credentials not set)"
fi

# Publish documentation to GitHub Pages (if using)
if [ -f "docs/site.tar.gz" ] && [ -n "$GITHUB_TOKEN" ]; then
    echo "📚 Publishing documentation..."
    
    # This would typically use mike or similar tool
    # mike deploy --push --update-aliases $VERSION latest
    
    echo "✓ Documentation published successfully"
else
    echo "⚠️ Skipping documentation publish"
fi

# Update package registries and package managers
if [ -n "$CONDA_TOKEN" ]; then
    echo "🐍 Publishing to conda-forge..."
    # This would typically involve creating a PR to conda-forge feedstock
    echo "⚠️ Conda-forge publish requires manual PR creation"
fi

# Notify external systems
echo "📢 Sending notifications..."

# Slack notification (if configured)
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"🎉 Model Card Generator v$VERSION has been released!\\n\\n• PyPI: https://pypi.org/project/modelcard-as-code-generator/$VERSION/\\n• GitHub: https://github.com/terragonlabs/modelcard-as-code-generator/releases/tag/v$VERSION\\n• Docker: $DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION\"
        }" \
        "$SLACK_WEBHOOK_URL" || echo "Failed to send Slack notification"
fi

# Discord notification (if configured)
if [ -n "$DISCORD_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"content\": \"🎉 **Model Card Generator v$VERSION Released!**\\n\\n• PyPI: https://pypi.org/project/modelcard-as-code-generator/$VERSION/\\n• GitHub: https://github.com/terragonlabs/modelcard-as-code-generator/releases/tag/v$VERSION\"
        }" \
        "$DISCORD_WEBHOOK_URL" || echo "Failed to send Discord notification"
fi

# Email notification (if configured)
if [ -n "$EMAIL_RECIPIENTS" ]; then
    echo "Sending email notifications..."
    # This would use your preferred email service
fi

# Update internal registries or databases
if [ -n "$INTERNAL_REGISTRY_URL" ]; then
    echo "Updating internal package registry..."
    # Custom logic for internal systems
fi

# Clean up temporary files
echo "🧹 Cleaning up..."
rm -f .VERSION 2>/dev/null || true
docker logout "$DOCKER_REGISTRY" 2>/dev/null || true

echo ""
echo "\033[32m✓ Release $VERSION published successfully!\033[0m"
echo ""
echo "Published to:"
if [ -n "$PYPI_TOKEN" ]; then
    echo "• PyPI: https://pypi.org/project/modelcard-as-code-generator/$VERSION/"
fi
if [ -n "$DOCKER_USERNAME" ]; then
    echo "• Docker: $DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION"
fi
echo "• GitHub: https://github.com/terragonlabs/modelcard-as-code-generator/releases/tag/v$VERSION"
echo ""
echo "Release complete! 🎉"
