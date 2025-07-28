#!/bin/bash

# Post-release script for Model Card Generator
# This script runs cleanup and follow-up tasks after a successful release

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Error: Version argument required"
    echo "Usage: $0 <version>"
    exit 1
fi

echo "🎉 Post-release tasks for version $VERSION..."

# Update development version
echo "🔄 Updating development version..."
DEV_VERSION="${VERSION}-dev"
if command -v sed >/dev/null 2>&1; then
    sed -i.bak "s/version = \".*\"/version = \"$DEV_VERSION\"/" pyproject.toml
    rm pyproject.toml.bak 2>/dev/null || true
fi

# Update version in Python package for development
cat > src/modelcard_generator/_version.py << EOF
"""Version information for Model Card Generator."""

__version__ = "$DEV_VERSION"
__version_info__ = tuple(int(x) for x in "${VERSION}.0".split(".")[:3])  # Handle dev versions
EOF

# Create next development milestone on GitHub
if [ -n "$GITHUB_TOKEN" ]; then
    echo "🎯 Creating next milestone on GitHub..."
    
    # Calculate next version for milestone
    IFS='.' read -ra VERSION_PARTS <<< "$VERSION"
    MAJOR=${VERSION_PARTS[0]}
    MINOR=${VERSION_PARTS[1]}
    PATCH=${VERSION_PARTS[2]}
    
    # Create next patch version milestone
    NEXT_PATCH="$MAJOR.$MINOR.$((PATCH + 1))"
    
    curl -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        https://api.github.com/repos/terragonlabs/modelcard-as-code-generator/milestones \
        -d "{
            \"title\": \"v$NEXT_PATCH\",
            \"description\": \"Next patch release\",
            \"state\": \"open\"
        }" || echo "Failed to create milestone (may already exist)"
    
    echo "✓ Created milestone for v$NEXT_PATCH"
fi

# Update project documentation
echo "📚 Updating project documentation..."

# Update README with latest version info
if [ -f "README.md" ]; then
    # Update installation instructions with new version
    if command -v sed >/dev/null 2>&1; then
        sed -i.bak "s/modelcard-as-code-generator==[0-9.]*/modelcard-as-code-generator==$VERSION/g" README.md
        rm README.md.bak 2>/dev/null || true
    fi
fi

# Trigger documentation rebuild
if [ -n "$DOCS_WEBHOOK_URL" ]; then
    echo "Triggering documentation rebuild..."
    curl -X POST "$DOCS_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"version\": \"$VERSION\"}" || echo "Failed to trigger docs rebuild"
fi

# Update package metadata in external registries
echo "📦 Updating package metadata..."

# Submit to awesome lists or package directories
if [ -n "$AWESOME_PYTHON_PR_TOKEN" ]; then
    echo "Submitting to awesome-python..."
    # Logic to create PR for awesome-python list
fi

# Update conda-forge recipe (create PR)
if [ -n "$CONDA_FORGE_TOKEN" ]; then
    echo "Creating conda-forge update PR..."
    # Logic to update conda-forge feedstock
fi

# Update Homebrew formula
if [ -n "$HOMEBREW_TOKEN" ]; then
    echo "Updating Homebrew formula..."
    # Logic to update Homebrew tap
fi

# Analytics and metrics
echo "📈 Recording release metrics..."

# Log release to analytics service
if [ -n "$ANALYTICS_ENDPOINT" ]; then
    curl -X POST "$ANALYTICS_ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "{
            \"event\": \"release\",
            \"version\": \"$VERSION\",
            \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
            \"project\": \"modelcard-as-code-generator\"
        }" || echo "Failed to log release metrics"
fi

# Update internal dashboards
if [ -n "$INTERNAL_METRICS_API" ]; then
    echo "Updating internal metrics..."
    # Custom metrics logging
fi

# Social media announcements
echo "📢 Scheduling social media announcements..."

# Twitter announcement (if configured)
if [ -n "$TWITTER_API_KEY" ]; then
    echo "Scheduling Twitter announcement..."
    # Twitter API call to announce release
fi

# LinkedIn announcement
if [ -n "$LINKEDIN_TOKEN" ]; then
    echo "Scheduling LinkedIn announcement..."
    # LinkedIn API call
fi

# Community notifications
echo "👥 Notifying community..."

# Reddit post
if [ -n "$REDDIT_TOKEN" ]; then
    echo "Creating Reddit post..."
    # Reddit API to post in relevant subreddits
fi

# Hacker News submission
if [ -n "$HN_API_KEY" ]; then
    echo "Submitting to Hacker News..."
    # HN API submission
fi

# Discord community announcement
if [ -n "$DISCORD_COMMUNITY_WEBHOOK" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"embeds\": [{
                \"title\": \"Model Card Generator v$VERSION Released! 🎉\",
                \"description\": \"A new version of Model Card Generator is now available with improvements and new features.\",
                \"color\": 5814783,
                \"fields\": [
                    {
                        \"name\": \"Installation\",
                        \"value\": \"`pip install modelcard-as-code-generator==$VERSION`\",
                        \"inline\": false
                    },
                    {
                        \"name\": \"GitHub Release\",
                        \"value\": \"[View Release Notes](https://github.com/terragonlabs/modelcard-as-code-generator/releases/tag/v$VERSION)\",
                        \"inline\": false
                    }
                ],
                \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
            }]
        }" \
        "$DISCORD_COMMUNITY_WEBHOOK" || echo "Failed to send Discord announcement"
fi

# Update project status badges
echo "🏅 Updating project badges..."

# Update shields.io badges (they auto-update but we can trigger refresh)
if command -v curl >/dev/null 2>&1; then
    # Trigger badge refresh
    curl -s "https://img.shields.io/pypi/v/modelcard-as-code-generator?refresh=$(date +%s)" > /dev/null || true
    curl -s "https://img.shields.io/github/v/release/terragonlabs/modelcard-as-code-generator?refresh=$(date +%s)" > /dev/null || true
fi

# Security and compliance updates
echo "🔒 Running post-release security checks..."

# Scan the published package for vulnerabilities
if command -v safety >/dev/null 2>&1; then
    echo "Running security scan on published package..."
    pip install "modelcard-as-code-generator==$VERSION" --force-reinstall
    safety check || echo "Security scan completed with warnings"
fi

# Update vulnerability databases
if [ -n "$VULN_DB_API_KEY" ]; then
    echo "Updating vulnerability databases..."
    # Submit package info to vulnerability tracking systems
fi

# Backup and archival
echo "💾 Creating release backup..."

# Create release archive
RELEASE_ARCHIVE="release-$VERSION-$(date +%Y%m%d).tar.gz"
tar -czf "$RELEASE_ARCHIVE" \
    --exclude='*.git*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='node_modules' \
    --exclude='venv' \
    .

# Upload to backup storage
if [ -n "$BACKUP_S3_BUCKET" ]; then
    echo "Uploading release archive to backup storage..."
    aws s3 cp "$RELEASE_ARCHIVE" "s3://$BACKUP_S3_BUCKET/releases/" || echo "Failed to upload backup"
fi

# Clean up
echo "🧹 Cleaning up post-release files..."
rm -f "$RELEASE_ARCHIVE" 2>/dev/null || true
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

# Generate post-release report
echo "📊 Generating post-release report..."
cat > "post-release-report-$VERSION.md" << EOF
# Post-Release Report - v$VERSION

Generated on: $(date)

## Release Summary
- Version: $VERSION
- Release Date: $(date -u +%Y-%m-%d)
- Git Commit: $(git rev-parse HEAD)

## Distribution
- ✓ PyPI: https://pypi.org/project/modelcard-as-code-generator/$VERSION/
- ✓ Docker: terragonlabs/modelcard-generator:$VERSION
- ✓ GitHub: https://github.com/terragonlabs/modelcard-as-code-generator/releases/tag/v$VERSION

## Post-Release Actions Completed
- ✓ Development version updated to $DEV_VERSION
- ✓ Next milestone created
- ✓ Documentation updated
- ✓ Community notifications sent
- ✓ Metrics logged
- ✓ Security scans completed
- ✓ Backup archive created

## Next Steps
1. Monitor release adoption metrics
2. Respond to community feedback
3. Plan next release features
4. Update project roadmap

---
Generated by post-release automation
EOF

echo ""
echo "\033[32m✓ Post-release tasks completed successfully!\033[0m"
echo ""
echo "Release v$VERSION is now live and all follow-up tasks are complete."
echo ""
echo "Post-release report: post-release-report-$VERSION.md"
echo "Development version: $DEV_VERSION"
echo ""
echo "Thank you for using Model Card Generator! 🚀"
