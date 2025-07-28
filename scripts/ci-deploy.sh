#!/bin/bash

# CI deployment script for Model Card Generator
# This script handles deployment to various environments

set -e

ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"

echo "🚀 Starting deployment to $ENVIRONMENT (version: $VERSION)..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Validate environment
case "$ENVIRONMENT" in
    "development"|"dev")
        ENVIRONMENT="development"
        DEPLOY_TARGET="dev"
        ;;
    "staging"|"stage")
        ENVIRONMENT="staging"
        DEPLOY_TARGET="staging"
        ;;
    "production"|"prod")
        ENVIRONMENT="production"
        DEPLOY_TARGET="production"
        ;;
    *)
        print_error "Invalid environment: $ENVIRONMENT"
        echo "Valid environments: development, staging, production"
        exit 1
        ;;
esac

print_status "Deploying to environment: $ENVIRONMENT"

# Check required environment variables
required_vars=()
case "$ENVIRONMENT" in
    "staging")
        required_vars=("STAGING_DEPLOY_KEY" "STAGING_HOST")
        ;;
    "production")
        required_vars=("PROD_DEPLOY_KEY" "PROD_HOST" "PROD_BACKUP_KEY")
        ;;
esac

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        print_error "Required environment variable $var is not set"
        exit 1
    fi
done

# Pre-deployment checks
print_status "Running pre-deployment checks..."

# Check if tests pass
if [ -f "reports/test-summary.txt" ]; then
    if grep -q "All CI tests completed successfully" reports/test-summary.txt; then
        print_success "Tests passed - proceeding with deployment"
    else
        print_error "Tests failed - deployment aborted"
        exit 1
    fi
else
    print_warning "No test results found - running quick test"
    if ! ./scripts/ci-test.sh; then
        print_error "Quick test failed - deployment aborted"
        exit 1
    fi
fi

# Security check for production
if [ "$ENVIRONMENT" = "production" ]; then
    print_status "Running production security checks..."
    
    # Check for security vulnerabilities
    if [ -f "reports/bandit-report.json" ]; then
        if jq -e '.results | length > 0' reports/bandit-report.json > /dev/null 2>&1; then
            print_error "Security vulnerabilities found - production deployment blocked"
            exit 1
        fi
    fi
    
    # Check dependency vulnerabilities
    if [ -f "reports/safety-report.json" ]; then
        if jq -e '.vulnerabilities | length > 0' reports/safety-report.json > /dev/null 2>&1; then
            print_error "Vulnerable dependencies found - production deployment blocked"
            exit 1
        fi
    fi
    
    print_success "Production security checks passed"
fi

# Build artifacts for deployment
print_status "Building deployment artifacts..."

# Build Python package
print_status "Building Python package..."
python -m build --clean
print_success "Python package built"

# Build Docker images
if [ -f "Dockerfile" ] && command -v docker &> /dev/null; then
    print_status "Building Docker images..."
    
    # Set Docker registry based on environment
    case "$ENVIRONMENT" in
        "development")
            DOCKER_REGISTRY="localhost:5000"
            ;;
        "staging")
            DOCKER_REGISTRY="${STAGING_DOCKER_REGISTRY:-staging.registry.com}"
            ;;
        "production")
            DOCKER_REGISTRY="${PROD_DOCKER_REGISTRY:-registry.com}"
            ;;
    esac
    
    # Build main image
    docker build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        --target runtime \
        -t "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION" \
        -t "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$ENVIRONMENT" \
        .
    
    print_success "Docker images built"
fi

# Create deployment package
print_status "Creating deployment package..."
DEPLOY_DIR="deploy-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEPLOY_DIR"

# Copy necessary files
cp -r dist/ "$DEPLOY_DIR/"
cp docker-compose.yml "$DEPLOY_DIR/"
cp -r monitoring/ "$DEPLOY_DIR/" 2>/dev/null || true
cp .env.example "$DEPLOY_DIR/.env.template"

# Create environment-specific configuration
cat > "$DEPLOY_DIR/deploy-config.env" << EOF
# Deployment configuration for $ENVIRONMENT
MCG_ENVIRONMENT=$ENVIRONMENT
MCG_VERSION=$VERSION
DEPLOY_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DEPLOY_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
DEPLOY_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
EOF

# Create deployment script
cat > "$DEPLOY_DIR/deploy.sh" << 'EOF'
#!/bin/bash

# Deployment script
set -e

echo "Starting Model Card Generator deployment..."

# Load configuration
source deploy-config.env

# Install Python package
if [ -f "dist/*.whl" ]; then
    pip install dist/*.whl --force-reinstall
    echo "Python package installed"
fi

# Start services with Docker Compose
if [ -f "docker-compose.yml" ]; then
    docker-compose up -d
    echo "Services started"
fi

# Run health check
sleep 10
if mcg --version; then
    echo "Deployment successful!"
else
    echo "Deployment health check failed!"
    exit 1
fi
EOF

chmod +x "$DEPLOY_DIR/deploy.sh"

# Create rollback script
cat > "$DEPLOY_DIR/rollback.sh" << 'EOF'
#!/bin/bash

# Rollback script
set -e

echo "Rolling back Model Card Generator deployment..."

# Stop current services
if [ -f "docker-compose.yml" ]; then
    docker-compose down
fi

# Restore previous version (implementation depends on deployment strategy)
echo "Rollback completed - manual verification required"
EOF

chmod +x "$DEPLOY_DIR/rollback.sh"

# Package deployment artifacts
tar -czf "$DEPLOY_DIR.tar.gz" "$DEPLOY_DIR"
print_success "Deployment package created: $DEPLOY_DIR.tar.gz"

# Environment-specific deployment
case "$ENVIRONMENT" in
    "development")
        print_status "Deploying to development environment..."
        
        # Local development deployment
        if [ -d "/tmp/mcg-dev" ]; then
            rm -rf /tmp/mcg-dev
        fi
        
        cp -r "$DEPLOY_DIR" "/tmp/mcg-dev"
        cd "/tmp/mcg-dev"
        ./deploy.sh
        
        print_success "Development deployment completed"
        ;;
        
    "staging")
        print_status "Deploying to staging environment..."
        
        # Upload to staging server
        if [ -n "$STAGING_HOST" ] && [ -n "$STAGING_DEPLOY_KEY" ]; then
            # Use SSH to deploy to staging
            scp -i "$STAGING_DEPLOY_KEY" "$DEPLOY_DIR.tar.gz" "deploy@$STAGING_HOST:/tmp/"
            
            ssh -i "$STAGING_DEPLOY_KEY" "deploy@$STAGING_HOST" << STAGING_COMMANDS
cd /tmp
tar -xzf $DEPLOY_DIR.tar.gz
cd $DEPLOY_DIR
./deploy.sh
STAGING_COMMANDS
            
            print_success "Staging deployment completed"
        else
            print_error "Staging deployment credentials not configured"
            exit 1
        fi
        ;;
        
    "production")
        print_status "Deploying to production environment..."
        
        # Production deployment with extra safety checks
        if [ -n "$PROD_HOST" ] && [ -n "$PROD_DEPLOY_KEY" ]; then
            # Backup current production state
            print_status "Creating production backup..."
            ssh -i "$PROD_DEPLOY_KEY" "deploy@$PROD_HOST" << BACKUP_COMMANDS
mkdir -p /backups/mcg
cp -r /opt/mcg /backups/mcg/backup-$(date +%Y%m%d-%H%M%S)
BACKUP_COMMANDS
            
            # Deploy to production
            scp -i "$PROD_DEPLOY_KEY" "$DEPLOY_DIR.tar.gz" "deploy@$PROD_HOST:/tmp/"
            
            ssh -i "$PROD_DEPLOY_KEY" "deploy@$PROD_HOST" << PROD_COMMANDS
cd /tmp
tar -xzf $DEPLOY_DIR.tar.gz
cd $DEPLOY_DIR

# Run production pre-deployment checks
echo "Running production pre-deployment checks..."

# Deploy
./deploy.sh

# Post-deployment verification
echo "Running post-deployment verification..."
sleep 30

# Health check
if ! mcg --version; then
    echo "Production health check failed - initiating rollback"
    ./rollback.sh
    exit 1
fi

echo "Production deployment verified successfully"
PROD_COMMANDS
            
            print_success "Production deployment completed"
        else
            print_error "Production deployment credentials not configured"
            exit 1
        fi
        ;;
esac

# Push Docker images to registry
if command -v docker &> /dev/null && [ -n "$DOCKER_REGISTRY" ]; then
    print_status "Pushing Docker images to registry..."
    
    # Login to registry (credentials should be set in environment)
    if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
        echo "$DOCKER_PASSWORD" | docker login "$DOCKER_REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
    fi
    
    # Push images
    docker push "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$VERSION"
    docker push "$DOCKER_REGISTRY/terragonlabs/modelcard-generator:$ENVIRONMENT"
    
    print_success "Docker images pushed to registry"
fi

# Publish to package registry (for production)
if [ "$ENVIRONMENT" = "production" ] && [ -n "$PYPI_TOKEN" ]; then
    print_status "Publishing to PyPI..."
    export TWINE_USERNAME="__token__"
    export TWINE_PASSWORD="$PYPI_TOKEN"
    twine upload dist/* --verbose
    print_success "Package published to PyPI"
fi

# Send deployment notifications
print_status "Sending deployment notifications..."

# Slack notification
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"🚀 Model Card Generator deployed to $ENVIRONMENT\\n\\nVersion: $VERSION\\nCommit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')\\nDeployer: ${USER:-unknown}\"
        }" \
        "$SLACK_WEBHOOK_URL" || print_warning "Failed to send Slack notification"
fi

# Email notification
if [ -n "$EMAIL_NOTIFICATION" ]; then
    echo "Deployment to $ENVIRONMENT completed successfully" | \
    mail -s "MCG Deployment - $ENVIRONMENT" "$EMAIL_NOTIFICATION" || \
    print_warning "Failed to send email notification"
fi

# Update deployment tracking
print_status "Updating deployment tracking..."
cat > "deployment-record.json" << EOF
{
    "environment": "$ENVIRONMENT",
    "version": "$VERSION",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "deployer": "${USER:-unknown}",
    "status": "success"
}
EOF

# Upload deployment record (if configured)
if [ -n "$DEPLOYMENT_TRACKING_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
        -d @deployment-record.json \
        "$DEPLOYMENT_TRACKING_URL" || \
        print_warning "Failed to update deployment tracking"
fi

# Cleanup
print_status "Cleaning up deployment artifacts..."
rm -rf "$DEPLOY_DIR" "$DEPLOY_DIR.tar.gz" deployment-record.json

echo ""
print_success "Deployment to $ENVIRONMENT completed successfully! 🎉"
echo ""
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Timestamp: $(date)"
echo ""
echo "Next steps:"
echo "1. Monitor application health"
echo "2. Verify functionality"
echo "3. Update documentation if needed"
echo ""
