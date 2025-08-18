#!/bin/bash
# Build automation script for Model Card Generator
# Handles multi-platform builds, security scanning, and SBOM generation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_NAME="modelcard-as-code-generator"
IMAGE_NAME="terragonlabs/modelcard-generator"
REGISTRY="${REGISTRY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TARGET="production"
PLATFORMS="linux/amd64,linux/arm64"
PUSH_IMAGE=false
SCAN_SECURITY=true
GENERATE_SBOM=true
RUN_TESTS=true
SKIP_CACHE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build automation script for Model Card Generator

OPTIONS:
    -t, --target TARGET     Build target (development|testing|production) [default: production]
    -p, --platforms PLATFORMS  Target platforms for multi-arch build [default: linux/amd64,linux/arm64]
    --push                  Push built images to registry
    --no-security-scan      Skip security vulnerability scanning
    --no-sbom               Skip SBOM generation
    --no-tests              Skip running tests
    --skip-cache            Skip Docker build cache
    -r, --registry REGISTRY Container registry URL
    -v, --version VERSION   Explicit version tag
    -h, --help              Show this help message

EXAMPLES:
    $0                                          # Build production image locally
    $0 --target development --no-tests          # Build dev image without tests
    $0 --push --registry ghcr.io/terragonlabs  # Build and push to GitHub Registry
    $0 --platforms linux/amd64 --version v1.2.0  # Single platform with explicit version

ENVIRONMENT VARIABLES:
    REGISTRY        - Default container registry
    VERSION         - Override version detection
    DOCKER_BUILDKIT - Enable BuildKit (recommended)
    CI              - Detect CI environment
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            -p|--platforms)
                PLATFORMS="$2"
                shift 2
                ;;
            --push)
                PUSH_IMAGE=true
                shift
                ;;
            --no-security-scan)
                SCAN_SECURITY=false
                shift
                ;;
            --no-sbom)
                GENERATE_SBOM=false
                shift
                ;;
            --no-tests)
                RUN_TESTS=false
                shift
                ;;
            --skip-cache)
                SKIP_CACHE=true
                shift
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Detect version from various sources
detect_version() {
    if [[ -n "${VERSION:-}" ]]; then
        echo "$VERSION"
        return
    fi
    
    # Try to get version from git tag
    if git describe --tags --exact-match 2>/dev/null; then
        git describe --tags --exact-match 2>/dev/null
        return
    fi
    
    # Try to get version from pyproject.toml
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]] && command -v python3 >/dev/null; then
        python3 -c "
import tomllib
with open('$PROJECT_ROOT/pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    print(data.get('project', {}).get('version', 'dev'))
" 2>/dev/null || echo "dev"
        return
    fi
    
    # Fallback to git short hash
    if git rev-parse --short HEAD 2>/dev/null; then
        echo "dev-$(git rev-parse --short HEAD)"
    else
        echo "dev"
    fi
}

# Validate build environment
validate_environment() {
    print_status "Validating build environment..."
    
    # Check required tools
    local required_tools=("docker" "git")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null; then
            print_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check Docker buildx for multi-platform builds
    if [[ "$PLATFORMS" == *","* ]] && ! docker buildx version >/dev/null 2>&1; then
        print_error "Docker buildx is required for multi-platform builds"
        exit 1
    fi
    
    # Validate build target
    case "$BUILD_TARGET" in
        development|testing|production|cicd)
            ;;
        *)
            print_error "Invalid build target: $BUILD_TARGET"
            print_error "Valid targets: development, testing, production, cicd"
            exit 1
            ;;
    esac
    
    print_success "Environment validation passed"
}

# Set up build environment
setup_build() {
    print_status "Setting up build environment..."
    
    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    export BUILDX_EXPERIMENTAL=1
    
    # Create buildx builder if needed for multi-platform builds
    if [[ "$PLATFORMS" == *","* ]]; then
        if ! docker buildx inspect multiarch >/dev/null 2>&1; then
            print_status "Creating multi-platform builder..."
            docker buildx create --name multiarch --driver docker-container --use
            docker buildx inspect --bootstrap
        else
            docker buildx use multiarch
        fi
    fi
    
    print_success "Build environment ready"
}

# Run tests before building
run_tests() {
    if [[ "$RUN_TESTS" != true ]]; then
        print_warning "Skipping tests as requested"
        return 0
    fi
    
    print_status "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run quick tests appropriate for the build
    case "$BUILD_TARGET" in
        development)
            make test-unit || { print_error "Unit tests failed"; exit 1; }
            ;;
        testing|cicd)
            make test || { print_error "Tests failed"; exit 1; }
            ;;
        production)
            make test-unit test-integration || { print_error "Tests failed"; exit 1; }
            ;;
    esac
    
    print_success "Tests passed"
}

# Build Docker image
build_image() {
    local version="$1"
    local full_image_name="${REGISTRY:+$REGISTRY/}$IMAGE_NAME"
    local build_args=(
        "--target" "$BUILD_TARGET"
        "--tag" "$full_image_name:$version"
        "--tag" "$full_image_name:latest"
        "--build-arg" "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--build-arg" "VERSION=$version"
        "--build-arg" "VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
        "--file" "$PROJECT_ROOT/Dockerfile"
        "$PROJECT_ROOT"
    )
    
    if [[ "$SKIP_CACHE" == true ]]; then
        build_args+=("--no-cache")
    fi
    
    print_status "Building Docker image..."
    print_status "Image: $full_image_name:$version"
    print_status "Target: $BUILD_TARGET"
    print_status "Platforms: $PLATFORMS"
    
    if [[ "$PLATFORMS" == *","* ]]; then
        # Multi-platform build
        build_args+=("--platform" "$PLATFORMS")
        
        if [[ "$PUSH_IMAGE" == true ]]; then
            build_args+=("--push")
        else
            print_warning "Multi-platform build requires --push to save images"
            build_args+=("--push")
            PUSH_IMAGE=true
        fi
        
        docker buildx build "${build_args[@]}"
    else
        # Single platform build
        if [[ -n "$PLATFORMS" ]] && [[ "$PLATFORMS" != "linux/amd64" ]]; then
            build_args+=("--platform" "$PLATFORMS")
        fi
        
        docker build "${build_args[@]}"
        
        if [[ "$PUSH_IMAGE" == true ]]; then
            docker push "$full_image_name:$version"
            docker push "$full_image_name:latest"
        fi
    fi
    
    print_success "Image built successfully: $full_image_name:$version"
}

# Security scanning with Trivy
security_scan() {
    if [[ "$SCAN_SECURITY" != true ]]; then
        print_warning "Skipping security scan as requested"
        return 0
    fi
    
    if ! command -v trivy >/dev/null; then
        print_warning "Trivy not found, attempting to install..."
        # Try to install trivy temporarily
        if command -v curl >/dev/null; then
            curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /tmp
            export PATH="/tmp:$PATH"
        else
            print_warning "Cannot install Trivy, skipping security scan"
            return 0
        fi
    fi
    
    local version="$1"
    local full_image_name="${REGISTRY:+$REGISTRY/}$IMAGE_NAME:$version"
    
    print_status "Running security scan on $full_image_name..."
    
    # Create reports directory
    mkdir -p "$PROJECT_ROOT/security-reports"
    
    # Scan for vulnerabilities
    trivy image \
        --format json \
        --output "$PROJECT_ROOT/security-reports/trivy-report.json" \
        "$full_image_name"
    
    # Generate human-readable report
    trivy image \
        --format table \
        --output "$PROJECT_ROOT/security-reports/trivy-report.txt" \
        "$full_image_name"
    
    # Check for high/critical vulnerabilities
    local high_critical_count
    high_critical_count=$(trivy image --format json "$full_image_name" | \
        jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH" or .Severity == "CRITICAL")] | length')
    
    if [[ "$high_critical_count" -gt 0 ]]; then
        print_warning "Found $high_critical_count high/critical vulnerabilities"
        print_warning "Check security-reports/trivy-report.txt for details"
    else
        print_success "No high/critical vulnerabilities found"
    fi
}

# Generate SBOM
generate_sbom() {
    if [[ "$GENERATE_SBOM" != true ]]; then
        print_warning "Skipping SBOM generation as requested"
        return 0
    fi
    
    print_status "Generating Software Bill of Materials (SBOM)..."
    
    cd "$PROJECT_ROOT"
    
    # Generate different SBOM formats
    python3 scripts/generate-sbom.py --format spdx-json --output sbom/
    python3 scripts/generate-sbom.py --format cyclone-json --output sbom/
    
    # Try to generate with syft if available (for container image SBOM)
    if command -v syft >/dev/null; then
        local version="$1"
        local full_image_name="${REGISTRY:+$REGISTRY/}$IMAGE_NAME:$version"
        
        syft "$full_image_name" -o spdx-json > "sbom/${PROJECT_NAME}-container-spdx.json"
        syft "$full_image_name" -o cyclone-json > "sbom/${PROJECT_NAME}-container-cyclone.json"
    else
        print_warning "Syft not available, skipping container SBOM generation"
    fi
    
    print_success "SBOM generated in sbom/ directory"
}

# Generate build report
generate_build_report() {
    local version="$1"
    local build_time="$2"
    local full_image_name="${REGISTRY:+$REGISTRY/}$IMAGE_NAME"
    
    print_status "Generating build report..."
    
    cat > "$PROJECT_ROOT/build-report.json" << EOF
{
  "project": "$PROJECT_NAME",
  "version": "$version",
  "build_target": "$BUILD_TARGET",
  "platforms": "$PLATFORMS",
  "image": "$full_image_name:$version",
  "build_time": "$build_time",
  "build_date": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "pushed": $PUSH_IMAGE,
  "security_scanned": $SCAN_SECURITY,
  "sbom_generated": $GENERATE_SBOM,
  "tests_run": $RUN_TESTS
}
EOF
    
    print_success "Build report generated: build-report.json"
}

# Main build process
main() {
    local start_time
    start_time=$(date +%s)
    
    parse_args "$@"
    
    print_status "Starting build process for $PROJECT_NAME"
    print_status "Build target: $BUILD_TARGET"
    print_status "Platforms: $PLATFORMS"
    
    validate_environment
    setup_build
    
    local version
    version=$(detect_version)
    print_status "Detected version: $version"
    
    run_tests
    build_image "$version"
    security_scan "$version"
    generate_sbom "$version"
    
    local end_time
    end_time=$(date +%s)
    local build_time=$((end_time - start_time))
    
    generate_build_report "$version" "$build_time"
    
    print_success "Build completed successfully in ${build_time}s"
    print_success "Image: ${REGISTRY:+$REGISTRY/}$IMAGE_NAME:$version"
    
    if [[ "$PUSH_IMAGE" == true ]]; then
        print_success "Image pushed to registry"
    else
        print_status "To push the image, run with --push flag"
    fi
}

# Run main function with all arguments
main "$@"