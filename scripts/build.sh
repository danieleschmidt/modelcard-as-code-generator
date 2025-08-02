#!/bin/bash
# Build script for Model Card Generator
# Supports multiple targets and build configurations

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="terragonlabs/modelcard-generator"
BUILD_TARGET="runtime"
PUSH_IMAGE=false
RUN_TESTS=false
PLATFORM=""
VERSION=""
BUILD_ARGS=""
DOCKER_REGISTRY=""
NO_CACHE=false

# Get version from pyproject.toml or git
get_version() {
    if [ -f "pyproject.toml" ]; then
        grep -E '^version = ' pyproject.toml | cut -d'"' -f2
    else
        git describe --tags --always --dirty 2>/dev/null || echo "dev"
    fi
}

# Print usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Docker images for Model Card Generator

OPTIONS:
    -t, --target TARGET         Build target (runtime, development, cicd, docs, cli)
    -v, --version VERSION       Version tag for the image (default: auto-detect)
    -p, --push                  Push image to registry after build
    -r, --registry REGISTRY     Docker registry URL
    -T, --test                  Run tests after build
    -P, --platform PLATFORM     Target platform (e.g., linux/amd64,linux/arm64)
    --no-cache                  Build without using cache
    --build-arg KEY=VALUE       Pass build argument
    -h, --help                  Show this help message

TARGETS:
    runtime                     Production runtime image (default)
    development                 Development image with dev dependencies
    cicd                       CI/CD image with testing tools
    docs                       Documentation image with MkDocs
    cli                        Minimal CLI-only image

EXAMPLES:
    $0                                          # Build runtime image
    $0 -t development                           # Build development image
    $0 -t runtime -v 1.2.3 -p                 # Build and push v1.2.3
    $0 -t cicd -T                              # Build CI image and run tests
    $0 --platform linux/amd64,linux/arm64      # Multi-platform build
    $0 --build-arg VERSION=1.0.0               # Pass custom build arg

EOF
}

# Log messages with colors
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
}

# Check if buildx is available for multi-platform builds
check_buildx() {
    if [ -n "$PLATFORM" ]; then
        if ! docker buildx version &> /dev/null; then
            log_error "Docker buildx is required for multi-platform builds"
            exit 1
        fi
    fi
}

# Build the Docker image
build_image() {
    local tag="${IMAGE_NAME}:${VERSION}"
    local latest_tag="${IMAGE_NAME}:latest"
    
    if [ "$BUILD_TARGET" != "runtime" ]; then
        tag="${IMAGE_NAME}:${BUILD_TARGET}-${VERSION}"
        latest_tag="${IMAGE_NAME}:${BUILD_TARGET}"
    fi

    log_info "Building Docker image: $tag"
    log_info "Target: $BUILD_TARGET"
    log_info "Platform: ${PLATFORM:-default}"

    # Prepare build command
    local build_cmd="docker"
    local build_args_array=()

    # Use buildx for multi-platform builds
    if [ -n "$PLATFORM" ]; then
        build_cmd="docker buildx"
        build_args_array+=("build")
        build_args_array+=("--platform" "$PLATFORM")
        
        if [ "$PUSH_IMAGE" = true ]; then
            build_args_array+=("--push")
        else
            build_args_array+=("--load")
        fi
    else
        build_args_array+=("build")
    fi

    # Add build arguments
    build_args_array+=("--target" "$BUILD_TARGET")
    build_args_array+=("--tag" "$tag")
    build_args_array+=("--tag" "$latest_tag")

    # Add custom build arguments
    build_args_array+=("--build-arg" "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')")
    build_args_array+=("--build-arg" "VERSION=$VERSION")
    build_args_array+=("--build-arg" "VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')")

    if [ -n "$BUILD_ARGS" ]; then
        IFS=',' read -ra ARGS <<< "$BUILD_ARGS"
        for arg in "${ARGS[@]}"; do
            build_args_array+=("--build-arg" "$arg")
        done
    fi

    # Add cache options
    if [ "$NO_CACHE" = true ]; then
        build_args_array+=("--no-cache")
    fi

    # Add context
    build_args_array+=(".")

    # Execute build command
    log_info "Executing: $build_cmd ${build_args_array[*]}"
    
    if ! $build_cmd "${build_args_array[@]}"; then
        log_error "Docker build failed"
        exit 1
    fi

    log_success "Successfully built $tag"

    # Push image if requested and not using buildx with --push
    if [ "$PUSH_IMAGE" = true ] && [ -z "$PLATFORM" ]; then
        push_image "$tag" "$latest_tag"
    fi
}

# Push image to registry
push_image() {
    local tag=$1
    local latest_tag=$2

    if [ -n "$DOCKER_REGISTRY" ]; then
        tag="${DOCKER_REGISTRY}/${tag}"
        latest_tag="${DOCKER_REGISTRY}/${latest_tag}"
    fi

    log_info "Pushing image: $tag"
    
    if ! docker push "$tag"; then
        log_error "Failed to push $tag"
        exit 1
    fi

    log_info "Pushing latest tag: $latest_tag"
    
    if ! docker push "$latest_tag"; then
        log_error "Failed to push $latest_tag"
        exit 1
    fi

    log_success "Successfully pushed images to registry"
}

# Run tests using the built image
run_tests() {
    local test_tag="${IMAGE_NAME}:cicd-${VERSION}"
    
    if [ "$BUILD_TARGET" != "cicd" ]; then
        log_info "Building test image for testing..."
        docker build --target cicd --tag "$test_tag" \
            --build-arg "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
            --build-arg "VERSION=$VERSION" \
            --build-arg "VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
            .
    else
        test_tag="${IMAGE_NAME}:${VERSION}"
    fi

    log_info "Running tests using image: $test_tag"
    
    if ! docker run --rm \
        -v "$(pwd):/app" \
        -w /app \
        "$test_tag" \
        python -m pytest --cov=modelcard_generator --cov-report=term-missing -v; then
        log_error "Tests failed"
        exit 1
    fi

    log_success "All tests passed"
}

# Security scan using Trivy
security_scan() {
    local tag="${IMAGE_NAME}:${VERSION}"
    
    if [ "$BUILD_TARGET" != "runtime" ]; then
        tag="${IMAGE_NAME}:${BUILD_TARGET}-${VERSION}"
    fi

    log_info "Running security scan on: $tag"
    
    if command -v trivy &> /dev/null; then
        if ! trivy image --exit-code 1 --severity HIGH,CRITICAL "$tag"; then
            log_warning "Security scan found vulnerabilities"
        else
            log_success "Security scan passed"
        fi
    else
        log_warning "Trivy not available, skipping security scan"
    fi
}

# Get image size
get_image_size() {
    local tag="${IMAGE_NAME}:${VERSION}"
    
    if [ "$BUILD_TARGET" != "runtime" ]; then
        tag="${IMAGE_NAME}:${BUILD_TARGET}-${VERSION}"
    fi

    local size=$(docker images --format "table {{.Size}}" "$tag" | tail -n +2)
    log_info "Image size: $size"
}

# Clean up old images
cleanup() {
    log_info "Cleaning up old images..."
    
    # Remove old development images
    docker images "${IMAGE_NAME}" --format "table {{.Tag}} {{.ID}}" | \
    grep -E "(dev-|test-|docs-)" | \
    head -n -3 | \
    awk '{print $2}' | \
    xargs -r docker rmi || true

    # Clean up build cache
    docker buildx prune -f || docker builder prune -f || true

    log_success "Cleanup completed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -T|--test)
            RUN_TESTS=true
            shift
            ;;
        -P|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --build-arg)
            if [ -n "$BUILD_ARGS" ]; then
                BUILD_ARGS="$BUILD_ARGS,$2"
            else
                BUILD_ARGS="$2"
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate build target
case $BUILD_TARGET in
    runtime|development|cicd|docs|cli)
        ;;
    *)
        log_error "Invalid build target: $BUILD_TARGET"
        log_error "Valid targets: runtime, development, cicd, docs, cli"
        exit 1
        ;;
esac

# Set default version if not provided
if [ -z "$VERSION" ]; then
    VERSION=$(get_version)
    log_info "Auto-detected version: $VERSION"
fi

# Main execution
main() {
    log_info "Starting Docker build process..."
    log_info "Target: $BUILD_TARGET"
    log_info "Version: $VERSION"
    
    check_docker
    check_buildx
    
    # Build the image
    build_image
    
    # Get image size
    get_image_size
    
    # Run security scan
    if [ "$BUILD_TARGET" = "runtime" ] || [ "$BUILD_TARGET" = "cli" ]; then
        security_scan
    fi
    
    # Run tests if requested
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi
    
    # Cleanup if this is a development build
    if [ "$BUILD_TARGET" = "development" ]; then
        cleanup
    fi
    
    log_success "Build process completed successfully!"
    
    # Print usage instructions
    local tag="${IMAGE_NAME}:${VERSION}"
    if [ "$BUILD_TARGET" != "runtime" ]; then
        tag="${IMAGE_NAME}:${BUILD_TARGET}-${VERSION}"
    fi
    
    echo
    log_info "Image built: $tag"
    log_info "Usage examples:"
    echo "  docker run --rm $tag mcg --version"
    echo "  docker run --rm -v \$(pwd):/data $tag mcg generate /data/eval.json"
    
    if [ "$BUILD_TARGET" = "development" ]; then
        echo "  docker run --rm -it -v \$(pwd):/app/src $tag bash"
    fi
}

# Execute main function
main