# GitHub Workflows

This directory contains GitHub Actions workflows for the Model Card Generator project. Due to GitHub security restrictions, these workflow files need to be manually added to `.github/workflows/` by a repository maintainer with appropriate permissions.

## Available Workflows

### 1. CI Workflow (`ci.yml`)
Comprehensive continuous integration pipeline including:
- Multi-platform testing (Ubuntu, Windows, macOS)
- Python version matrix (3.9, 3.10, 3.11, 3.12)
- Code quality checks (linting, formatting, type checking)
- Security scanning (Bandit, Safety, CodeQL)
- Docker build and test
- Documentation build
- Performance testing

### 2. CD Workflow (`cd.yml`)
Continuous deployment pipeline including:
- Staging deployment (on main branch)
- Production deployment (on tags)
- Container image building and scanning
- Security validation
- PyPI package publishing
- Deployment notifications

### 3. Security Workflow (`security.yml`)
Comprehensive security scanning including:
- Dependency vulnerability scanning
- Static application security testing (SAST)
- Container security scanning
- Secret detection
- Infrastructure as Code security
- License compliance checking
- SBOM generation

### 4. Dependencies Workflow (`dependencies.yml`)
Automated dependency management including:
- Daily security update checks
- Weekly dependency updates (patch/minor)
- Automated PR creation for updates
- Comprehensive dependency auditing
- Update validation and testing

### 5. Release Workflow (`release.yml`)
Release management automation including:
- Automated changelog generation
- GitHub release creation
- Asset publishing
- Version tagging

## Manual Installation

To install these workflows, a repository maintainer should:

1. Copy the workflow files from this directory to `.github/workflows/`
2. Ensure the repository has the necessary secrets configured
3. Review and adjust any repository-specific settings
4. Test the workflows on a feature branch first

## Required Secrets

The workflows require the following GitHub secrets to be configured:

- `GITHUB_TOKEN` (automatically provided)
- `PYPI_API_TOKEN` (for PyPI publishing)
- `TEST_PYPI_API_TOKEN` (for test PyPI publishing)

## Permissions

The workflows require the following permissions:
- `contents: write` (for releases)
- `packages: write` (for container publishing)
- `security-events: write` (for security scanning)
- `pull-requests: write` (for automated PRs)

## Notes

These workflows are designed to work together as a comprehensive SDLC automation suite. They follow GitHub Actions best practices and include proper error handling, security measures, and reporting.