# Manual Setup Requirements

## Overview
This document outlines manual setup steps required after the automated SDLC implementation.

## GitHub Repository Settings

### Branch Protection Rules
Configure branch protection for `main` branch:
- Require pull request reviews before merging
- Require status checks to pass before merging  
- Require branches to be up to date before merging
- Include administrators in restrictions

### Repository Secrets
Configure the following secrets in GitHub repository settings:
- `PYPI_API_TOKEN` - For automated PyPI publishing
- `TEST_PYPI_API_TOKEN` - For test PyPI publishing

### GitHub Actions Workflows
Copy workflow files from `workflows/` directory to `.github/workflows/`:
- `ci.yml` - Continuous integration pipeline
- `cd.yml` - Continuous deployment pipeline
- `security.yml` - Security scanning automation
- `dependencies.yml` - Dependency management automation
- `release.yml` - Release management automation

## Required Permissions
Repository administrators must have:
- Admin access to configure branch protection
- Ability to manage repository secrets
- Permission to enable GitHub Actions

For more details, see [workflows/README.md](../workflows/README.md)