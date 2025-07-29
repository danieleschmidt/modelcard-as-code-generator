# GitHub Actions Workflows

This document outlines the required GitHub Actions workflows for the Model Card Generator project.

## Overview

The following workflows provide comprehensive CI/CD automation for development, testing, security, and deployment processes.

## Core Workflows

### 1. Continuous Integration (`ci.yml`)

**Trigger**: Push to any branch, Pull requests
**Location**: `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Run quality checks
      run: make quality-check
    
    - name: Run tests with coverage
      run: make test-coverage
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Security Scanning (`security.yml`)

**Trigger**: Push to main, scheduled weekly
**Location**: `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit Security Scan
      uses: securecodewarrior/github-action-bandit@v1
      with:
        exit_zero: false
    
    - name: Run Safety Security Scan
      run: |
        pip install safety
        safety check --json --output safety-report.json
    
    - name: Run Semgrep
      uses: returntocorp/semgrep-action@v1
      with:
        config: >-
          p/security-audit
          p/secrets
          p/python
```

### 3. Release Automation (`release.yml`)

**Trigger**: Tagged releases
**Location**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write
  packages: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Build distribution
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        user: __token__
        password: ${{ secrets.PYPI_API_TOKEN }}
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

## Advanced Workflows

### 4. Performance Testing (`performance.yml`)

```yaml
name: Performance Testing

on:
  pull_request:
    paths:
      - 'src/**'
      - 'tests/performance/**'
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Run performance tests
      run: |
        pip install -e ".[test]"
        pytest tests/performance/ -v --benchmark-json=benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

### 5. Documentation Deployment (`docs.yml`)

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e ".[docs]"
    
    - name: Deploy documentation
      run: |
        mkdocs gh-deploy --force
```

## Workflow Configuration

### Required Secrets

Set these secrets in your GitHub repository settings:

- `PYPI_API_TOKEN`: PyPI token for package publishing
- `CODECOV_TOKEN`: Codecov token for coverage reporting
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password/token

### Required Permissions

Ensure workflows have these permissions:

- `contents: write` - For creating releases
- `packages: write` - For publishing packages  
- `security-events: write` - For security scanning

## Integration Guidelines

### Pre-commit Integration

Workflows complement pre-commit hooks:
- Pre-commit: Fast local checks
- CI: Comprehensive cross-platform testing
- Security: Deep vulnerability scanning

### Branch Protection

Configure branch protection rules:

```yaml
# Required status checks
required_status_checks:
  - test (3.9)
  - test (3.10) 
  - test (3.11)
  - test (3.12)
  - security
```

### Manual Workflow Triggers

All workflows support manual dispatch:

```bash
# Trigger security scan
gh workflow run security.yml

# Trigger performance tests
gh workflow run performance.yml
```

## Monitoring and Alerts

### Failure Notifications

Configure notifications for:
- Build failures
- Security vulnerabilities
- Performance regressions
- Deployment issues

### Metrics Collection

Track these workflow metrics:
- Build success rate
- Test execution time
- Deployment frequency
- Security scan results

## Troubleshooting

### Common Issues

1. **Test Failures**: Check matrix compatibility
2. **Security Alerts**: Review dependency updates
3. **Performance Regressions**: Analyze benchmark trends
4. **Deployment Failures**: Verify secrets and permissions

### Debug Actions

Use these for debugging:

```yaml
- name: Debug Environment
  run: |
    echo "Python: $(python --version)"
    echo "Pip: $(pip --version)"
    echo "PWD: $(pwd)"
    ls -la
```

## Best Practices

1. **Fast Feedback**: Keep CI builds under 10 minutes
2. **Parallel Execution**: Use matrix strategies
3. **Caching**: Cache dependencies and build artifacts  
4. **Security First**: Run security scans on every push
5. **Documentation**: Keep workflows documented and updated