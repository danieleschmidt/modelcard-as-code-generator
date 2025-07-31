# GitHub Actions Workflow Integration Guide

This document provides comprehensive guidance for integrating and customizing the GitHub Actions workflows implemented for the Model Card Generator project.

## 🔄 Workflow Overview

The project implements a comprehensive CI/CD pipeline with the following workflows:

| Workflow | Trigger | Purpose | Duration |
|----------|---------|---------|----------|
| **CI/CD Pipeline** | Push, PR | Core testing, security, deployment | ~15-20 min |
| **Release Pipeline** | Push to main, Manual | Automated releases with semantic versioning | ~25-30 min |
| **Security & Compliance** | Daily, Push, PR | Security scanning and compliance monitoring | ~10-15 min |
| **Performance Monitoring** | Daily, Push, PR | Performance benchmarking and regression detection | ~20-25 min |
| **Chaos Engineering** | Weekly, Manual | Resilience testing and fault injection | ~30-45 min |

## 🚀 Quick Setup

### 1. Required Secrets

Add these secrets to your GitHub repository settings (`Settings > Secrets and variables > Actions`):

```bash
# Required for all workflows
GITHUB_TOKEN                    # Automatically provided by GitHub

# Required for PyPI publishing
PYPI_API_TOKEN                  # PyPI API token for package publishing

# Optional - for enhanced notifications  
SLACK_WEBHOOK_URL              # Slack webhook for notifications

# Optional - for security scanning
GITLEAKS_LICENSE               # GitLeaks Pro license (if available)

# Optional - for infrastructure deployment
AWS_ACCESS_KEY_ID              # AWS access key for Terraform
AWS_SECRET_ACCESS_KEY          # AWS secret key for Terraform
DATADOG_API_KEY                # DataDog API key for monitoring
DATADOG_APP_KEY                # DataDog application key
```

### 2. Repository Settings

Configure the following repository settings:

```yaml
# Branch Protection Rules (Settings > Branches)
main:
  - Require a pull request before merging
  - Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Required status checks:
    - "Security & Compliance Scan"
    - "Code Quality & Standards" 
    - "Tests (Python 3.11)"
    - "Performance & Benchmarks"
  - Restrict pushes that create files larger than 100 MB
  - Require signed commits (recommended)

# General Settings
- Allow merge commits: ✅
- Allow squash merging: ✅  
- Allow rebase merging: ✅
- Automatically delete head branches: ✅
- Enable Dependabot security updates: ✅
```

### 3. Environment Setup

Create GitHub Environments for deployment control:

```yaml
# Production Environment (Settings > Environments)
production:
  protection_rules:
    - required_reviewers: ["@terragon-labs/maintainers"]
    - deployment_branches: ["main"]
  environment_secrets:
    - PYPI_API_TOKEN
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
  variables:
    - ENVIRONMENT: "production"
    - DEPLOY_REGION: "us-west-2"

# Staging Environment  
staging:
  protection_rules:
    - deployment_branches: ["main", "develop"]
  variables:
    - ENVIRONMENT: "staging"
    - DEPLOY_REGION: "us-west-2"
```

## 📋 Workflow Details

### CI/CD Pipeline (`ci.yml`)

**Purpose**: Comprehensive continuous integration and deployment

**Key Features**:
- Multi-Python version testing (3.9, 3.10, 3.11, 3.12)
- Security scanning (Trivy, Bandit, Safety)
- Code quality checks (Black, Ruff, MyPy)
- Container building and scanning
- Performance benchmarking
- Automated deployment to production

**Customization Options**:

```yaml
# Environment Variables (at workflow level)
env:
  PYTHON_VERSION: '3.11'          # Default Python version
  NODE_VERSION: '18'              # Node.js version for tools
  REGISTRY: ghcr.io              # Container registry
  IMAGE_NAME: ${{ github.repository }}

# Matrix Testing (modify test job)
strategy:
  fail-fast: false
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']  # Modify versions
    os: [ubuntu-latest]                               # Add: windows-latest, macos-latest
```

**Skip Options**: Add to commit message to skip certain jobs:
- `[skip ci]` - Skip entire workflow
- `[skip tests]` - Skip test suite  
- `[skip security]` - Skip security scans
- `[skip performance]` - Skip performance tests

### Release Pipeline (`release.yml`)

**Purpose**: Automated semantic versioning and multi-platform publishing

**Key Features**:
- Semantic versioning with conventional commits
- Multi-platform testing before release
- Container image building and publishing
- PyPI package publishing
- GitHub release creation with changelog
- Documentation updates

**Customization Options**:

```yaml
# Release Types (manual trigger)
inputs:
  release_type:
    options: [patch, minor, major, prerelease]
  skip_tests:
    type: boolean  # Emergency releases only

# Version Configuration (modify .releaserc.json)
{
  "branches": ["main"],  # Release from main branch only
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    ["@semantic-release/changelog", {
      "changelogFile": "CHANGELOG.md"
    }]
  ]
}
```

**Commit Message Format** (for automatic versioning):
```
feat: add new model card template        # Minor version bump
fix: resolve validation error            # Patch version bump  
perf: optimize template rendering        # Patch version bump
BREAKING CHANGE: change API structure    # Major version bump
```

### Security & Compliance (`security.yml`)

**Purpose**: Continuous security monitoring and compliance validation

**Key Features**:
- Dependency vulnerability scanning (Safety, pip-audit)
- Source code security analysis (Bandit, Semgrep)
- Container security scanning (Trivy)
- Secrets detection (GitLeaks, TruffleHog)
- Compliance monitoring (licenses, documentation)
- SBOM generation

**Customization Options**:

```yaml
# Scan Configuration
inputs:
  scan_type:
    options: [full, dependencies, container, secrets, compliance]
  severity_threshold:
    options: [LOW, MEDIUM, HIGH, CRITICAL]
    default: HIGH

# Thresholds (modify in workflow)
env:
  MEMORY_LIMIT_MB: 512           # Memory usage limit
  CPU_LIMIT_CORES: 2             # CPU usage limit
  VULNERABILITY_THRESHOLD: HIGH   # Fail on HIGH+ vulnerabilities
```

**Security Policy Integration**:
- Links to `SECURITY.md` for vulnerability reporting
- Integrates with GitHub Security Advisories
- Supports custom security patterns and rules

### Performance Monitoring (`performance.yml`)

**Purpose**: Continuous performance monitoring and regression detection

**Key Features**:
- Multi-suite benchmarking (core, CLI, templates, validation)
- Memory profiling and leak detection
- CPU profiling and analysis
- I/O performance testing
- Performance regression detection
- Performance dashboard generation

**Customization Options**:

```yaml
# Benchmark Configuration
env:
  BENCHMARK_ITERATIONS: 100      # Number of benchmark iterations
  MEMORY_LIMIT_MB: 512          # Memory usage threshold
  CPU_LIMIT_CORES: 2            # CPU usage threshold

# Performance Targets
inputs:
  performance_target:
    options: [baseline, optimized, stress]
  benchmark_type:
    options: [full, memory, cpu, io, network, custom]
```

**Performance Thresholds**:
```python
# Modify in workflow scripts
PERFORMANCE_THRESHOLDS = {
    'memory_usage_mb': 512,
    'cpu_time_seconds': 30,
    'io_read_time_seconds': 5,
    'io_write_time_seconds': 2,
    'regression_threshold_percent': 10
}
```

### Chaos Engineering (`chaos-engineering.yml`)

**Purpose**: Resilience testing and system reliability validation

**Key Features**:
- Network latency simulation
- Memory pressure testing  
- CPU stress testing
- Fault injection and error handling
- Graceful degradation validation
- Recovery time measurement

**Customization Options**:

```yaml
# Chaos Configuration
inputs:
  chaos_level:
    options: [basic, intermediate, advanced, extreme]
  target_environment:
    options: [staging, production-safe, isolated]
  duration_minutes:
    default: 30

# Safety Controls
env:
  CHAOS_SAFE_MODE: true          # Enable safety limits
  MAX_IMPACT_DURATION: 600       # Maximum chaos duration (seconds)
```

**Chaos Experiments**:
- `basic`: Network latency, memory pressure, CPU stress
- `intermediate`: + disk I/O, process termination
- `advanced`: + container termination, network partitions
- `extreme`: All available chaos experiments

## 🔧 Advanced Configuration

### Custom Workflow Triggers

```yaml
# Add to any workflow for custom triggers
on:
  schedule:
    - cron: '0 2 * * 1'           # Weekly on Monday at 2 AM
  workflow_dispatch:               # Manual trigger
    inputs:
      custom_input:
        description: 'Custom parameter'
        required: true
        type: choice
        options: [option1, option2]
  repository_dispatch:             # External API trigger
    types: [custom-event]
```

### Matrix Strategy Customization

```yaml
# Complex matrix configurations
strategy:
  fail-fast: false
  matrix:
    include:
      - python-version: '3.9'
        os: ubuntu-latest
        arch: x64
      - python-version: '3.11'
        os: windows-latest
        arch: x64
      - python-version: '3.12'
        os: macos-latest
        arch: arm64
    exclude:
      - python-version: '3.9'
        os: macos-latest
```

### Conditional Job Execution

```yaml
# Run jobs based on conditions
jobs:
  deploy:
    if: github.ref == 'refs/heads/main' && !contains(github.event.head_commit.message, '[skip deploy]')
    
  security-scan:
    if: contains(github.event.pull_request.labels.*.name, 'security-review')
    
  performance-test:
    if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[perf test]')
```

### Environment-Specific Configurations

```yaml
# Different configurations per environment
- name: Set environment variables
  run: |
    if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
      echo "ENVIRONMENT=production" >> $GITHUB_ENV
      echo "DEPLOY_REPLICAS=3" >> $GITHUB_ENV
      echo "RESOURCE_LIMITS=high" >> $GITHUB_ENV
    elif [[ "${{ github.ref }}" == "refs/heads/develop" ]]; then
      echo "ENVIRONMENT=staging" >> $GITHUB_ENV
      echo "DEPLOY_REPLICAS=2" >> $GITHUB_ENV
      echo "RESOURCE_LIMITS=medium" >> $GITHUB_ENV
    else
      echo "ENVIRONMENT=development" >> $GITHUB_ENV
      echo "DEPLOY_REPLICAS=1" >> $GITHUB_ENV
      echo "RESOURCE_LIMITS=low" >> $GITHUB_ENV
    fi
```

## 🔒 Security Best Practices

### Secret Management

```yaml
# Use secrets for sensitive data
- name: Deploy to production
  env:
    API_KEY: ${{ secrets.API_KEY }}
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
  run: |
    # Never echo secrets
    echo "Deploying with API key: ${API_KEY:0:8}..."
```

### Permissions Configuration

```yaml
# Minimal permissions per workflow
permissions:
  contents: read                 # Read repository contents
  security-events: write         # Write security scan results
  pull-requests: write          # Comment on PRs
  issues: write                 # Create issues
  packages: write               # Publish packages
  id-token: write               # OIDC token for cloud authentication
```

### Input Validation

```yaml
# Validate inputs and parameters
- name: Validate inputs
  run: |
    if [[ ! "${{ github.event.inputs.environment }}" =~ ^(dev|staging|prod)$ ]]; then
      echo "Invalid environment specified"
      exit 1
    fi
```

## 📊 Monitoring and Alerting

### GitHub Integration

```yaml
# Create issues for failures
- name: Create failure issue
  if: failure()
  uses: actions/github-script@v6
  with:
    script: |
      github.rest.issues.create({
        owner: context.repo.owner,
        repo: context.repo.repo,
        title: `Workflow failure: ${context.workflow}`,
        body: `Workflow failed: ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
        labels: ['bug', 'ci/cd', 'needs-investigation']
      })
```

### Slack Notifications

```yaml
# Notify Slack on important events
- name: Notify Slack
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: |
      ${{ job.status == 'success' && '✅' || '❌' }} Workflow: ${{ github.workflow }}
      Branch: ${{ github.ref_name }}
      Commit: ${{ github.sha }}
      Author: ${{ github.actor }}
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Performance Dashboards

```yaml
# Update performance dashboard
- name: Update dashboard
  run: |
    python scripts/update_performance_dashboard.py \
      --results performance-results/ \
      --dashboard-url "https://terragonlabs.github.io/modelcard-generator/performance"
```

## 🛠️ Troubleshooting

### Common Issues

**1. Workflow Permission Errors**
```yaml
# Solution: Add required permissions
permissions:
  contents: write
  packages: write
```

**2. Secret Access Issues**
```bash
# Verify secrets are configured
# Settings > Secrets and variables > Actions
# Check environment-specific secrets
```

**3. Performance Test Failures**
```yaml
# Solution: Adjust thresholds or skip performance tests
env:
  SKIP_PERFORMANCE_TESTS: true
```

**4. Security Scan False Positives**
```yaml
# Solution: Configure ignore patterns
bandit:
  exclude_dirs: [tests/, docs/]
  skip_tests: [B101, B601]
```

### Debug Techniques

```yaml
# Enable debug logging
- name: Debug workflow
  run: |
    echo "::debug::Debug information"
    echo "Environment: ${{ runner.os }}"
    echo "Python version: ${{ matrix.python-version }}"
    env
```

```yaml
# Conditional debugging
- name: Debug on failure
  if: failure()
  run: |
    echo "Workflow failed, gathering debug info..."
    ps aux
    df -h
    free -m
```

## 🔄 Workflow Maintenance

### Regular Updates

1. **Monthly**: Update action versions
2. **Quarterly**: Review and update dependencies
3. **Bi-annually**: Review workflow effectiveness and optimization opportunities

### Version Pinning Strategy

```yaml
# Recommended pinning strategy
- uses: actions/checkout@v4           # Pin to major version
- uses: actions/setup-python@v4.7.1   # Pin to specific version for critical actions
- uses: user/action@sha256            # Pin to commit SHA for security-critical actions
```

### Performance Optimization

```yaml
# Cache dependencies
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}

# Parallel job execution
jobs:
  test:
    strategy:
      matrix:
        python-version: [3.9, 3.11, 3.12]
      max-parallel: 3
```

## 📖 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Action Marketplace](https://github.com/marketplace)
- [Security Hardening Guide](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

## 📞 Support

For workflow-related issues:

1. Check the [GitHub Actions logs](https://github.com/terragonlabs/modelcard-generator/actions)
2. Review this integration guide
3. Create an issue with the `ci/cd` label
4. Contact the DevOps team: `devops@terragonlabs.com`

---

*This document is automatically updated with workflow changes. Last updated: $(date +'%Y-%m-%d')*