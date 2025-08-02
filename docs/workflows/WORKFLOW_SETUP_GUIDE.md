# Complete Workflow Setup Guide

## Overview

This guide provides step-by-step instructions for setting up GitHub Actions workflows for the Model Card Generator. Since GitHub Apps may have limited permissions, most workflows need to be created manually by repository administrators.

## Prerequisites

### Required Permissions

Repository administrators need:
- **Actions**: Write permissions for workflow management
- **Secrets**: Write permissions for secret configuration  
- **Branch Protection**: Admin permissions for protection rules
- **Packages**: Write permissions for container registry (if using)

### Required Secrets

Configure these secrets in your repository settings:

```bash
# PyPI Publishing
PYPI_API_TOKEN=pypi-your-token-here

# Docker Registry (if using)
DOCKER_USERNAME=your-docker-username
DOCKER_PASSWORD=your-docker-password

# Security Scanning
SONAR_TOKEN=your-sonar-token  # Optional
SNYK_TOKEN=your-snyk-token    # Optional

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## Workflow Installation

### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Template Files

Copy the workflow templates from `docs/github-workflows-templates/` to `.github/workflows/`:

```bash
# Core workflows (required)
cp docs/github-workflows-templates/ci.yml .github/workflows/
cp docs/github-workflows-templates/security.yml .github/workflows/
cp docs/github-workflows-templates/release.yml .github/workflows/

# Optional workflows
cp docs/github-workflows-templates/dependabot.yml .github/workflows/
cp docs/github-workflows-templates/performance.yml .github/workflows/
cp docs/github-workflows-templates/chaos-engineering.yml .github/workflows/
```

### Step 3: Configure Workflows

Each workflow may need customization based on your environment:

#### CI Workflow (`ci.yml`)

```yaml
# Example customizations needed:
name: CI

on:
  push:
    branches: [ main, develop ]  # Adjust branch names
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION_MATRIX: "['3.9', '3.10', '3.11', '3.12']"
  # Add environment-specific variables
```

#### Security Workflow (`security.yml`)

```yaml
# Configure scanning tools:
env:
  ENABLE_SONAR: false        # Set to true if using SonarCloud
  ENABLE_SNYK: false         # Set to true if using Snyk
  ENABLE_CODEQL: true        # GitHub's built-in scanner
```

#### Release Workflow (`release.yml`)

```yaml
# Configure release triggers:
on:
  push:
    tags:
      - 'v*'  # Triggers on version tags like v1.0.0
```

## Branch Protection Rules

### Required Protection Rules

Configure these rules for your main branch:

1. **Basic Protection**
   - Require pull request reviews (minimum 1 reviewer)
   - Dismiss stale reviews when new commits are pushed
   - Require review from code owners (if CODEOWNERS exists)

2. **Status Checks**
   - Require status checks to pass before merging
   - Required checks:
     - `ci / test-python-3.9`
     - `ci / test-python-3.10` 
     - `ci / test-python-3.11`
     - `ci / test-python-3.12`
     - `security / security-scan`
     - `security / dependency-scan`

3. **Additional Rules**
   - Require branches to be up to date
   - Require signed commits (recommended)
   - Include administrators in restrictions

### GitHub CLI Setup

```bash
# Install GitHub CLI if not already installed
# Then configure branch protection:

gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci / test-python-3.9","ci / test-python-3.10","ci / test-python-3.11","ci / test-python-3.12","security / security-scan"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null
```

## Secret Configuration

### Via GitHub Web Interface

1. Navigate to: `Settings` → `Secrets and variables` → `Actions`
2. Click `New repository secret`
3. Add each required secret:

### Via GitHub CLI

```bash
# Set PyPI token
gh secret set PYPI_API_TOKEN --body "pypi-your-token-here"

# Set Docker credentials
gh secret set DOCKER_USERNAME --body "your-username"
gh secret set DOCKER_PASSWORD --body "your-password"

# Set notification webhook
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/..."
```

### Environment-Specific Secrets

For production deployments, consider using environment-specific secrets:

```bash
# Production environment
gh secret set PYPI_API_TOKEN_PROD --env production
gh secret set DOCKER_REGISTRY_PROD --env production

# Staging environment  
gh secret set PYPI_API_TOKEN_STAGING --env staging
gh secret set DOCKER_REGISTRY_STAGING --env staging
```

## Workflow Testing

### Initial Testing Workflow

1. **Create Test Branch**
   ```bash
   git checkout -b test-workflows
   git push -u origin test-workflows
   ```

2. **Create Test PR**
   - Open a pull request from `test-workflows` to `main`
   - Verify all workflow checks pass
   - Check workflow logs for any issues

3. **Test Release Process**
   ```bash
   # Create and push a test tag
   git tag v0.0.1-test
   git push origin v0.0.1-test
   
   # Verify release workflow triggers
   # Check GitHub releases page
   ```

### Troubleshooting Common Issues

#### Permission Errors

```yaml
# If workflows fail due to permissions:
permissions:
  contents: read
  actions: write
  security-events: write
  pull-requests: write
```

#### Secret Access Issues

```yaml
# Verify secret names match exactly:
env:
  PYPI_TOKEN: ${{ secrets.PYPI_API_TOKEN }}  # Must match secret name
```

#### Matrix Strategy Failures

```yaml
# Debug matrix builds:
strategy:
  fail-fast: false  # Don't stop all jobs if one fails
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']
    os: [ubuntu-latest, windows-latest, macos-latest]
```

## Advanced Configuration

### Custom Workflow Variables

```yaml
# .github/workflows/variables.yml
env:
  # Global variables for all workflows
  PYTHON_DEFAULT_VERSION: '3.11'
  NODE_VERSION: '18'
  DOCKER_REGISTRY: 'ghcr.io'
  PACKAGE_NAME: 'modelcard-as-code-generator'
```

### Workflow Dependencies

```yaml
# Coordinate workflow execution:
name: Deploy
needs: [ci, security-scan, performance-test]
if: ${{ needs.ci.result == 'success' && needs.security-scan.result == 'success' }}
```

### Conditional Execution

```yaml
# Run workflows conditionally:
jobs:
  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    # Only deploy on main branch pushes
```

## Monitoring and Maintenance

### Workflow Health Monitoring

1. **Regular Review Schedule**
   - Weekly review of workflow success rates
   - Monthly security dependency updates
   - Quarterly workflow optimization review

2. **Key Metrics to Track**
   - Workflow success rate (target: >95%)
   - Average build time (target: <10 minutes)
   - Security scan coverage (target: 100%)
   - Dependency update frequency

3. **Alerting Setup**
   ```yaml
   # Add to workflows for failure notifications:
   - name: Notify on failure
     if: failure()
     uses: 8398a7/action-slack@v3
     with:
       status: failure
       webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
   ```

### Dependency Management

#### Dependabot Configuration

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 10
    target-branch: "develop"
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 5
```

#### Manual Dependency Updates

```bash
# Update workflow action versions quarterly:
grep -r "uses:" .github/workflows/ | grep "@v" | sort | uniq

# Check for security updates:
npm audit  # For Node.js actions
pip-audit  # For Python dependencies
```

## Security Considerations

### Workflow Security Best Practices

1. **Principle of Least Privilege**
   ```yaml
   permissions:
     contents: read      # Only what's needed
     actions: write      # For workflow dispatch
     security-events: write  # For security scanning
   ```

2. **Secret Management**
   - Use environment-specific secrets
   - Rotate secrets regularly (quarterly)
   - Never log secret values
   - Use masked logging where possible

3. **Third-Party Actions**
   ```yaml
   # Pin actions to specific versions (not @main)
   - uses: actions/checkout@v4.1.1
   - uses: actions/setup-python@v4.7.1
   ```

4. **Input Validation**
   ```yaml
   # Validate inputs in custom workflows:
   - name: Validate input
     run: |
       if [[ ! "${{ inputs.version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
         echo "Invalid version format"
         exit 1
       fi
   ```

### Security Scanning Configuration

#### CodeQL Setup

```yaml
# .github/workflows/codeql.yml
name: "CodeQL"

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 12 * * 1'  # Weekly scans

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}

    - name: Autobuild
      uses: github/codeql-action/autobuild@v3

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
```

## Integration with External Services

### Slack Notifications

```yaml
# Custom notification template:
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: custom
    custom_payload: |
      {
        "text": "Model Card Generator Build",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Build Status:* ${{ job.status }}\n*Repository:* ${{ github.repository }}\n*Branch:* ${{ github.ref }}\n*Commit:* ${{ github.sha }}"
            }
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
  if: always()
```

### External Quality Gates

```yaml
# SonarCloud integration:
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
  with:
    args: >
      -Dsonar.projectKey=terragon_modelcard-generator
      -Dsonar.organization=terragon
      -Dsonar.python.coverage.reportPaths=coverage.xml
```

## Workflow Optimization

### Performance Improvements

1. **Caching Strategies**
   ```yaml
   - name: Cache dependencies
     uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
       restore-keys: |
         ${{ runner.os }}-pip-
   ```

2. **Parallel Execution**
   ```yaml
   strategy:
     matrix:
       python-version: ['3.9', '3.10', '3.11', '3.12']
     max-parallel: 4  # Optimize for your runner limits
   ```

3. **Conditional Steps**
   ```yaml
   - name: Run expensive tests
     if: github.event_name == 'push' && github.ref == 'refs/heads/main'
     run: pytest tests/integration/
   ```

### Cost Optimization

1. **Runner Selection**
   ```yaml
   runs-on: ubuntu-latest  # Most cost-effective
   # Use ubuntu-latest-4-cores only when needed
   ```

2. **Workflow Triggers**
   ```yaml
   on:
     push:
       branches: [ main ]
       paths-ignore:
         - 'docs/**'
         - '*.md'
         - '.gitignore'
   ```

## Rollback Procedures

### Workflow Rollback Plan

1. **Identify Issue**
   - Monitor workflow success rates
   - Check error logs and notifications
   - Verify impact on deployments

2. **Quick Fixes**
   ```bash
   # Disable problematic workflow
   gh api repos/:owner/:repo/actions/workflows/ci.yml \
     --method PUT \
     --field state=disabled_manually
   
   # Revert to previous workflow version
   git revert <commit-hash-of-workflow-change>
   ```

3. **Emergency Procedures**
   - Have manual deployment process documented
   - Maintain emergency contact list
   - Document rollback decision criteria

## Documentation and Training

### Team Onboarding

1. **Workflow Documentation**
   - Maintain up-to-date README files
   - Document all customizations
   - Include troubleshooting guides

2. **Training Materials**
   - Create workflow overview presentation
   - Provide hands-on exercises
   - Maintain FAQ document

3. **Regular Reviews**
   - Monthly team workflow reviews
   - Quarterly security reviews
   - Annual complete workflow audit

This comprehensive guide ensures your team can successfully implement and maintain robust CI/CD workflows for the Model Card Generator project.