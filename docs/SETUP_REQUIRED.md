# Manual Setup Requirements

## Overview

Due to GitHub App permission limitations, several components require manual setup by repository administrators. This document provides a comprehensive checklist and instructions.

## Required Manual Actions

### 🔧 1. GitHub Actions Workflows

**Status**: ⚠️ **REQUIRED - Manual Setup**

**Actions Needed**:
1. Copy workflow templates from `docs/github-workflows-templates/` to `.github/workflows/`
2. Configure repository secrets
3. Set up branch protection rules
4. Test workflow execution

**Templates Available**:
- `ci.yml` - Continuous integration testing
- `security.yml` - Security scanning and vulnerability checks  
- `release.yml` - Automated releases and PyPI publishing
- `dependabot.yml` - Dependency update automation
- `performance.yml` - Performance testing and benchmarks
- `chaos-engineering.yml` - Chaos engineering tests

**Detailed Instructions**: See [docs/workflows/WORKFLOW_SETUP_GUIDE.md](workflows/WORKFLOW_SETUP_GUIDE.md)

### 🔐 2. Repository Secrets Configuration

**Status**: ⚠️ **REQUIRED - Manual Setup**

**Required Secrets**:
```bash
# PyPI Publishing (Required for releases)
PYPI_API_TOKEN=pypi-your-token-here

# Docker Registry (Optional, for container publishing)
DOCKER_USERNAME=your-docker-username  
DOCKER_PASSWORD=your-docker-password

# Notifications (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Security Scanning (Optional)
SONAR_TOKEN=your-sonar-token
SNYK_TOKEN=your-snyk-token
```

**Setup Instructions**:
1. Navigate to: Repository → Settings → Secrets and variables → Actions
2. Click "New repository secret" for each required secret
3. Test secret access in workflow runs

### 🛡️ 3. Branch Protection Rules

**Status**: ⚠️ **REQUIRED - Manual Setup**

**Required Protections for `main` branch**:
- ✅ Require pull request reviews (minimum 1)
- ✅ Dismiss stale reviews when new commits are pushed
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators in restrictions

**Required Status Checks**:
- `ci / test-python-3.9`
- `ci / test-python-3.10`
- `ci / test-python-3.11` 
- `ci / test-python-3.12`
- `security / security-scan`
- `security / dependency-scan`

**Setup Instructions**:
1. Navigate to: Repository → Settings → Branches
2. Click "Add rule" for main branch
3. Configure protection settings as listed above
4. Test with a test pull request

## Quick Setup Checklist

### Essential Setup (Required for Core Functionality)
- [ ] Copy workflow templates to `.github/workflows/`
- [ ] Configure `PYPI_API_TOKEN` secret for releases
- [ ] Set up branch protection rules for main branch
- [ ] Test CI workflow with a pull request
- [ ] Verify security scanning is working

### Enhanced Setup (Recommended for Production)
- [ ] Configure Docker registry credentials
- [ ] Set up Slack/email notifications
- [ ] Enable SonarCloud or Snyk scanning
- [ ] Configure monitoring and alerting
- [ ] Set up deployment automation

## Success Criteria

Setup is considered complete when:
- [ ] All workflows execute successfully on pull requests
- [ ] Security scanning detects and reports vulnerabilities
- [ ] Releases are automatically published to PyPI
- [ ] Branch protection prevents direct pushes to main
- [ ] Team members can create and merge pull requests
- [ ] Monitoring shows green status for all services

**Timeline Estimate**: 2-4 hours for essential setup, 1-2 days for full production setup

For detailed setup instructions, see [docs/workflows/WORKFLOW_SETUP_GUIDE.md](workflows/WORKFLOW_SETUP_GUIDE.md)