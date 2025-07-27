#!/bin/bash

# Security Check Script for Model Card Generator
# Comprehensive security scanning and validation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/security-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create reports directory
mkdir -p "$REPORTS_DIR"

echo -e "${BLUE}🔒 Model Card Generator Security Check${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python security tools if needed
install_security_tools() {
    print_section "Installing Security Tools"
    
    echo "Installing Python security tools..."
    pip install --quiet bandit safety semgrep detect-secrets 2>/dev/null || {
        echo -e "${YELLOW}Warning: Some tools may not be available${NC}"
    }
    
    # Install additional tools if available
    if command_exists npm; then
        echo "Installing Node.js security tools..."
        npm install -g --silent audit-ci 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ Security tools installation completed${NC}"
    echo ""
}

# Function to run Bandit security scan
run_bandit_scan() {
    print_section "Bandit Static Security Analysis"
    
    if ! command_exists bandit; then
        echo -e "${YELLOW}⚠️ Bandit not available, skipping...${NC}"
        return 0
    fi
    
    echo "Running Bandit security scan..."
    
    bandit_output="$REPORTS_DIR/bandit_report_$TIMESTAMP.json"
    bandit_txt="$REPORTS_DIR/bandit_report_$TIMESTAMP.txt"
    
    if bandit -r src/ -f json -o "$bandit_output" 2>/dev/null; then
        echo -e "${GREEN}✓ Bandit scan completed successfully${NC}"
        bandit -r src/ -f txt > "$bandit_txt" 2>/dev/null || true
    else
        echo -e "${RED}❌ Bandit found security issues${NC}"
        bandit -r src/ -f txt > "$bandit_txt" 2>/dev/null || true
        echo "See report: $bandit_txt"
    fi
    
    # Show summary
    if [[ -f "$bandit_output" ]]; then
        high_issues=$(jq '.metrics._totals.SEVERITY.HIGH // 0' "$bandit_output" 2>/dev/null || echo "0")
        medium_issues=$(jq '.metrics._totals.SEVERITY.MEDIUM // 0' "$bandit_output" 2>/dev/null || echo "0")
        low_issues=$(jq '.metrics._totals.SEVERITY.LOW // 0' "$bandit_output" 2>/dev/null || echo "0")
        
        echo "  High severity: $high_issues"
        echo "  Medium severity: $medium_issues"
        echo "  Low severity: $low_issues"
    fi
    
    echo ""
}

# Function to run Safety dependency scan
run_safety_scan() {
    print_section "Safety Dependency Security Scan"
    
    if ! command_exists safety; then
        echo -e "${YELLOW}⚠️ Safety not available, skipping...${NC}"
        return 0
    fi
    
    echo "Running Safety dependency scan..."
    
    safety_output="$REPORTS_DIR/safety_report_$TIMESTAMP.json"
    safety_txt="$REPORTS_DIR/safety_report_$TIMESTAMP.txt"
    
    if safety check --json --output "$safety_output" 2>/dev/null; then
        echo -e "${GREEN}✓ Safety scan completed - no vulnerabilities found${NC}"
    else
        echo -e "${YELLOW}⚠️ Safety found vulnerabilities in dependencies${NC}"
        safety check --output "$safety_txt" 2>/dev/null || true
        echo "See report: $safety_txt"
    fi
    
    echo ""
}

# Function to run Semgrep scan
run_semgrep_scan() {
    print_section "Semgrep Security Analysis"
    
    if ! command_exists semgrep; then
        echo -e "${YELLOW}⚠️ Semgrep not available, skipping...${NC}"
        return 0
    fi
    
    echo "Running Semgrep security scan..."
    
    semgrep_output="$REPORTS_DIR/semgrep_report_$TIMESTAMP.json"
    
    if semgrep --config=auto --json --output="$semgrep_output" src/ 2>/dev/null; then
        echo -e "${GREEN}✓ Semgrep scan completed${NC}"
        
        # Show summary
        if [[ -f "$semgrep_output" ]]; then
            total_findings=$(jq '.results | length' "$semgrep_output" 2>/dev/null || echo "0")
            echo "  Total findings: $total_findings"
        fi
    else
        echo -e "${YELLOW}⚠️ Semgrep scan completed with findings${NC}"
    fi
    
    echo ""
}

# Function to run secret detection
run_secret_detection() {
    print_section "Secret Detection Scan"
    
    if ! command_exists detect-secrets; then
        echo -e "${YELLOW}⚠️ detect-secrets not available, skipping...${NC}"
        return 0
    fi
    
    echo "Running secret detection scan..."
    
    secrets_output="$REPORTS_DIR/secrets_report_$TIMESTAMP.json"
    
    # Run detect-secrets scan
    if detect-secrets scan --all-files --baseline .secrets.baseline 2>/dev/null > "$secrets_output"; then
        echo -e "${GREEN}✓ Secret detection completed${NC}"
    else
        echo -e "${YELLOW}⚠️ Potential secrets detected${NC}"
    fi
    
    # Check for new secrets
    if detect-secrets audit .secrets.baseline 2>/dev/null; then
        echo -e "${GREEN}✓ No new secrets detected${NC}"
    else
        echo -e "${YELLOW}⚠️ New potential secrets found - please review${NC}"
    fi
    
    echo ""
}

# Function to check file permissions
check_file_permissions() {
    print_section "File Permissions Check"
    
    echo "Checking file permissions..."
    
    # Check for files with overly permissive permissions
    suspicious_files=()
    
    # Find world-writable files
    while IFS= read -r -d '' file; do
        suspicious_files+=("$file")
    done < <(find "$PROJECT_ROOT" -type f -perm -002 -print0 2>/dev/null || true)
    
    # Find executable files that shouldn't be
    while IFS= read -r -d '' file; do
        if [[ "$file" == *.py ]] || [[ "$file" == *.json ]] || [[ "$file" == *.md ]]; then
            suspicious_files+=("$file")
        fi
    done < <(find "$PROJECT_ROOT" -type f -executable -print0 2>/dev/null || true)
    
    if [[ ${#suspicious_files[@]} -eq 0 ]]; then
        echo -e "${GREEN}✓ File permissions look good${NC}"
    else
        echo -e "${YELLOW}⚠️ Found files with suspicious permissions:${NC}"
        for file in "${suspicious_files[@]}"; do
            echo "  $file"
        done
    fi
    
    echo ""
}

# Function to check Docker security
check_docker_security() {
    print_section "Docker Security Check"
    
    if [[ ! -f "$PROJECT_ROOT/Dockerfile" ]]; then
        echo -e "${YELLOW}⚠️ No Dockerfile found, skipping...${NC}"
        return 0
    fi
    
    echo "Analyzing Dockerfile security..."
    
    docker_issues=()
    
    # Check for common security issues
    if grep -q "^USER root" "$PROJECT_ROOT/Dockerfile" 2>/dev/null; then
        docker_issues+=("Running as root user")
    fi
    
    if grep -q "RUN.*sudo" "$PROJECT_ROOT/Dockerfile" 2>/dev/null; then
        docker_issues+=("Using sudo in container")
    fi
    
    if grep -q "ADD.*http" "$PROJECT_ROOT/Dockerfile" 2>/dev/null; then
        docker_issues+=("Using ADD with URLs (prefer COPY)")
    fi
    
    if ! grep -q "USER.*[^r][^o][^o][^t]" "$PROJECT_ROOT/Dockerfile" 2>/dev/null; then
        docker_issues+=("Not explicitly setting non-root user")
    fi
    
    if [[ ${#docker_issues[@]} -eq 0 ]]; then
        echo -e "${GREEN}✓ Dockerfile security looks good${NC}"
    else
        echo -e "${YELLOW}⚠️ Dockerfile security issues found:${NC}"
        for issue in "${docker_issues[@]}"; do
            echo "  - $issue"
        done
    fi
    
    echo ""
}

# Function to check environment configuration
check_environment_config() {
    print_section "Environment Configuration Check"
    
    echo "Checking environment configuration..."
    
    config_issues=()
    
    # Check .env files
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        config_issues+=(".env file present - should not be committed")
    fi
    
    # Check for hardcoded secrets in config files
    for config_file in "$PROJECT_ROOT"/*.yml "$PROJECT_ROOT"/*.yaml "$PROJECT_ROOT"/*.json; do
        if [[ -f "$config_file" ]] && grep -qi "password\|secret\|key.*=" "$config_file" 2>/dev/null; then
            config_issues+=("Potential hardcoded credentials in $(basename "$config_file")")
        fi
    done
    
    if [[ ${#config_issues[@]} -eq 0 ]]; then
        echo -e "${GREEN}✓ Environment configuration looks secure${NC}"
    else
        echo -e "${YELLOW}⚠️ Configuration security issues:${NC}"
        for issue in "${config_issues[@]}"; do
            echo "  - $issue"
        done
    fi
    
    echo ""
}

# Function to check CI/CD security
check_cicd_security() {
    print_section "CI/CD Security Check"
    
    echo "Checking CI/CD configuration..."
    
    cicd_issues=()
    
    # Check GitHub Actions workflows
    if [[ -d "$PROJECT_ROOT/.github/workflows" ]]; then
        for workflow in "$PROJECT_ROOT/.github/workflows"/*.yml; do
            if [[ -f "$workflow" ]]; then
                # Check for potential security issues
                if grep -q "github.token" "$workflow" && ! grep -q "permissions:" "$workflow"; then
                    cicd_issues+=("Workflow $(basename "$workflow") uses GitHub token without explicit permissions")
                fi
                
                if grep -q "pull_request_target" "$workflow"; then
                    cicd_issues+=("Workflow $(basename "$workflow") uses pull_request_target (potential security risk)")
                fi
            fi
        done
    fi
    
    if [[ ${#cicd_issues[@]} -eq 0 ]]; then
        echo -e "${GREEN}✓ CI/CD configuration looks secure${NC}"
    else
        echo -e "${YELLOW}⚠️ CI/CD security issues:${NC}"
        for issue in "${cicd_issues[@]}"; do
            echo "  - $issue"
        done
    fi
    
    echo ""
}

# Function to generate security summary
generate_security_summary() {
    print_section "Security Summary Report"
    
    summary_file="$REPORTS_DIR/security_summary_$TIMESTAMP.md"
    
    cat > "$summary_file" << EOF
# Security Scan Summary

**Scan Date**: $(date)
**Project**: Model Card Generator
**Scan ID**: $TIMESTAMP

## Scan Results

EOF
    
    # Add results from each scan
    for report in "$REPORTS_DIR"/*_"$TIMESTAMP".*; do
        if [[ -f "$report" ]]; then
            echo "- $(basename "$report")" >> "$summary_file"
        fi
    done
    
    cat >> "$summary_file" << EOF

## Recommendations

1. **Regular Scans**: Run security scans on every commit
2. **Dependency Updates**: Keep dependencies updated regularly
3. **Secret Management**: Use environment variables or secret management systems
4. **Code Review**: Include security review in code review process
5. **Training**: Ensure team is trained on secure coding practices

## Next Steps

1. Review all security reports in the security-reports directory
2. Address any high or critical severity issues immediately
3. Plan remediation for medium severity issues
4. Update security baseline as needed
5. Schedule regular security reviews

EOF
    
    echo "Security summary report generated: $summary_file"
    echo ""
}

# Function to cleanup old reports
cleanup_old_reports() {
    echo "Cleaning up old security reports (keeping last 10)..."
    
    # Keep only the 10 most recent reports
    find "$REPORTS_DIR" -name "*_*.json" -o -name "*_*.txt" -o -name "*_*.md" | \
        sort -r | tail -n +31 | xargs rm -f 2>/dev/null || true
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
    echo ""
}

# Main execution
main() {
    echo "Starting security check at $(date)"
    echo "Project root: $PROJECT_ROOT"
    echo "Reports directory: $REPORTS_DIR"
    echo ""
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Install tools if needed
    install_security_tools
    
    # Run security scans
    run_bandit_scan
    run_safety_scan
    run_semgrep_scan
    run_secret_detection
    check_file_permissions
    check_docker_security
    check_environment_config
    check_cicd_security
    
    # Generate summary
    generate_security_summary
    
    # Cleanup old reports
    cleanup_old_reports
    
    print_section "Security Check Complete"
    echo -e "${GREEN}✓ Security check completed successfully${NC}"
    echo "Reports are available in: $REPORTS_DIR"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Review security reports for any issues"
    echo "2. Address high and critical severity findings"
    echo "3. Update security baseline if needed"
    echo "4. Consider adding security checks to CI/CD pipeline"
}

# Run main function
main "$@"