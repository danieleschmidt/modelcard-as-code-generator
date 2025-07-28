#!/bin/bash

# Quality check script for Model Card Generator
# This script runs comprehensive code quality checks

set -e

echo "🔍 Starting comprehensive quality checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[QUALITY]${NC} $1"
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

# Initialize counters
ERRORS=0
WARNINGS=0

# Function to increment error counter
increment_errors() {
    ERRORS=$((ERRORS + 1))
}

# Function to increment warning counter
increment_warnings() {
    WARNINGS=$((WARNINGS + 1))
}

# Create reports directory
mkdir -p reports/quality

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 1. Code Formatting Check
print_status "Checking code formatting with Black..."
if black --check --diff src/ tests/ > reports/quality/black-check.txt 2>&1; then
    print_success "Code formatting check passed"
else
    print_error "Code formatting issues found"
    echo "Run 'black src/ tests/' to fix formatting issues"
    increment_errors
fi

# 2. Import Sorting Check
print_status "Checking import sorting with isort..."
if isort --check-only --diff src/ tests/ > reports/quality/isort-check.txt 2>&1; then
    print_success "Import sorting check passed"
else
    print_error "Import sorting issues found"
    echo "Run 'isort src/ tests/' to fix import sorting"
    increment_errors
fi

# 3. Linting with Ruff
print_status "Running comprehensive linting with Ruff..."
if ruff check src/ tests/ --output-format=json > reports/quality/ruff-report.json 2>&1; then
    print_success "Ruff linting passed"
else
    print_warning "Linting issues found (see reports/quality/ruff-report.json)"
    increment_warnings
fi

# 4. Type Checking with MyPy
print_status "Running type checking with MyPy..."
if mypy src/ --xml-report reports/quality/ > reports/quality/mypy-report.txt 2>&1; then
    print_success "Type checking passed"
else
    print_warning "Type checking issues found (see reports/quality/mypy-report.txt)"
    increment_warnings
fi

# 5. Security Analysis with Bandit
print_status "Running security analysis with Bandit..."
if bandit -r src/ -f json -o reports/quality/bandit-report.json > /dev/null 2>&1; then
    print_success "Security analysis passed"
else
    SECURITY_ISSUES=$(jq '.results | length' reports/quality/bandit-report.json 2>/dev/null || echo "0")
    if [ "$SECURITY_ISSUES" -gt 0 ]; then
        print_warning "$SECURITY_ISSUES security issues found (see reports/quality/bandit-report.json)"
        increment_warnings
    else
        print_success "Security analysis passed"
    fi
fi

# 6. Dependency Security Check
print_status "Checking dependency security with Safety..."
if safety check --json --output reports/quality/safety-report.json > /dev/null 2>&1; then
    print_success "Dependency security check passed"
else
    print_warning "Vulnerable dependencies found (see reports/quality/safety-report.json)"
    increment_warnings
fi

# 7. Code Complexity Analysis
print_status "Analyzing code complexity..."
if command -v radon &> /dev/null; then
    radon cc src/ -j > reports/quality/complexity-report.json
    
    # Check for high complexity functions
    if python3 -c "
import json
with open('reports/quality/complexity-report.json') as f:
    data = json.load(f)
    high_complexity = []
    for file_data in data.values():
        for item in file_data:
            if item['complexity'] > 10:
                high_complexity.append(f\"{item['name']} (complexity: {item['complexity']})\")
    if high_complexity:
        print('High complexity functions found:')
        for func in high_complexity[:5]:  # Show first 5
            print(f'  - {func}')
        exit(1)
" 2> /dev/null; then
        print_success "Code complexity check passed"
    else
        print_warning "High complexity functions found (see reports/quality/complexity-report.json)"
        increment_warnings
    fi
else
    print_warning "Radon not available - skipping complexity analysis"
fi

# 8. Documentation Quality Check
print_status "Checking documentation quality..."
if command -v pydocstyle &> /dev/null; then
    if pydocstyle src/ --convention=google > reports/quality/pydocstyle-report.txt 2>&1; then
        print_success "Documentation style check passed"
    else
        DOC_ISSUES=$(wc -l < reports/quality/pydocstyle-report.txt)
        print_warning "$DOC_ISSUES documentation style issues found"
        increment_warnings
    fi
else
    print_warning "Pydocstyle not available - skipping documentation check"
fi

# 9. Test Coverage Analysis
print_status "Analyzing test coverage..."
if command -v pytest &> /dev/null; then
    if pytest --cov=src/modelcard_generator --cov-report=json:reports/quality/coverage.json --cov-report=term > reports/quality/coverage-report.txt 2>&1; then
        COVERAGE=$(jq '.totals.percent_covered' reports/quality/coverage.json 2>/dev/null || echo "0")
        if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
            print_success "Test coverage check passed ($COVERAGE%)"
        else
            print_warning "Test coverage below threshold: $COVERAGE% < 80%"
            increment_warnings
        fi
    else
        print_warning "Coverage analysis failed"
        increment_warnings
    fi
else
    print_warning "Pytest not available - skipping coverage analysis"
fi

# 10. Dependency Analysis
print_status "Analyzing dependencies..."
if command -v pipdeptree &> /dev/null; then
    pipdeptree --json > reports/quality/dependencies.json
    
    # Check for outdated packages
    if pip list --outdated --format=json > reports/quality/outdated-packages.json 2>/dev/null; then
        OUTDATED_COUNT=$(jq '. | length' reports/quality/outdated-packages.json)
        if [ "$OUTDATED_COUNT" -gt 0 ]; then
            print_warning "$OUTDATED_COUNT outdated packages found"
            increment_warnings
        else
            print_success "All packages are up to date"
        fi
    fi
else
    print_warning "Pipdeptree not available - skipping dependency analysis"
fi

# 11. License Compliance Check
print_status "Checking license compliance..."
if command -v pip-licenses &> /dev/null; then
    pip-licenses --format=json --output-file=reports/quality/licenses.json
    
    # Check for problematic licenses
    if python3 -c "
import json
with open('reports/quality/licenses.json') as f:
    licenses = json.load(f)
    problematic = ['GPL', 'AGPL', 'LGPL']
    issues = []
    for pkg in licenses:
        license_name = pkg.get('License', '')
        for prob in problematic:
            if prob in license_name.upper():
                issues.append(f\"{pkg['Name']}: {license_name}\")
    if issues:
        print('Problematic licenses found:')
        for issue in issues:
            print(f'  - {issue}')
        exit(1)
" 2> /dev/null; then
        print_success "License compliance check passed"
    else
        print_warning "Problematic licenses found (see reports/quality/licenses.json)"
        increment_warnings
    fi
else
    print_warning "Pip-licenses not available - skipping license check"
fi

# 12. Code Duplication Analysis
print_status "Checking for code duplication..."
if command -v vulture &> /dev/null; then
    vulture src/ --min-confidence 80 > reports/quality/dead-code.txt 2>&1 || true
    DEAD_CODE_LINES=$(wc -l < reports/quality/dead-code.txt)
    
    if [ "$DEAD_CODE_LINES" -gt 0 ]; then
        print_warning "Potential dead code found (see reports/quality/dead-code.txt)"
        increment_warnings
    else
        print_success "No dead code detected"
    fi
else
    print_warning "Vulture not available - skipping dead code analysis"
fi

# 13. Performance Analysis
print_status "Running performance analysis..."
if [ -d "tests/performance" ]; then
    if pytest tests/performance/ --benchmark-only --benchmark-json=reports/quality/benchmark.json > /dev/null 2>&1; then
        print_success "Performance benchmarks completed"
    else
        print_warning "Performance benchmarks failed"
        increment_warnings
    fi
else
    print_warning "No performance tests found - skipping performance analysis"
fi

# 14. Documentation Build Test
print_status "Testing documentation build..."
if [ -f "mkdocs.yml" ]; then
    if mkdocs build --strict > reports/quality/docs-build.log 2>&1; then
        print_success "Documentation build test passed"
    else
        print_error "Documentation build failed"
        increment_errors
    fi
else
    print_warning "No mkdocs.yml found - skipping documentation build test"
fi

# 15. Package Metadata Validation
print_status "Validating package metadata..."
if python3 -c "
import toml
with open('pyproject.toml') as f:
    data = toml.load(f)
    project = data.get('project', {})
    required_fields = ['name', 'version', 'description', 'authors', 'license']
    missing = [field for field in required_fields if not project.get(field)]
    if missing:
        print(f'Missing required fields: {missing}')
        exit(1)
" 2> /dev/null; then
    print_success "Package metadata validation passed"
else
    print_error "Package metadata validation failed"
    increment_errors
fi

# Generate Quality Report
print_status "Generating quality report..."
cat > reports/quality/quality-summary.md << EOF
# Code Quality Report

Generated: $(date)
Project: Model Card Generator
Branch: $(git branch --show-current 2>/dev/null || echo "unknown")
Commit: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

## Summary

- **Errors**: $ERRORS
- **Warnings**: $WARNINGS
- **Overall Status**: $([ $ERRORS -eq 0 ] && echo "PASS" || echo "FAIL")

## Quality Metrics

### Code Quality
- Code Formatting (Black): $([ -f reports/quality/black-check.txt ] && echo "✓ PASS" || echo "❌ FAIL")
- Import Sorting (isort): $([ -f reports/quality/isort-check.txt ] && echo "✓ PASS" || echo "❌ FAIL")
- Linting (Ruff): $([ -f reports/quality/ruff-report.json ] && echo "⚠️ CHECK REPORT" || echo "❌ FAIL")
- Type Checking (MyPy): $([ -f reports/quality/mypy-report.txt ] && echo "⚠️ CHECK REPORT" || echo "❌ FAIL")

### Security
- Security Analysis (Bandit): $([ -f reports/quality/bandit-report.json ] && echo "⚠️ CHECK REPORT" || echo "❌ FAIL")
- Dependency Security (Safety): $([ -f reports/quality/safety-report.json ] && echo "⚠️ CHECK REPORT" || echo "❌ FAIL")

### Testing
- Test Coverage: $([ -f reports/quality/coverage.json ] && jq -r '.totals.percent_covered_display' reports/quality/coverage.json || echo "Unknown")%

### Dependencies
- Outdated Packages: $([ -f reports/quality/outdated-packages.json ] && jq '. | length' reports/quality/outdated-packages.json || echo "Unknown")
- License Compliance: $([ -f reports/quality/licenses.json ] && echo "✓ CHECKED" || echo "❌ NOT CHECKED")

### Documentation
- Documentation Style: $([ -f reports/quality/pydocstyle-report.txt ] && echo "⚠️ CHECK REPORT" || echo "✓ PASS")
- Documentation Build: $([ -f reports/quality/docs-build.log ] && echo "✓ PASS" || echo "❌ FAIL")

## Detailed Reports

$([ -f reports/quality/ruff-report.json ] && echo "- [Linting Report](ruff-report.json)")
$([ -f reports/quality/mypy-report.txt ] && echo "- [Type Checking Report](mypy-report.txt)")
$([ -f reports/quality/bandit-report.json ] && echo "- [Security Report](bandit-report.json)")
$([ -f reports/quality/safety-report.json ] && echo "- [Dependency Security Report](safety-report.json)")
$([ -f reports/quality/coverage.json ] && echo "- [Coverage Report](coverage.json)")
$([ -f reports/quality/complexity-report.json ] && echo "- [Complexity Report](complexity-report.json)")
$([ -f reports/quality/licenses.json ] && echo "- [License Report](licenses.json)")

## Recommendations

$([ $ERRORS -gt 0 ] && echo "❌ **Critical Issues**: Fix $ERRORS error(s) before proceeding")
$([ $WARNINGS -gt 0 ] && echo "⚠️ **Improvements**: Address $WARNINGS warning(s) for better code quality")
$([ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ] && echo "✅ **Excellent**: No issues found - code quality is excellent!")

EOF

# Display summary
echo ""
print_status "Quality check completed!"
echo ""
echo "Results:"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        print_success "Perfect! No issues found. 🎉"
    else
        print_warning "Good! No errors, but $WARNINGS warnings to review."
    fi
    echo ""
    echo "Quality report: reports/quality/quality-summary.md"
    exit 0
else
    print_error "Quality check failed with $ERRORS errors and $WARNINGS warnings."
    echo ""
    echo "Quality report: reports/quality/quality-summary.md"
    exit 1
fi
