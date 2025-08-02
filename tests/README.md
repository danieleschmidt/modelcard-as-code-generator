# Testing Guide

## Overview

The Model Card Generator uses a comprehensive testing strategy to ensure reliability, security, and performance across all features.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for component interaction
├── security/          # Security-focused tests
├── performance/       # Performance and benchmarking tests
├── fixtures/          # Test data and mock files
├── conftest.py        # Pytest configuration and shared fixtures
└── README.md          # This file
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/security/                # Security tests only
pytest tests/performance/             # Performance tests only

# Run tests with coverage
pytest --cov=src/modelcard_generator --cov-report=html

# Run tests with detailed output
pytest -v

# Run specific test file
pytest tests/unit/test_generator.py

# Run specific test method
pytest tests/unit/test_generator.py::test_basic_generation
```

### Test Markers

Tests are organized using pytest markers:

```bash
# Run by markers
pytest -m unit                       # Unit tests
pytest -m integration                # Integration tests
pytest -m slow                       # Slow tests
pytest -m "not slow"                 # Fast tests only
pytest -m network                    # Tests requiring network
pytest -m security                   # Security tests
pytest -m performance                # Performance benchmarks
pytest -m smoke                      # Quick smoke tests
```

### Parallel Testing

```bash
# Run tests in parallel
pytest -n auto                       # Auto-detect CPU cores
pytest -n 4                          # Use 4 processes
```

### Test Configuration

Tests can be configured via environment variables:

```bash
# Test environment
export MCG_ENVIRONMENT=test
export MCG_LOG_LEVEL=DEBUG

# Enable integration tests
export ENABLE_INTEGRATION_TESTS=true

# Network timeouts
export MCG_REQUEST_TIMEOUT=5

# Test data location
export MCG_TEST_DATA_DIR=./tests/fixtures
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **Generator Tests**: Core model card generation logic
- **Template Tests**: Template rendering and validation
- **Validator Tests**: Schema and content validation
- **CLI Tests**: Command-line interface functionality
- **Utility Tests**: Helper functions and utilities

Example:
```python
def test_basic_generation(sample_eval_results):
    generator = ModelCardGenerator()
    card = generator.generate(eval_results=sample_eval_results)
    assert card.model_name == "test-sentiment-classifier"
    assert card.metrics["accuracy"] == 0.92
```

### Integration Tests (`tests/integration/`)

Test component interactions and end-to-end workflows:

- **CLI Integration**: Full command-line workflows
- **Template Integration**: Multi-format generation
- **Validation Integration**: Cross-validator testing
- **File I/O Integration**: Reading/writing various formats

Example:
```python
def test_cli_generation_workflow(temp_dir, eval_results_file):
    result = runner.invoke(cli, [
        'generate', str(eval_results_file),
        '--format', 'huggingface',
        '--output', str(temp_dir / 'card.md')
    ])
    assert result.exit_code == 0
    assert (temp_dir / 'card.md').exists()
```

### Security Tests (`tests/security/`)

Validate security measures and compliance:

- **Secret Detection**: Ensure no secrets in generated cards
- **Input Validation**: Test against malicious inputs
- **Access Control**: Validate file permissions and access
- **Vulnerability Scanning**: Check for known vulnerabilities

Example:
```python
def test_secret_detection():
    card_content = generate_card_with_secrets()
    detector = SecretDetector()
    secrets = detector.scan(card_content)
    assert len(secrets) == 0, f"Secrets detected: {secrets}"
```

### Performance Tests (`tests/performance/`)

Benchmark performance and resource usage:

- **Generation Speed**: Time to generate various card types
- **Memory Usage**: Memory consumption during generation
- **Large File Handling**: Performance with large datasets
- **Concurrent Processing**: Multi-threaded performance

Example:
```python
@pytest.mark.performance
def test_large_file_generation(large_eval_results):
    start_time = time.time()
    generator = ModelCardGenerator()
    card = generator.generate(eval_results=large_eval_results)
    duration = time.time() - start_time
    assert duration < 30, f"Generation took {duration}s, expected < 30s"
```

## Fixtures and Test Data

### Available Fixtures

- `temp_dir`: Temporary directory for test files
- `sample_eval_results`: Standard evaluation results
- `sample_training_config`: Training configuration data
- `sample_huggingface_card`: Pre-generated model card
- `eval_results_file`: Temporary JSON file with eval data
- `config_file`: Temporary YAML configuration file
- `model_card_file`: Temporary model card file
- `mock_env_vars`: Mocked environment variables
- `mock_requests`: Mocked HTTP requests

### Creating Custom Test Data

```python
# Use test utilities
from tests.conftest import create_test_file, assert_valid_json

def test_custom_data(temp_dir):
    data = {"custom": "test data"}
    file_path = create_test_file(temp_dir, "test.json", json.dumps(data))
    
    with open(file_path) as f:
        content = f.read()
    
    parsed = assert_valid_json(content)
    assert parsed["custom"] == "test data"
```

## Mocking and Test Doubles

### External API Mocking

```python
def test_mlflow_integration(mock_requests):
    # mock_requests fixture automatically mocks common APIs
    integration = MLflowIntegration()
    experiment = integration.get_experiment("test")
    assert experiment["experiment_id"] == "1"
```

### Custom Mocking

```python
from unittest.mock import Mock, patch

@patch('modelcard_generator.integrations.wandb.Api')
def test_wandb_integration(mock_wandb_api):
    mock_wandb_api.return_value.runs.return_value = [Mock(config={}, summary={})]
    
    integration = WandbIntegration()
    runs = integration.get_runs("project")
    assert len(runs) == 1
```

## Property-Based Testing

Using Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st

@given(
    accuracy=st.floats(min_value=0.0, max_value=1.0),
    model_name=st.text(min_size=1, max_size=50)
)
def test_card_generation_properties(accuracy, model_name):
    eval_results = {"accuracy": accuracy, "model_name": model_name}
    generator = ModelCardGenerator()
    card = generator.generate(eval_results=eval_results)
    
    # Properties that should always hold
    assert card.metrics["accuracy"] == accuracy
    assert card.model_name == model_name
    assert 0 <= card.metrics["accuracy"] <= 1
```

## Test Data Management

### Fixtures Directory

```
tests/fixtures/
├── eval_results/
│   ├── basic.json
│   ├── large.json
│   └── malformed.json
├── configs/
│   ├── training.yaml
│   └── production.yaml
├── cards/
│   ├── huggingface.md
│   ├── google.json
│   └── eu_cra.md
└── schemas/
    ├── huggingface_v1.json
    └── google_v1.json
```

### Loading Test Data

```python
import json
from pathlib import Path

def load_fixture(name: str) -> dict:
    """Load a test fixture by name."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    with open(fixtures_dir / f"{name}.json") as f:
        return json.load(f)

def test_with_fixture():
    data = load_fixture("eval_results/basic")
    assert "accuracy" in data
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Daily scheduled runs

Configuration in `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[test]"
      
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Testing

Tests run on every commit via pre-commit hooks:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest-check
      name: pytest-check
      entry: pytest tests/unit/ -x
      language: system
      pass_filenames: false
      always_run: true
```

## Test Best Practices

### Writing Good Tests

1. **Test One Thing**: Each test should verify one specific behavior
2. **Clear Names**: Test names should describe what is being tested
3. **Independent**: Tests should not depend on each other
4. **Fast**: Unit tests should run quickly
5. **Reliable**: Tests should pass consistently

### Example Good Test

```python
def test_huggingface_card_generation_includes_required_sections():
    """Test that Hugging Face format includes all required sections."""
    # Arrange
    eval_results = {"accuracy": 0.95, "model_name": "test-model"}
    generator = ModelCardGenerator()
    
    # Act
    card = generator.generate(eval_results=eval_results, format="huggingface")
    content = card.render()
    
    # Assert
    required_sections = ["# ", "## Model Description", "## Training Data"]
    for section in required_sections:
        assert section in content, f"Missing required section: {section}"
```

### Test Organization

```python
class TestModelCardGenerator:
    """Group related tests in classes."""
    
    def test_basic_generation(self):
        """Test basic model card generation."""
        pass
    
    def test_custom_template(self):
        """Test generation with custom template."""
        pass
    
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test generation with large dataset."""
        pass
```

## Debugging Tests

### Running Individual Tests

```bash
# Run with pdb debugger
pytest --pdb tests/unit/test_generator.py::test_basic_generation

# Run with detailed output
pytest -vvv tests/unit/test_generator.py

# Run with print statements
pytest -s tests/unit/test_generator.py
```

### Test Debugging Tips

1. Use `pytest.set_trace()` for debugging
2. Add `print()` statements and run with `-s`
3. Use `--tb=short` for shorter tracebacks
4. Use `--tb=long` for detailed tracebacks
5. Use `--lf` to run only failed tests from last run

## Coverage Requirements

- **Unit Tests**: Minimum 90% coverage
- **Integration Tests**: Cover all major workflows
- **Security Tests**: Cover all security-critical paths
- **Performance Tests**: Baseline performance metrics

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html

# Generate coverage summary
pytest --cov=src --cov-report=term-missing
```

## Contributing to Tests

### Adding New Tests

1. Choose appropriate test category (unit/integration/security/performance)
2. Use existing fixtures when possible
3. Follow naming conventions: `test_[what]_[when]_[expected]`
4. Add appropriate markers (`@pytest.mark.unit`, etc.)
5. Update this documentation if adding new patterns

### Test Review Checklist

- [ ] Tests cover new functionality
- [ ] Tests follow naming conventions
- [ ] Appropriate markers are used
- [ ] Tests are independent and reliable
- [ ] Performance impact is acceptable
- [ ] Security considerations are addressed
- [ ] Documentation is updated if needed