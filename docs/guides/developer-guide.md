# Developer Guide

## Development Setup

### Prerequisites
- Python 3.9+
- Git
- Docker (optional)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/terragonlabs/modelcard-as-code-generator
cd modelcard-as-code-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Development Workflow

```bash
# Run tests
pytest

# Run linting
ruff check src/ tests/
black --check src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/

# Build documentation
mkdocs serve
```

## Architecture Overview

### Core Components

```
src/modelcard_generator/
├── core/           # Core generation logic
├── templates/      # Template engine and built-ins
├── validators/     # Validation and compliance
├── integrations/   # External platform integrations
├── cli/           # Command-line interface
└── api/           # Python API
```

### Key Classes

```python
# Core generation
class ModelCardGenerator:
    def generate(self, **sources) -> ModelCard
    def validate(self, card: ModelCard) -> ValidationResult

# Model card representation
class ModelCard:
    def add_section(self, name: str, content: str)
    def update_metric(self, name: str, value: float)
    def render(self, format: str) -> str

# Template system
class Template:
    def render(self, **context) -> str
    def validate_context(self, **context) -> bool
```

## Extending the Generator

### Custom Templates

Create a new template:

```python
from modelcard_generator.templates import BaseTemplate

class BiometricTemplate(BaseTemplate):
    name = "biometric_model"
    required_sections = [
        "privacy_protection",
        "consent_mechanism", 
        "data_retention"
    ]
    
    def render_privacy_protection(self, measures):
        return f"""
## Privacy Protection

The following measures protect biometric data:
{self._format_list(measures)}

### Data Minimization
Only essential biometric features are collected and processed.

### Encryption
All biometric data is encrypted at rest and in transit using AES-256.
"""

    def render_consent_mechanism(self, consent_process):
        return f"""
## Consent Mechanism

{consent_process}

### Withdrawal Process
Users can withdraw consent at any time through:
- Account settings dashboard
- Email request to privacy@company.com
- Physical form submission
"""
```

Register the template:

```python
from modelcard_generator.templates import TemplateRegistry

TemplateRegistry.register(BiometricTemplate)
```

### Custom Validators

Create domain-specific validation:

```python
from modelcard_generator.validators import BaseValidator

class BiometricValidator(BaseValidator):
    def validate(self, card: ModelCard) -> ValidationResult:
        errors = []
        
        # Check for required privacy sections
        if not card.has_section("privacy_protection"):
            errors.append("Missing privacy protection section")
            
        # Validate biometric-specific metrics
        if "false_acceptance_rate" not in card.metrics:
            errors.append("Missing FAR metric")
            
        if "false_rejection_rate" not in card.metrics:
            errors.append("Missing FRR metric")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
```

### Custom Integrations

Add support for new platforms:

```python
from modelcard_generator.integrations import BaseIntegration

class CustomMLPlatformIntegration(BaseIntegration):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        
    def extract_model_info(self, model_id: str) -> dict:
        # Fetch model metadata
        response = requests.get(
            f"{self.base_url}/models/{model_id}",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
        
    def extract_metrics(self, experiment_id: str) -> dict:
        # Fetch evaluation metrics
        response = requests.get(
            f"{self.base_url}/experiments/{experiment_id}/metrics",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
        
    def generate_card(self, model_id: str, experiment_id: str) -> ModelCard:
        model_info = self.extract_model_info(model_id)
        metrics = self.extract_metrics(experiment_id)
        
        card = ModelCard()
        card.add_section("model_details", self._format_model_details(model_info))
        card.add_metrics(metrics)
        
        return card
```

## Testing

### Unit Tests

```python
import pytest
from modelcard_generator import ModelCardGenerator

def test_basic_generation():
    generator = ModelCardGenerator()
    card = generator.generate(
        eval_results={"accuracy": 0.95, "f1": 0.92}
    )
    
    assert "accuracy" in card.metrics
    assert card.metrics["accuracy"] == 0.95

def test_template_rendering():
    from modelcard_generator.templates import HuggingFaceTemplate
    
    template = HuggingFaceTemplate()
    content = template.render(
        model_name="test-model",
        accuracy=0.95
    )
    
    assert "test-model" in content
    assert "95%" in content
```

### Integration Tests

```python
def test_mlflow_integration():
    from modelcard_generator.integrations import MLflowIntegration
    
    integration = MLflowIntegration(tracking_uri="sqlite:///test.db")
    card = integration.from_model("test_model", version=1)
    
    assert card.model_name == "test_model"
    assert len(card.metrics) > 0
```

### Performance Tests

```python
import time
import pytest

def test_large_file_performance():
    # Generate large evaluation file
    large_eval = {"metrics": {f"metric_{i}": i * 0.01 for i in range(10000)}}
    
    start_time = time.time()
    generator = ModelCardGenerator()
    card = generator.generate(eval_results=large_eval)
    end_time = time.time()
    
    # Should complete within 30 seconds
    assert end_time - start_time < 30
```

## Documentation

### API Documentation

Use docstrings for all public APIs:

```python
def generate_card(
    self,
    eval_results: Dict[str, Any],
    format: str = "huggingface",
    template: Optional[str] = None
) -> ModelCard:
    """Generate a model card from evaluation results.
    
    Args:
        eval_results: Dictionary containing evaluation metrics and metadata
        format: Output format ('huggingface', 'google', 'eu-cra')
        template: Optional template name to use
        
    Returns:
        Generated ModelCard instance
        
    Raises:
        ValidationError: If evaluation results are invalid
        TemplateError: If template rendering fails
        
    Example:
        >>> generator = ModelCardGenerator()
        >>> card = generator.generate_card(
        ...     eval_results={"accuracy": 0.95},
        ...     format="huggingface"
        ... )
        >>> print(card.render())
    """
```

### Adding Examples

Create comprehensive examples:

```python
# examples/medical_ai_example.py
from modelcard_generator import ModelCardGenerator
from modelcard_generator.templates import MedicalAITemplate

def generate_medical_model_card():
    """Example: Generating a medical AI model card."""
    
    # Evaluation results from clinical validation
    eval_results = {
        "sensitivity": 0.95,
        "specificity": 0.98,
        "auc": 0.97,
        "patient_cohort_size": 1000,
        "validation_sites": ["Hospital A", "Hospital B", "Hospital C"]
    }
    
    # Clinical metadata
    clinical_info = {
        "intended_population": "Adults 18-65 with chest symptoms",
        "contraindications": [
            "Pediatric patients",
            "Pregnancy",
            "Severe obesity (BMI > 40)"
        ],
        "regulatory_status": "FDA 510(k) pending",
        "clinical_validation": {
            "study_design": "Multi-center retrospective study",
            "primary_endpoint": "Diagnostic accuracy vs radiologist consensus",
            "statistical_power": 0.8
        }
    }
    
    # Generate card using medical template
    generator = ModelCardGenerator()
    card = generator.generate(
        eval_results=eval_results,
        clinical_info=clinical_info,
        template="medical_ai",
        format="eu_cra"  # EU Medical Device Regulation compliance
    )
    
    # Add regulatory compliance sections
    card.add_section("risk_classification", "Class IIa Medical Device")
    card.add_section("quality_management", "ISO 13485 certified")
    
    # Validate compliance
    validation = card.validate_compliance("medical_device_regulation")
    if not validation.is_compliant:
        print(f"Compliance issues: {validation.missing_requirements}")
    
    # Save card
    card.save("MEDICAL_MODEL_CARD.md")
    card.export_pdf("medical_compliance_report.pdf")

if __name__ == "__main__":
    generate_medical_model_card()
```

## Contributing Guidelines

### Code Style

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 88 characters
- Use descriptive variable names

### Commit Messages

Follow conventional commits:

```
feat: add biometric model template
fix: resolve drift detection false positives
docs: update API reference for new validators
test: add integration tests for MLflow
refactor: simplify template rendering logic
```

### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit PR with clear description

### Code Review Checklist

- [ ] Tests pass and coverage maintained
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Security considerations addressed
- [ ] Performance impact assessed
- [ ] Backward compatibility maintained

## Release Process

### Version Management

Use semantic versioning:
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build and test package
5. Create release tag
6. Publish to PyPI
7. Update documentation

```bash
# Build package
python -m build

# Test package
python -m twine check dist/*

# Upload to test PyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Debugging

### Common Issues

**Template rendering errors:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check template context
template.debug_context(**context)
```

**Validation failures:**
```python
# Verbose validation
result = validator.validate(card, verbose=True)
for error in result.errors:
    print(f"Line {error.line}: {error.message}")
```

**Integration issues:**
```python
# Test connection
integration.test_connection()

# Check credentials
integration.validate_credentials()
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_generation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    generator = ModelCardGenerator()
    card = generator.generate(large_eval_results)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

profile_generation()
```