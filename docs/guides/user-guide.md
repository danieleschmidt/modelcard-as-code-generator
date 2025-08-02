# User Guide

## Overview

The Model Card as Code Generator automates the creation of standardized model documentation that satisfies regulatory requirements and enables drift detection in CI/CD pipelines.

## Core Concepts

### Model Cards
Model cards are standardized documents that provide essential information about machine learning models, including:
- Model details and intended use
- Training data and methodology
- Evaluation results and limitations
- Ethical considerations and bias analysis

### Formats Supported
- **Hugging Face**: Community standard format
- **Google Model Cards**: Structured schema approach
- **EU CRA**: Compliance-focused format for European regulations

### Source Integration
The generator can extract information from:
- Evaluation results (JSON, CSV, YAML)
- Training logs and configurations
- ML tracking platforms (MLflow, Weights & Biases)
- Version control metadata

## Command Line Interface

### Basic Commands

```bash
# Generate model card
mcg generate [sources] --format [format] --output [file]

# Validate existing card
mcg validate [card] --standard [standard]

# Check for drift
mcg check-drift [card] --against [new-data]

# Update existing card
mcg update [card] --with [new-data]
```

### Advanced Options

```bash
# Custom template
mcg generate eval.json --template custom.j2

# Multiple formats
mcg generate eval.json --format huggingface,google

# Compliance checking
mcg validate card.md --standard eu-cra --strict

# Auto-update mode
mcg watch results/ --auto-update card.md
```

## Python API

### Basic Usage

```python
from modelcard_generator import ModelCardGenerator

generator = ModelCardGenerator()
card = generator.generate(eval_results="eval.json")
card.save("MODEL_CARD.md")
```

### Advanced Configuration

```python
from modelcard_generator import CardConfig

config = CardConfig(
    format="huggingface",
    template="nlp_classification",
    include_bias_analysis=True,
    include_carbon_footprint=True,
    regulatory_standard="gdpr",
    auto_update=True
)

generator = ModelCardGenerator(config)
```

### Working with Cards

```python
# Load existing card
card = ModelCard.load("MODEL_CARD.md")

# Update metrics
card.update_metric("accuracy", 0.95)
card.add_limitation("Performance varies by language")

# Validate
validation = card.validate()
if not validation.is_valid:
    print(f"Issues: {validation.errors}")

# Export formats
card.export_json("card.json")
card.export_html("card.html")
```

## Templates

### Using Built-in Templates

```python
from modelcard_generator.templates import TemplateLibrary

# List available templates
templates = TemplateLibrary.list_templates()

# Use specific template
template = TemplateLibrary.get("computer_vision")
card = template.create(
    model_name="image-classifier",
    dataset="imagenet",
    accuracy=0.92
)
```

### Custom Templates

```python
from modelcard_generator import Template

class CustomTemplate(Template):
    def __init__(self):
        super().__init__(
            name="custom_template",
            required_sections=["overview", "metrics", "limitations"]
        )
    
    def render_overview(self, model_name, description):
        return f"# {model_name}\n\n{description}"

# Register template
TemplateLibrary.register(CustomTemplate())
```

## Validation and Compliance

### Schema Validation

```python
from modelcard_generator import Validator

validator = Validator()
result = validator.validate_schema(
    card_path="MODEL_CARD.md",
    schema="huggingface_v2"
)

if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.message}")
```

### Compliance Checking

```python
from modelcard_generator.compliance import ComplianceChecker

checker = ComplianceChecker()
result = checker.check(card, "eu_ai_act")

if result.compliant:
    print("✅ Compliant with EU AI Act")
else:
    print(f"❌ Missing: {result.missing_requirements}")
```

## Integration Patterns

### MLflow Integration

```python
from modelcard_generator.integrations import MLflowIntegration

mlflow_gen = MLflowIntegration(tracking_uri="...")
card = mlflow_gen.from_model("sentiment-classifier", version=2)
```

### Weights & Biases Integration

```python
from modelcard_generator.integrations import WandbIntegration

wandb_gen = WandbIntegration(api_key="...")
card = wandb_gen.from_run("project/run_id")
```

### Git Integration

```python
from modelcard_generator.integrations import GitIntegration

git_gen = GitIntegration()
card = git_gen.enhance_with_repo_info(card, repo_path=".")
```

## Best Practices

### 1. Automated Updates
Set up CI/CD to automatically update model cards:
```yaml
on:
  push:
    paths: ['results/**', 'config/**']
jobs:
  update-card:
    steps:
      - name: Update Model Card
        run: mcg update MODEL_CARD.md --auto
```

### 2. Drift Monitoring
Implement drift detection with appropriate thresholds:
```python
drift_config = {
    "accuracy": 0.02,     # 2% tolerance
    "f1_score": 0.03,     # 3% tolerance
    "latency_ms": 10      # 10ms tolerance
}
```

### 3. Version Control
Keep model cards in version control alongside code:
```
project/
├── src/
├── models/
├── results/
└── MODEL_CARD.md  # Version controlled
```

### 4. Compliance First
Always validate compliance before deployment:
```bash
mcg validate MODEL_CARD.md --standard eu-cra --fail-fast
```

## Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
pip install modelcard-as-code-generator[integrations]
```

**Template Not Found**
```python
# List available templates
from modelcard_generator.templates import TemplateLibrary
print(TemplateLibrary.list_templates())
```

**Validation Errors**
```bash
# Verbose validation
mcg validate MODEL_CARD.md --verbose
```

**Performance Issues**
```bash
# Process large files
mcg generate large_eval.json --chunk-size 1000
```

### Getting Help

- Check the [API Reference](../api-reference.md)
- Review [Examples](../examples/)
- File issues on [GitHub](https://github.com/terragonlabs/modelcard-as-code-generator/issues)
- Join our [Discord community](https://discord.gg/terragon)