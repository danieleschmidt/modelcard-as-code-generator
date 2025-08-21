# Model Card Generator API Reference

## Table of Contents

- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Generation Methods](#generation-methods)
- [Validation](#validation)
- [Formats](#formats)
- [CLI Interface](#cli-interface)
- [Internationalization](#internationalization)

## Core Classes

### ModelCardGenerator

The main class for generating model cards.

```python
from modelcard_generator import ModelCardGenerator, CardConfig, CardFormat

# Basic usage
generator = ModelCardGenerator()
card = generator.generate(eval_results="results.json")

# With configuration
config = CardConfig(
    format=CardFormat.HUGGINGFACE,
    include_ethical_considerations=True,
    include_bias_analysis=True
)
generator = ModelCardGenerator(config)
```

#### Methods

##### `generate()`

Generate a model card from various sources.

**Parameters:**
- `eval_results` (str|Path|dict, optional): Evaluation results file or data
- `training_history` (str|Path, optional): Training history/logs file
- `dataset_info` (str|Path|dict, optional): Dataset information file or data
- `model_config` (str|Path|dict, optional): Model configuration file or data
- `**kwargs`: Additional metadata

**Returns:** `ModelCard` instance

**Example:**
```python
card = generator.generate(
    eval_results="eval.json",
    training_history="training.log",
    dataset_info="dataset.json",
    model_config="config.yaml",
    model_name="sentiment-classifier",
    authors=["Alice", "Bob"]
)
```

##### `generate_batch()`

Generate multiple model cards concurrently.

**Parameters:**
- `tasks` (List[Dict]): List of task dictionaries
- `max_workers` (int, optional): Maximum concurrent workers

**Returns:** `List[ModelCard]`

**Example:**
```python
tasks = [
    {"eval_results": "eval1.json", "model_name": "model1"},
    {"eval_results": "eval2.json", "model_name": "model2"},
]
cards = generator.generate_batch(tasks, max_workers=4)
```

### ModelCard

Represents a model card with all sections and metadata.

#### Properties

- `model_details`: Model information (name, version, authors, etc.)
- `intended_use`: Description of intended use cases
- `training_details`: Training process information
- `evaluation_results`: Performance metrics
- `ethical_considerations`: Bias risks and mitigation
- `limitations`: Known limitations and recommendations

#### Methods

##### `add_metric(name, value, **kwargs)`

Add a performance metric.

```python
card.add_metric("accuracy", 0.95)
card.add_metric("f1_score", 0.92, confidence_interval=[0.90, 0.94])
```

##### `render(format="markdown")`

Render the model card in specified format.

**Formats:** "markdown", "html", "json"

```python
markdown_content = card.render("markdown")
html_content = card.render("html") 
json_content = card.render("json")
```

##### `save(path)`

Save the model card to file.

```python
card.save("MODEL_CARD.md")
```

##### `export_jsonld(path)`

Export as JSON-LD for machine reading.

```python
card.export_jsonld("model_card.jsonld")
```

## Configuration

### CardConfig

Configuration for model card generation.

```python
config = CardConfig(
    format=CardFormat.HUGGINGFACE,           # Output format
    include_ethical_considerations=True,      # Include ethics section
    include_carbon_footprint=True,          # Include carbon footprint
    include_bias_analysis=True,              # Include bias analysis
    regulatory_standard="gdpr",              # Compliance standard
    template_name="nlp_classification",      # Template to use
    auto_populate=True,                      # Auto-populate missing fields
    validation_strict=False,                 # Strict validation mode
    output_format="markdown"                 # Default output format
)
```

### CardFormat

Supported output formats:

- `CardFormat.HUGGINGFACE`: Hugging Face model card format
- `CardFormat.GOOGLE`: Google Model Cards format
- `CardFormat.EU_CRA`: EU CRA compliant format
- `CardFormat.CUSTOM`: Custom format

## Generation Methods

### From Evaluation Results

```python
# From JSON file
card = generator.generate(eval_results="results.json")

# From dictionary
results = {
    "accuracy": 0.95,
    "f1_score": 0.92,
    "model_name": "classifier"
}
card = generator.generate(eval_results=results)
```

### From MLflow

```python
card = generator.from_mlflow("model_name", version=2)
```

### From Weights & Biases

```python
card = generator.from_wandb("run_id", project="project_name")
```

## Validation

### Enhanced Validation

```python
from modelcard_generator.core.enhanced_validation import validate_model_card_enhanced

result = await validate_model_card_enhanced(
    card, 
    enable_auto_fix=True,
    learn_patterns=True
)

print(f"Valid: {result.is_valid}")
print(f"Score: {result.overall_score:.2%}")
print(f"Issues: {len(result.issues)}")
```

### Validation Result

```python
class ValidationResult:
    is_valid: bool                    # Overall validity
    overall_score: float             # Quality score (0-1)
    issues: List[ValidationIssue]    # Found issues
    suggestions: List[str]           # Improvement suggestions
    auto_fixes_applied: List[str]    # Applied auto-fixes
    validation_time_ms: float        # Validation time
```

### Validation Issue

```python
class ValidationIssue:
    category: ValidationCategory      # Issue category
    severity: ValidationSeverity     # Issue severity
    message: str                     # Description
    field_path: str                  # Field location
    suggested_fix: str               # Fix suggestion
    auto_fixable: bool              # Can be auto-fixed
```

## Formats

### Hugging Face Format

Standard Hugging Face model card with README.md structure.

```python
config = CardConfig(format=CardFormat.HUGGINGFACE)
card = generator.generate(eval_results="results.json")
card.save("README.md")
```

### Google Model Cards

Structured Google Model Cards format with schema validation.

```python
from modelcard_generator.formats import GoogleModelCard

card = GoogleModelCard()
card.model_details.name = "text-classifier"
card.add_performance_metric("accuracy", 0.95)
```

### EU CRA Compliant

European Cyber Resilience Act compliant format.

```python
from modelcard_generator.formats import EUCRAModelCard

card = EUCRAModelCard()
card.risk_assessment(
    risk_level="limited",
    mitigation_measures=["Human oversight", "Regular audits"]
)
```

## CLI Interface

### Commands

#### Generate

```bash
# Basic generation
mcg generate results.json --output MODEL_CARD.md

# With multiple sources
mcg generate \
  --eval results.json \
  --training training.log \
  --dataset dataset.json \
  --config model_config.yaml \
  --output card.md

# With metadata
mcg generate results.json \
  --model-name "classifier" \
  --model-version "1.2.0" \
  --authors "Alice,Bob" \
  --license "apache-2.0"
```

#### Validate

```bash
# Validate model card
mcg validate MODEL_CARD.md --standard huggingface

# Check compliance
mcg validate MODEL_CARD.md --standard eu-cra --min-score 0.9

# Generate validation report
mcg validate MODEL_CARD.md --output validation_report.json
```

#### Check Drift

```bash
# Check for metric drift
mcg check-drift MODEL_CARD.md --against new_results.json

# With custom threshold
mcg check-drift MODEL_CARD.md --against results.json --threshold 0.05

# Auto-update on drift
mcg check-drift MODEL_CARD.md --against results.json --update
```

#### Initialize Template

```bash
# Create basic template
mcg init --format huggingface

# Create domain-specific template
mcg init --format huggingface --template llm --output LLM_CARD.md
```

### CLI Options

- `--format`: Output format (huggingface, google, eu-cra)
- `--output`, `-o`: Output file path
- `--verbose`, `-v`: Enable verbose logging
- `--config-file`: Configuration file path
- `--auto-populate`: Auto-populate missing sections
- `--include-ethical`: Include ethical considerations
- `--regulatory-standard`: Compliance standard

## Internationalization

### Setting Language

```python
from modelcard_generator.i18n import set_language, _

# Set language
set_language("es")  # Spanish
set_language("fr")  # French
set_language("de")  # German

# Get translated text
title = _("model_card.title")  # "Tarjeta del Modelo" in Spanish
```

### Supported Languages

- `en`: English
- `es`: Spanish (Español)
- `fr`: French (Français)  
- `de`: German (Deutsch)
- `ja`: Japanese (日本語)
- `zh`: Chinese (中文)

### Localized Model Cards

```python
from modelcard_generator.i18n import LocalizedModelCard

# Create localized card
localized = LocalizedModelCard(language="es")
title = localized.get_section_title("model_details")  # "Detalles del Modelo"
```

## Error Handling

### Exceptions

```python
from modelcard_generator.core.exceptions import (
    ModelCardError,
    ValidationError,
    DataSourceError,
    SecurityError
)

try:
    card = generator.generate(eval_results="invalid.json")
except DataSourceError as e:
    print(f"Data source error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except ModelCardError as e:
    print(f"General error: {e}")
```

## Examples

### Complete Example

```python
from modelcard_generator import (
    ModelCardGenerator, 
    CardConfig, 
    CardFormat
)

# Configure generator
config = CardConfig(
    format=CardFormat.HUGGINGFACE,
    include_ethical_considerations=True,
    regulatory_standard="gdpr",
    auto_populate=True
)

generator = ModelCardGenerator(config)

# Generate comprehensive model card
card = generator.generate(
    eval_results="evaluation_results.json",
    training_history="training.log",
    dataset_info="dataset_info.json",
    model_config="model_config.yaml",
    model_name="sentiment-classifier-v2",
    model_version="2.1.0",
    authors=["Data Science Team"],
    license="apache-2.0",
    intended_use="Sentiment analysis for product reviews"
)

# Validate
from modelcard_generator.core.enhanced_validation import validate_model_card_enhanced

result = await validate_model_card_enhanced(card, enable_auto_fix=True)
print(f"Validation score: {result.overall_score:.2%}")

# Save in multiple formats
card.save("README.md")                    # Markdown
card.export_jsonld("model_card.jsonld")  # JSON-LD

# Render as HTML
html_content = card.render("html")
Path("model_card.html").write_text(html_content)
```

### Batch Processing

```python
# Process multiple models
tasks = []
for i in range(10):
    task = {
        "eval_results": f"results/model_{i}_eval.json",
        "model_config": f"configs/model_{i}_config.yaml",
        "model_name": f"model_{i}",
        "model_version": "1.0.0"
    }
    tasks.append(task)

# Generate all cards concurrently
cards = generator.generate_batch(tasks, max_workers=4)

# Save all cards
for i, card in enumerate(cards):
    card.save(f"cards/MODEL_CARD_{i}.md")
```

### Integration with CI/CD

```python
from modelcard_generator import ModelCardGenerator
from modelcard_generator.core.drift_detector import DriftDetector

# Generate card from CI artifacts
generator = ModelCardGenerator()
card = generator.generate(
    eval_results="ci_artifacts/eval_results.json",
    model_config="ci_artifacts/model_config.yaml"
)

# Check for drift against previous version
detector = DriftDetector()
drift_report = detector.check(
    card=card,
    new_eval_results="ci_artifacts/eval_results.json",
    thresholds={"accuracy": 0.02, "f1_score": 0.03}
)

if drift_report.has_drift:
    print("⚠️ Model drift detected!")
    for change in drift_report.significant_changes:
        print(f"{change.metric_name}: {change.old_value} → {change.new_value}")
    
    # Fail CI if drift is significant
    if len(drift_report.significant_changes) > 2:
        exit(1)

# Save card for deployment
card.save("deployment/MODEL_CARD.md")
```

## Best Practices

1. **Always validate** model cards before deployment
2. **Use auto-population** to reduce manual work
3. **Enable ethical considerations** for responsible AI
4. **Check for drift** regularly in production
5. **Use appropriate compliance** standards for your region
6. **Batch process** multiple cards for efficiency
7. **Version control** model cards with your models
8. **Automate generation** in CI/CD pipelines
