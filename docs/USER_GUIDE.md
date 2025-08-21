# Model Card Generator User Guide

## Getting Started

### Installation

Install the Model Card Generator using pip:

```bash
pip install modelcard-as-code-generator
```

For CLI tools:
```bash
pip install modelcard-as-code-generator[cli]
```

For development:
```bash
pip install modelcard-as-code-generator[dev]
```

### Quick Start

1. **Generate your first model card:**

```bash
mcg generate evaluation_results.json --output MODEL_CARD.md
```

2. **Validate the generated card:**

```bash
mcg validate MODEL_CARD.md
```

3. **Check for drift:**

```bash
mcg check-drift MODEL_CARD.md --against new_results.json
```

## Basic Usage

### Command Line Interface

The easiest way to get started is with the command line interface (CLI).

#### Generate a Model Card

```bash
# From evaluation results
mcg generate results.json

# Specify output file
mcg generate results.json --output MODEL_CARD.md

# Include multiple sources
mcg generate \
  --eval results.json \
  --training training.log \
  --dataset dataset.json \
  --output comprehensive_card.md
```

#### Add Metadata

```bash
mcg generate results.json \
  --model-name "sentiment-classifier" \
  --model-version "1.2.0" \
  --authors "Alice,Bob" \
  --license "apache-2.0" \
  --intended-use "Analyze product review sentiment"
```

#### Choose Output Format

```bash
# Hugging Face format (default)
mcg generate results.json --format huggingface

# Google Model Cards format
mcg generate results.json --format google

# EU CRA compliant format
mcg generate results.json --format eu-cra
```

### Python API

For more control, use the Python API:

```python
from modelcard_generator import ModelCardGenerator

# Create generator
generator = ModelCardGenerator()

# Generate card
card = generator.generate(eval_results="results.json")

# Save card
card.save("MODEL_CARD.md")
```

## Data Sources

The Model Card Generator can extract information from various sources:

### Evaluation Results

**JSON Format:**
```json
{
  "model_name": "sentiment-classifier",
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.97,
  "f1_score": 0.95,
  "inference_time_ms": 23
}
```

**CSV Format:**
```csv
metric,value
accuracy,0.95
precision,0.93
recall,0.97
f1_score,0.95
```

### Model Configuration

**YAML Format:**
```yaml
name: sentiment-classifier
version: 1.2.0
description: Advanced sentiment analysis model
authors:
  - Alice Smith
  - Bob Johnson
license: apache-2.0
framework: transformers
architecture: BERT
hyperparameters:
  learning_rate: 2e-5
  batch_size: 32
  epochs: 3
```

### Dataset Information

```json
{
  "datasets": ["imdb", "amazon_reviews"],
  "training_data": ["imdb_train", "amazon_train"],
  "preprocessing": "Lowercase, remove special chars",
  "bias_analysis": {
    "bias_risks": [
      "May exhibit demographic bias",
      "Performance varies by product category"
    ]
  }
}
```

## Validation

### Automatic Validation

The generator includes automatic validation:

```bash
mcg validate MODEL_CARD.md
```

Output:
```
📋 Validation Results for MODEL_CARD.md
📊 Standard: huggingface
✅ Valid: Yes
🎯 Score: 92.5%
📈 Completeness: 85.0%
```

### Validation Standards

- `huggingface`: Hugging Face model card standard
- `google`: Google Model Cards standard  
- `eu-cra`: EU Cyber Resilience Act
- `gdpr`: GDPR compliance
- `eu_ai_act`: EU AI Act compliance

### Fix Issues Automatically

```bash
mcg validate MODEL_CARD.md --fix
```

## Drift Detection

Monitor your model cards for changes over time:

```bash
# Check for drift
mcg check-drift MODEL_CARD.md --against new_results.json

# Custom threshold (2% for accuracy)
mcg check-drift MODEL_CARD.md --against results.json --threshold 0.02

# Auto-update card if drift detected
mcg check-drift MODEL_CARD.md --against results.json --update
```

Example output:
```
🔍 Drift Detection Results
📊 Compared against: new_results.json
⚡ Drift detected: Yes
🔢 Total changes: 3
⚠️  Significant changes: 1

📈 Metric Changes:
  ⚠️ accuracy: 0.9200 → 0.8950 (-0.0250)
  ℹ️ precision: 0.9100 → 0.9050 (-0.0050)
  ℹ️ recall: 0.9300 → 0.9280 (-0.0020)
```

## Templates

### Use Existing Templates

```bash
# List available templates
mcg init --help

# Create NLP classification template
mcg init --format huggingface --template nlp_classification

# Create computer vision template  
mcg init --format huggingface --template computer_vision

# Create LLM template
mcg init --format huggingface --template llm
```

### Domain-Specific Templates

#### NLP Models
```bash
mcg init --template nlp_classification --output NLP_CARD.md
```

#### Computer Vision Models
```bash
mcg init --template computer_vision --output CV_CARD.md
```

#### Large Language Models
```bash
mcg init --template llm --output LLM_CARD.md
```

## Compliance

### GDPR Compliance

For models processing personal data in the EU:

```bash
mcg generate results.json \
  --regulatory-standard gdpr \
  --format eu-cra
```

Required sections for GDPR:
- Data protection measures
- Consent mechanism  
- Data retention policy
- Right to erasure

### EU AI Act Compliance

For AI systems in the EU:

```bash
mcg generate results.json \
  --regulatory-standard eu_ai_act \
  --format eu-cra
```

Required sections:
- AI system classification
- Risk assessment
- Conformity assessment
- Human oversight

## Internationalization

### Supported Languages

The Model Card Generator supports 6 languages:

- English (en)
- Spanish (es) 
- French (fr)
- German (de)
- Japanese (ja)
- Chinese (zh)

### Set Language

```bash
# Set environment variable
export LANG=es_ES.UTF-8
mcg generate results.json

# Or use Python API
from modelcard_generator.i18n import set_language
set_language("es")
```

### Generate Multilingual Cards

```python
from modelcard_generator import ModelCardGenerator
from modelcard_generator.i18n import set_language

generator = ModelCardGenerator()

# Generate in multiple languages
languages = ["en", "es", "fr", "de"]
for lang in languages:
    set_language(lang)
    card = generator.generate(eval_results="results.json")
    card.save(f"MODEL_CARD_{lang}.md")
```

## Integration

### CI/CD Integration

#### GitHub Actions

```yaml
name: Model Card Generation

on:
  push:
    paths: ['models/**', 'results/**']

jobs:
  generate-model-card:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install Model Card Generator
        run: pip install modelcard-as-code-generator
      
      - name: Generate Model Card
        run: |
          mcg generate results/eval.json \
            --output MODEL_CARD.md \
            --model-name "production-model" \
            --model-version "${{ github.sha }}"
      
      - name: Validate Model Card
        run: mcg validate MODEL_CARD.md --standard huggingface
      
      - name: Check Drift
        run: |
          mcg check-drift MODEL_CARD.md \
            --against results/eval.json \
            --fail-on-drift
      
      - name: Commit Model Card
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add MODEL_CARD.md
          git commit -m "Update model card" || exit 0
          git push
```

#### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Generate Model Card') {
            steps {
                sh '''
                    pip install modelcard-as-code-generator
                    mcg generate results/eval.json \
                      --output MODEL_CARD.md \
                      --model-name "${JOB_NAME}" \
                      --model-version "${BUILD_NUMBER}"
                '''
            }
        }
        
        stage('Validate') {
            steps {
                sh 'mcg validate MODEL_CARD.md --standard huggingface'
            }
        }
        
        stage('Check Drift') {
            steps {
                sh '''
                    mcg check-drift MODEL_CARD.md \
                      --against results/eval.json \
                      --threshold 0.05
                '''
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'MODEL_CARD.md'
        }
    }
}
```

### MLflow Integration

```python
import mlflow
from modelcard_generator import ModelCardGenerator

# Track model card generation
with mlflow.start_run():
    generator = ModelCardGenerator()
    card = generator.from_mlflow("model_name", version=1)
    
    # Log model card as artifact
    card.save("model_card.md")
    mlflow.log_artifact("model_card.md")
    
    # Log metadata
    mlflow.log_param("card_format", "huggingface")
    mlflow.log_metric("card_completeness", 0.85)
```

### Weights & Biases Integration

```python
import wandb
from modelcard_generator import ModelCardGenerator

# Initialize W&B
wandb.init(project="model-cards")

# Generate card from W&B run
generator = ModelCardGenerator()
card = generator.from_wandb(wandb.run.id, project="my-project")

# Log to W&B
wandb.log_artifact("model_card.md")
wandb.log({"card_completeness": 0.85})
```

## Best Practices

### 1. Version Control Model Cards

Always version control your model cards alongside your models:

```
project/
├── models/
│   └── v1.2.0/
│       ├── model.pkl
│       └── MODEL_CARD.md
├── data/
└── results/
```

### 2. Automate Generation

Set up automated model card generation in your ML pipeline:

```python
# In your training script
from modelcard_generator import ModelCardGenerator

def train_model():
    # ... training code ...
    
    # Generate model card automatically
    generator = ModelCardGenerator()
    card = generator.generate(
        eval_results=eval_results,
        model_config=config,
        model_name=model_name,
        model_version=version
    )
    card.save(f"models/{version}/MODEL_CARD.md")
```

### 3. Regular Validation

Validate model cards regularly:

```bash
# Add to your CI/CD pipeline
mcg validate MODEL_CARD.md --min-score 0.8 --standard huggingface
```

### 4. Monitor for Drift

Check for drift when deploying new model versions:

```bash
mcg check-drift production/MODEL_CARD.md \
  --against staging/eval_results.json \
  --fail-on-drift
```

### 5. Use Appropriate Standards

Choose the right compliance standard for your use case:

- **General use**: `huggingface` 
- **EU deployment**: `eu-cra`, `gdpr`
- **Healthcare**: Medical AI templates
- **Finance**: Financial AI templates

### 6. Document Limitations

Always document model limitations honestly:

```python
card.add_limitation("Performance degrades on out-of-distribution data")
card.add_limitation("Not validated for medical diagnosis")
card.add_limitation("May exhibit bias against underrepresented groups")
```

### 7. Include Ethical Considerations

Enable ethical considerations for responsible AI:

```python
config = CardConfig(include_ethical_considerations=True)
generator = ModelCardGenerator(config)
```

## Troubleshooting

### Common Issues

#### 1. File Not Found
```
Error: DataSourceError: File not found: results.json
```
**Solution**: Check file path and ensure file exists.

#### 2. Invalid JSON Format
```
Error: DataSourceError: Invalid JSON format: Expecting ',' delimiter
```
**Solution**: Validate JSON syntax using a JSON validator.

#### 3. Validation Failures
```
Error: ValidationError: Model card validation failed
```
**Solution**: Use `--fix` flag or review validation issues.

#### 4. Missing Dependencies
```
Error: ModuleNotFoundError: No module named 'wandb'
```
**Solution**: Install optional dependencies:
```bash
pip install modelcard-as-code-generator[integrations]
```

### Getting Help

1. **Check the documentation**: This user guide and API reference
2. **View examples**: Check the `examples/` directory
3. **Run with verbose output**: Use `--verbose` flag
4. **Check GitHub issues**: Search for similar problems
5. **Create an issue**: Report bugs or request features

### Debug Mode

Enable debug logging for troubleshooting:

```bash
mcg generate results.json --verbose --log-file debug.log
```

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from modelcard_generator import ModelCardGenerator
generator = ModelCardGenerator()
card = generator.generate(eval_results="results.json")
```

## Advanced Usage

### Custom Templates

Create custom templates for your organization:

```python
from modelcard_generator.templates import Template

class BiometricModelTemplate(Template):
    def __init__(self):
        super().__init__(
            name="biometric_model",
            required_sections=[
                "privacy_protection",
                "consent_mechanism", 
                "data_retention"
            ]
        )
    
    def privacy_protection(self, measures):
        return f"## Privacy Protection\n{measures}"

# Register template
from modelcard_generator.templates import TemplateLibrary
TemplateLibrary.register(BiometricModelTemplate())
```

### Custom Validation Rules

Add custom validation rules:

```python
from modelcard_generator.core.enhanced_validation import (
    EnhancedValidator, ValidationRule, ValidationCategory, ValidationSeverity
)

def check_custom_requirement(card):
    issues = []
    if not card.model_details.license:
        issues.append(ValidationIssue(
            category=ValidationCategory.COMPLIANCE,
            severity=ValidationSeverity.ERROR,
            message="License is required for internal models",
            field_path="model_details.license"
        ))
    return issues

# Register rule
validator = EnhancedValidator()
rule = ValidationRule(
    name="custom_license_check",
    category=ValidationCategory.COMPLIANCE,
    severity=ValidationSeverity.ERROR,
    check_function=check_custom_requirement
)
validator.register_rule(rule)
```

### Batch Processing

Process multiple models efficiently:

```python
import glob
from modelcard_generator import ModelCardGenerator

generator = ModelCardGenerator()

# Find all evaluation files
eval_files = glob.glob("results/*/eval.json")

# Create batch tasks
tasks = []
for eval_file in eval_files:
    model_name = eval_file.split("/")[-2]  # Extract model name from path
    task = {
        "eval_results": eval_file,
        "model_name": model_name,
        "model_version": "1.0.0"
    }
    tasks.append(task)

# Generate all cards
cards = generator.generate_batch(tasks, max_workers=4)

# Save all cards
for i, card in enumerate(cards):
    model_name = tasks[i]["model_name"] 
    card.save(f"cards/{model_name}/MODEL_CARD.md")
```

## Next Steps

1. **Explore examples**: Check out example model cards in various formats
2. **Set up automation**: Integrate with your ML pipeline
3. **Customize**: Create templates specific to your domain
4. **Contribute**: Help improve the tool by contributing code or documentation
5. **Stay updated**: Follow releases for new features and improvements

For more advanced usage, see the [API Reference](API_REFERENCE.md) and [Examples](examples/).
