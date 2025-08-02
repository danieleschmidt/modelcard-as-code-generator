# Quick Start Guide

## Installation

### From PyPI (Recommended)

```bash
pip install modelcard-as-code-generator
```

### Development Installation

```bash
git clone https://github.com/terragonlabs/modelcard-as-code-generator
cd modelcard-as-code-generator
pip install -e ".[dev]"
```

## Your First Model Card

### 1. Basic Generation

Create a model card from evaluation results:

```bash
mcg generate results/eval.json --format huggingface --output MODEL_CARD.md
```

### 2. Multi-Source Generation

Combine multiple data sources:

```bash
mcg generate \
  --eval results/eval.json \
  --training logs/training.log \
  --dataset data/dataset_stats.json \
  --config config/model_config.yaml \
  --output cards/model_card.md
```

### 3. Validation

Validate your model card:

```bash
mcg validate MODEL_CARD.md --standard huggingface
```

### 4. Drift Detection

Check for changes in model performance:

```bash
mcg check-drift MODEL_CARD.md --against results/new_eval.json
```

## Python API

```python
from modelcard_generator import ModelCardGenerator, CardConfig

# Configure generator
config = CardConfig(
    format="huggingface",
    include_ethical_considerations=True,
    regulatory_standard="eu_ai_act"
)

generator = ModelCardGenerator(config)

# Generate card
card = generator.generate(
    eval_results="results/eval.json",
    training_history="logs/training.log"
)

# Save and export
card.save("MODEL_CARD.md")
card.export_jsonld("model_card.jsonld")
```

## CI/CD Integration

Add to your `.github/workflows/ci.yml`:

```yaml
- name: Generate Model Card
  uses: terragonlabs/modelcard-action@v1
  with:
    eval-results: results/eval.json
    output: MODEL_CARD.md
    
- name: Check Card Drift
  uses: terragonlabs/modelcard-action@v1
  with:
    action: check-drift
    card: MODEL_CARD.md
    fail-on-drift: true
```

## Next Steps

- Read the [User Guide](user-guide.md) for detailed features
- Check [Developer Guide](developer-guide.md) for customization
- Review [Templates Guide](templates.md) for specialized use cases
- See [Compliance Guide](compliance.md) for regulatory requirements