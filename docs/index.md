# Model Card Generator

Automated generation of Model Cards as executable, versioned artifacts. Converts evaluation results, dataset statistics, and training logs into standardized documentation that satisfies regulatory requirements and enables "card drift" detection in CI/CD.

## 🎯 Key Features

- **Multi-Standard Support**: Generate cards for Hugging Face, Google Model Cards, and EU CRA
- **CI/CD Integration**: Fail builds on model card drift or missing information  
- **Executable Cards**: Model cards that can run their own validation tests
- **Version Control**: Track model card evolution with git-friendly formats
- **Auto-Population**: Extract metadata from training logs, configs, and evaluations
- **Regulatory Compliance**: Templates for GDPR, EU AI Act, and other frameworks

## 🚀 Quick Start

### Installation

```bash
pip install modelcard-as-code-generator
```

### Basic Usage

```bash
# Generate from evaluation results
mcg generate results/eval.json --format huggingface --output MODEL_CARD.md

# Validate existing card
mcg validate MODEL_CARD.md --standard eu-cra

# Check for drift
mcg check-drift MODEL_CARD.md --against results/new_eval.json
```

### Python API

```python
from modelcard_generator import ModelCardGenerator, CardConfig

config = CardConfig(format="huggingface")
generator = ModelCardGenerator(config)

card = generator.generate(
    eval_results="results/eval.json",
    training_history="logs/training.log"
)

card.save("MODEL_CARD.md")
```

## 📋 Documentation Sections

- **[Getting Started](getting-started/installation.md)**: Installation and setup
- **[User Guide](user-guide/overview.md)**: Comprehensive usage guide
- **[API Reference](api/cli.md)**: CLI and Python API documentation
- **[Integration](integration/cicd.md)**: CI/CD and platform integrations
- **[Examples](examples/basic.md)**: Practical usage examples

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](development/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [License](about/license.md) file for details.