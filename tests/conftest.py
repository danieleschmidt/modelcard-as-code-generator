"""
Pytest configuration and shared fixtures for Model Card Generator tests.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import yaml
from hypothesis import HealthCheck, settings

# Configure Hypothesis for faster test runs in CI
settings.register_profile(
    "ci",
    max_examples=10,
    deadline=5000,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=100,
    deadline=None,
)

# Load appropriate profile based on environment
import os
if os.getenv("CI"):
    settings.load_profile("ci")
else:
    settings.load_profile("dev")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_eval_results() -> Dict[str, Any]:
    """Sample evaluation results for testing."""
    return {
        "model_name": "test-sentiment-classifier",
        "model_version": "1.0.0",
        "evaluation_date": "2025-01-15T10:30:00Z",
        "metrics": {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91,
            "roc_auc": 0.96,
            "loss": 0.08
        },
        "dataset": {
            "name": "sentiment_test_set",
            "size": 10000,
            "source": "customer_reviews",
            "version": "2.1"
        },
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "adam"
        },
        "training_time": 3600,
        "inference_time_ms": 15.2
    }


@pytest.fixture
def sample_training_config() -> Dict[str, Any]:
    """Sample training configuration for testing."""
    return {
        "model": {
            "architecture": "transformer",
            "base_model": "bert-base-uncased",
            "num_labels": 3,
            "dropout": 0.1
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_epochs": 10,
            "early_stopping": True,
            "patience": 3
        },
        "data": {
            "train_path": "data/train.csv",
            "val_path": "data/val.csv",
            "test_path": "data/test.csv",
            "text_column": "review",
            "label_column": "sentiment"
        }
    }


@pytest.fixture
def sample_huggingface_card() -> str:
    """Sample Hugging Face model card content."""
    return """---
language: en
license: apache-2.0
tags:
- sentiment-analysis
- text-classification
datasets:
- imdb
metrics:
- accuracy
- f1
---

# Sentiment Analysis Model

## Model Description

This model classifies text sentiment into positive, negative, or neutral categories.

## Training Data

Trained on IMDB movie reviews dataset containing 50,000 reviews.

## Evaluation Results

- Accuracy: 92%
- F1 Score: 91%
- Precision: 89%
- Recall: 94%

## Usage

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="test-sentiment-classifier")
result = classifier("This movie is great!")
```

## Limitations

- May not perform well on domain-specific text
- Sensitive to text length variations
"""


@pytest.fixture
def sample_model_card_data() -> Dict[str, Any]:
    """Complete model card data structure for testing."""
    return {
        "model_details": {
            "name": "sentiment-classifier-v1",
            "version": "1.0.0",
            "description": "BERT-based sentiment classification model",
            "architecture": "transformer",
            "framework": "pytorch",
            "license": "apache-2.0",
            "languages": ["en"],
            "tags": ["sentiment-analysis", "text-classification"]
        },
        "intended_use": {
            "primary_intended_uses": "Sentiment analysis for customer feedback",
            "primary_intended_users": "Product teams and customer service",
            "out_of_scope_use_cases": "Medical or legal text analysis"
        },
        "factors": {
            "relevant_factors": ["text_length", "domain", "language_variety"],
            "evaluation_factors": ["accuracy", "fairness", "robustness"]
        },
        "metrics": {
            "model_performance_measures": "accuracy, precision, recall, f1",
            "decision_thresholds": "0.5 for binary classification",
            "variation_approaches": "stratified sampling"
        },
        "evaluation_data": {
            "dataset": "IMDB movie reviews",
            "motivation": "Standard benchmark for sentiment analysis",
            "preprocessing": "tokenization, lowercasing"
        },
        "training_data": {
            "dataset": "Combined IMDB and Amazon reviews",
            "preprocessing": "cleaning, augmentation",
            "size": "100,000 samples"
        },
        "quantitative_analyses": {
            "unitary_results": {"accuracy": 0.92, "f1": 0.91},
            "intersectional_results": {"by_domain": {"movies": 0.94, "products": 0.89}}
        },
        "ethical_considerations": {
            "sensitive_data": "No personal information used",
            "human_life": "Low risk application",
            "mitigations": "Regular bias testing"
        },
        "caveats_and_recommendations": {
            "limitations": "Performance degrades on short text",
            "recommendations": "Use ensemble for critical applications"
        }
    }


@pytest.fixture
def eval_results_file(temp_dir: Path, sample_eval_results: Dict[str, Any]) -> Path:
    """Create a temporary evaluation results JSON file."""
    file_path = temp_dir / "eval_results.json"
    with open(file_path, "w") as f:
        json.dump(sample_eval_results, f, indent=2)
    return file_path


@pytest.fixture
def config_file(temp_dir: Path, sample_training_config: Dict[str, Any]) -> Path:
    """Create a temporary configuration YAML file."""
    file_path = temp_dir / "config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(sample_training_config, f, default_flow_style=False)
    return file_path


@pytest.fixture
def model_card_file(temp_dir: Path, sample_huggingface_card: str) -> Path:
    """Create a temporary model card Markdown file."""
    file_path = temp_dir / "MODEL_CARD.md"
    with open(file_path, "w") as f:
        f.write(sample_huggingface_card)
    return file_path


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up mock environment variables for testing."""
    test_env_vars = {
        "MCG_ENVIRONMENT": "test",
        "MCG_LOG_LEVEL": "DEBUG",
        "MCG_CACHE_DIR": "/tmp/mcg_test_cache",
        "MCG_OUTPUT_DIR": "/tmp/mcg_test_output",
        "MCG_DEFAULT_FORMAT": "huggingface",
        "MCG_ENABLE_SECRET_SCANNING": "true",
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def mock_requests(requests_mock):
    """Mock external API requests for testing."""
    # Mock MLflow API
    requests_mock.get(
        "http://localhost:5000/api/2.0/mlflow/experiments/get",
        json={"experiment": {"experiment_id": "1", "name": "test-experiment"}}
    )
    
    # Mock Weights & Biases API
    requests_mock.get(
        "https://api.wandb.ai/api/v1/runs",
        json={"runs": [{"id": "test-run", "config": {}, "summary": {}}]}
    )
    
    # Mock Hugging Face Hub API
    requests_mock.get(
        "https://huggingface.co/api/models/test-model",
        json={"modelId": "test-model", "tags": ["sentiment-analysis"]}
    )
    
    return requests_mock


class TestConstants:
    """Test constants used across multiple test modules."""
    
    SAMPLE_METRICS = {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.94,
        "f1_score": 0.91
    }
    
    SAMPLE_TAGS = ["sentiment-analysis", "text-classification", "bert"]
    
    SAMPLE_DATASET_INFO = {
        "name": "imdb",
        "size": 50000,
        "splits": ["train", "test", "validation"]
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "network: marks tests as requiring network access"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.name or "slow" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        
        # Mark network tests
        if any(keyword in item.name.lower() for keyword in ["api", "http", "request", "network"]):
            item.add_marker(pytest.mark.network)


# Helper functions for test utilities
def assert_valid_json(content: str) -> Dict[str, Any]:
    """Assert that content is valid JSON and return parsed data."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON content: {e}")


def assert_valid_yaml(content: str) -> Dict[str, Any]:
    """Assert that content is valid YAML and return parsed data."""
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        pytest.fail(f"Invalid YAML content: {e}")


def assert_model_card_structure(card_content: str, required_sections: list = None) -> None:
    """Assert that model card has required structure."""
    if required_sections is None:
        required_sections = ["# ", "## "]  # At least one h1 and one h2
    
    for section in required_sections:
        assert section in card_content, f"Missing required section: {section}"


def create_test_file(temp_dir: Path, filename: str, content: str) -> Path:
    """Create a test file with given content."""
    file_path = temp_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)
    return file_path