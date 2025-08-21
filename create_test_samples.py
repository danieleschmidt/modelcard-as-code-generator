#!/usr/bin/env python3
"""Create test samples for Generation 2 validation."""

import json
import tempfile
from pathlib import Path

def create_test_files():
    """Create test evaluation results and configuration files."""
    
    # Create temporary directory for test files
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Sample evaluation results
    eval_results = {
        "model_name": "sentiment-classifier-v2",
        "accuracy": 0.924,
        "precision": 0.918,
        "recall": 0.931,
        "f1_score": 0.924,
        "roc_auc": 0.965,
        "inference_time_ms": 23.5,
        "dataset": ["imdb", "amazon_reviews"]
    }
    
    with open(test_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Sample model configuration
    model_config = {
        "name": "sentiment-classifier-v2",
        "version": "2.1.0",
        "description": "Advanced sentiment classification model for product reviews",
        "authors": ["Terry AI", "Terragon Labs"],
        "license": "apache-2.0",
        "base_model": "bert-base-multilingual",
        "language": ["en", "es", "fr"],
        "tags": ["sentiment-analysis", "nlp", "classification"],
        "framework": "transformers",
        "architecture": "BERT",
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "max_length": 512
        }
    }
    
    with open(test_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Sample dataset information
    dataset_info = {
        "datasets": ["imdb", "amazon_reviews_multi"],
        "training_data": ["imdb_train", "amazon_reviews_train"],
        "preprocessing": "Lowercase, remove special chars, max_length=512",
        "bias_analysis": {
            "bias_risks": [
                "May exhibit demographic bias in predictions",
                "Performance varies across different product categories"
            ],
            "fairness_metrics": {
                "demographic_parity": 0.02,
                "equal_opportunity": 0.015
            }
        }
    }
    
    with open(test_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Test files created in {test_dir}:")
    print(f"  - eval_results.json")
    print(f"  - model_config.json") 
    print(f"  - dataset_info.json")
    
    return test_dir

if __name__ == "__main__":
    create_test_files()