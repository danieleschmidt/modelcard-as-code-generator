#!/usr/bin/env python3
"""Basic Model Card Generation Example."""

from modelcard_generator import ModelCardGenerator, CardConfig, CardFormat

def main():
    # Create generator with configuration
    config = CardConfig(
        format=CardFormat.HUGGINGFACE,
        include_ethical_considerations=True,
        auto_populate=True
    )
    
    generator = ModelCardGenerator(config)
    
    # Generate from evaluation results
    card = generator.generate(
        eval_results={
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95
        },
        model_name="sentiment-classifier",
        model_version="1.0.0",
        authors=["Data Science Team"],
        license="apache-2.0",
        intended_use="Sentiment analysis for product reviews"
    )
    
    # Save model card
    card.save("examples/BASIC_MODEL_CARD.md")
    print("✅ Basic model card generated: examples/BASIC_MODEL_CARD.md")

if __name__ == "__main__":
    main()
