#!/usr/bin/env python3
"""Advanced Model Card Generation Example."""

import asyncio
from pathlib import Path

from modelcard_generator import ModelCardGenerator, CardConfig, CardFormat
from modelcard_generator.core.enhanced_validation import validate_model_card_enhanced
from modelcard_generator.core.drift_detector import DriftDetector

async def main():
    # Advanced configuration
    config = CardConfig(
        format=CardFormat.HUGGINGFACE,
        include_ethical_considerations=True,
        include_carbon_footprint=True,
        include_bias_analysis=True,
        regulatory_standard="gdpr",
        auto_populate=True,
        validation_strict=True
    )
    
    generator = ModelCardGenerator(config)
    
    # Generate comprehensive model card
    card = generator.generate(
        eval_results={
            "accuracy": 0.924,
            "precision": 0.918,
            "recall": 0.931,
            "f1_score": 0.924,
            "roc_auc": 0.965,
            "inference_time_ms": 23.5
        },
        training_history="Trained for 3 epochs with learning rate 2e-5",
        dataset_info={
            "datasets": ["imdb", "amazon_reviews"],
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
        },
        model_config={
            "framework": "transformers",
            "architecture": "BERT",
            "base_model": "bert-base-multilingual",
            "hyperparameters": {
                "learning_rate": 2e-5,
                "batch_size": 32,
                "epochs": 3,
                "max_length": 512
            }
        },
        model_name="advanced-sentiment-classifier",
        model_version="2.1.0",
        authors=["Advanced ML Team", "Terry AI"],
        license="apache-2.0",
        intended_use="Advanced sentiment analysis for multilingual product reviews"
    )
    
    print(f"📊 Generated model card: {card.model_details.name} v{card.model_details.version}")
    print(f"🏷️ Tags: {card.model_details.tags}")
    print(f"📈 Metrics: {len(card.evaluation_results)} evaluation metrics")
    
    # Enhanced validation with auto-fix
    validation_result = await validate_model_card_enhanced(
        card, 
        enable_auto_fix=True,
        learn_patterns=True
    )
    
    print(f"\n🔍 Validation Results:")
    print(f"✅ Valid: {validation_result.is_valid}")
    print(f"🎯 Score: {validation_result.overall_score:.2%}")
    print(f"⚠️  Issues: {len(validation_result.issues)}")
    print(f"🔧 Auto-fixes applied: {len(validation_result.auto_fixes_applied)}")
    print(f"⏱️  Validation time: {validation_result.validation_time_ms:.1f}ms")
    
    if validation_result.suggestions:
        print("\n💡 Suggestions:")
        for suggestion in validation_result.suggestions:
            print(f"   - {suggestion}")
    
    # Save in multiple formats
    card.save("examples/ADVANCED_MODEL_CARD.md")
    card.export_jsonld("examples/advanced_model_card.jsonld")
    
    # Export as HTML
    html_content = card.render("html")
    Path("examples/advanced_model_card.html").write_text(html_content)
    
    print("\n💾 Files saved:")
    print("   - examples/ADVANCED_MODEL_CARD.md")
    print("   - examples/advanced_model_card.jsonld") 
    print("   - examples/advanced_model_card.html")
    
    # Demonstrate drift detection
    print("\n🔄 Testing drift detection...")
    
    # Simulate new evaluation results with slight drift
    new_results = {
        "accuracy": 0.919,  # Slight decrease
        "precision": 0.920, # Slight increase
        "recall": 0.928,    # Slight decrease
        "f1_score": 0.924,  # Same
        "roc_auc": 0.963,   # Slight decrease
        "inference_time_ms": 25.2  # Slight increase
    }
    
    detector = DriftDetector()
    drift_report = detector.check(
        card=card,
        new_eval_results=new_results,
        thresholds={
            "accuracy": 0.01,      # 1% threshold
            "precision": 0.01,
            "recall": 0.01,
            "f1_score": 0.01,
            "roc_auc": 0.01,
            "inference_time_ms": 5  # 5ms threshold
        }
    )
    
    print(f"📊 Drift detection results:")
    print(f"⚡ Drift detected: {drift_report.has_drift}")
    print(f"🔢 Total changes: {len(drift_report.changes)}")
    print(f"⚠️  Significant changes: {len(drift_report.significant_changes)}")
    
    if drift_report.changes:
        print("\n📈 Metric changes:")
        for change in drift_report.changes:
            status = "⚠️ " if change.is_significant else "ℹ️ "
            print(f"   {status}{change.metric_name}: {change.old_value:.4f} → {change.new_value:.4f} ({change.delta:+.4f})")
    
    print("\n🎉 Advanced example completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
