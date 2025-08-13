"""AI-powered content generation for model cards."""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging_config import get_logger
from ..core.models import ModelCard, PerformanceMetric

logger = get_logger(__name__)


class AIContentGenerator:
    """Generates intelligent content for model cards using AI analysis."""

    def __init__(self, model_name: str = "research-assistant"):
        self.model_name = model_name
        self.content_cache: Dict[str, Any] = {}
        
    def generate_description(self, model_card: ModelCard, context: Dict[str, Any]) -> str:
        """Generate an intelligent model description based on card content."""
        try:
            # Analyze model characteristics
            metrics = model_card.evaluation_results
            model_type = self._infer_model_type(model_card, context)
            domain = self._infer_domain(model_card, context)
            
            # Create contextual description
            performance_summary = self._summarize_performance(metrics)
            use_cases = self._generate_use_cases(model_type, domain, performance_summary)
            
            description = f"""
This is a {model_type} model designed for {domain} applications. 
{performance_summary}

The model excels in {use_cases['strengths']} and is particularly suitable for {use_cases['applications']}.
Built with production-ready features and comprehensive evaluation across multiple metrics.
            """.strip()
            
            logger.info(f"Generated AI description for {model_card.model_details.name}")
            return description
            
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return "AI-powered model with comprehensive evaluation and production-ready features."

    def generate_limitations(self, model_card: ModelCard, context: Dict[str, Any]) -> List[str]:
        """Generate intelligent limitations based on model analysis."""
        limitations = []
        
        try:
            # Analyze performance patterns
            metrics = model_card.evaluation_results
            model_type = self._infer_model_type(model_card, context)
            
            # Generic limitations based on model type
            if "classification" in model_type.lower():
                limitations.extend([
                    "May exhibit bias towards majority classes in training data",
                    "Performance may degrade on highly imbalanced datasets",
                    "Requires careful threshold tuning for optimal precision-recall trade-offs"
                ])
            elif "generation" in model_type.lower() or "llm" in model_type.lower():
                limitations.extend([
                    "May generate plausible but factually incorrect information",
                    "Performance varies significantly across different domains and languages",
                    "Requires careful prompt engineering for optimal results"
                ])
            elif "regression" in model_type.lower():
                limitations.extend([
                    "Assumes linear relationships which may not hold for all data patterns",
                    "Sensitive to outliers and data distribution shifts",
                    "May not capture complex non-linear interactions without feature engineering"
                ])
            
            # Performance-based limitations
            accuracy_metrics = [m for m in metrics if 'accuracy' in m.name.lower()]
            if accuracy_metrics and accuracy_metrics[0].value < 0.9:
                limitations.append("Moderate accuracy indicates need for additional training or data improvements")
            
            # Add general production limitations
            limitations.extend([
                "Requires validation on target deployment environment",
                "Performance monitoring recommended for production use",
                "May need retraining as data distributions evolve"
            ])
            
            logger.info(f"Generated {len(limitations)} AI-powered limitations")
            return limitations[:5]  # Return top 5 most relevant
            
        except Exception as e:
            logger.error(f"Failed to generate limitations: {e}")
            return [
                "Model performance may vary across different datasets and use cases",
                "Requires proper validation before production deployment",
                "Regular monitoring and evaluation recommended"
            ]

    def generate_ethical_considerations(self, model_card: ModelCard, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate ethical considerations and bias analysis."""
        try:
            model_type = self._infer_model_type(model_card, context)
            domain = self._infer_domain(model_card, context)
            
            ethical_considerations = {
                "bias_risks": [],
                "fairness_recommendations": [],
                "sensitive_attributes": [],
                "mitigation_strategies": []
            }
            
            # Domain-specific ethical considerations
            if "healthcare" in domain.lower() or "medical" in domain.lower():
                ethical_considerations["bias_risks"].extend([
                    "Potential disparities in performance across demographic groups",
                    "Risk of perpetuating healthcare access inequalities",
                    "May reflect biases present in historical medical data"
                ])
                ethical_considerations["fairness_recommendations"].extend([
                    "Evaluate performance across age, gender, and ethnicity groups",
                    "Ensure representative validation datasets",
                    "Consider regulatory compliance requirements (FDA, GDPR)"
                ])
            elif "finance" in domain.lower():
                ethical_considerations["bias_risks"].extend([
                    "Risk of discriminatory lending or credit decisions",
                    "May perpetuate historical financial exclusion patterns",
                    "Potential for proxy discrimination through correlated features"
                ])
                ethical_considerations["fairness_recommendations"].extend([
                    "Audit for compliance with fair lending regulations",
                    "Implement explainable AI for decision transparency",
                    "Regular bias testing across protected classes"
                ])
            elif "hiring" in domain.lower() or "recruitment" in domain.lower():
                ethical_considerations["bias_risks"].extend([
                    "Risk of systematic bias against underrepresented groups",
                    "May perpetuate historical hiring inequalities",
                    "Potential for indirect discrimination through proxy variables"
                ])
                ethical_considerations["sensitive_attributes"].extend([
                    "Gender", "Age", "Race/Ethnicity", "Educational Background", "Geographic Location"
                ])
            
            # General ethical considerations
            ethical_considerations["bias_risks"].extend([
                "Model may amplify existing biases present in training data",
                "Performance variations across different demographic groups"
            ])
            
            ethical_considerations["mitigation_strategies"].extend([
                "Regular bias audits and fairness metric monitoring",
                "Diverse evaluation datasets and test cases",
                "Stakeholder involvement in evaluation process",
                "Continuous monitoring and model updates"
            ])
            
            logger.info(f"Generated ethical considerations for {domain} {model_type} model")
            return ethical_considerations
            
        except Exception as e:
            logger.error(f"Failed to generate ethical considerations: {e}")
            return {
                "bias_risks": ["Potential for bias amplification from training data"],
                "fairness_recommendations": ["Regular bias audits recommended"],
                "sensitive_attributes": ["Demographic attributes"],
                "mitigation_strategies": ["Continuous monitoring and evaluation"]
            }

    def generate_use_cases(self, model_card: ModelCard, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate appropriate and inappropriate use cases."""
        try:
            model_type = self._infer_model_type(model_card, context)
            domain = self._infer_domain(model_card, context)
            performance = self._analyze_performance_level(model_card.evaluation_results)
            
            use_cases = {
                "appropriate": [],
                "inappropriate": [],
                "recommended": []
            }
            
            # Generate appropriate use cases based on model characteristics
            if "classification" in model_type.lower():
                if performance >= 0.85:
                    use_cases["appropriate"].extend([
                        f"Production {domain} classification with human oversight",
                        "Automated screening and filtering applications",
                        "Decision support systems for expert users"
                    ])
                else:
                    use_cases["appropriate"].extend([
                        f"Research and development in {domain}",
                        "Proof-of-concept applications with validation",
                        "Educational and training purposes"
                    ])
            
            # Generate inappropriate use cases
            use_cases["inappropriate"].extend([
                "Safety-critical systems without human oversight",
                "Applications requiring 100% accuracy or reliability", 
                "Use on data significantly different from training distribution"
            ])
            
            if performance < 0.8:
                use_cases["inappropriate"].extend([
                    "Production deployment without additional validation",
                    "High-stakes decision making without expert review"
                ])
            
            # High-impact domain restrictions
            if any(term in domain.lower() for term in ["healthcare", "medical", "legal", "finance"]):
                use_cases["inappropriate"].extend([
                    "Autonomous decision making in regulated domains",
                    "Replacement for professional expertise and judgment"
                ])
            
            use_cases["recommended"] = [
                "Pilot testing in controlled environments",
                "Integration with existing validation workflows",
                "Gradual rollout with performance monitoring"
            ]
            
            logger.info(f"Generated use cases for {model_type} model in {domain}")
            return use_cases
            
        except Exception as e:
            logger.error(f"Failed to generate use cases: {e}")
            return {
                "appropriate": ["Research and development applications"],
                "inappropriate": ["Production use without validation"],
                "recommended": ["Thorough testing before deployment"]
            }

    def generate_training_recommendations(self, model_card: ModelCard, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for model improvement."""
        try:
            recommendations = {
                "data_improvements": [],
                "architecture_suggestions": [],
                "training_optimizations": [],
                "evaluation_enhancements": []
            }
            
            # Analyze current performance
            metrics = model_card.evaluation_results
            performance_level = self._analyze_performance_level(metrics)
            
            if performance_level < 0.8:
                recommendations["data_improvements"].extend([
                    "Increase training dataset size and diversity",
                    "Improve data quality and label consistency",
                    "Add data augmentation techniques",
                    "Balance class distributions"
                ])
                
                recommendations["training_optimizations"].extend([
                    "Experiment with different learning rates and schedules",
                    "Try ensemble methods for improved performance",
                    "Implement cross-validation for robust evaluation",
                    "Consider transfer learning from pre-trained models"
                ])
            
            elif performance_level < 0.9:
                recommendations["architecture_suggestions"].extend([
                    "Explore advanced architectures for the domain",
                    "Implement attention mechanisms if applicable",
                    "Optimize hyperparameters systematically",
                    "Consider multi-task learning approaches"
                ])
            
            else:
                recommendations["evaluation_enhancements"].extend([
                    "Add robustness testing with adversarial examples",
                    "Evaluate on diverse real-world test sets",
                    "Implement continuous evaluation pipelines",
                    "Measure fairness across demographic groups"
                ])
            
            # Always recommend monitoring and maintenance
            recommendations["training_optimizations"].append("Implement model versioning and experiment tracking")
            recommendations["evaluation_enhancements"].append("Set up automated performance monitoring")
            
            logger.info(f"Generated training recommendations based on {performance_level:.2f} performance level")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate training recommendations: {e}")
            return {
                "data_improvements": ["Increase data quality and diversity"],
                "training_optimizations": ["Systematic hyperparameter tuning"],
                "evaluation_enhancements": ["Comprehensive testing protocols"]
            }

    def _infer_model_type(self, model_card: ModelCard, context: Dict[str, Any]) -> str:
        """Infer the type of model from available information."""
        try:
            # Check model name and description for clues
            name_lower = model_card.model_details.name.lower()
            desc_lower = (model_card.model_details.description or "").lower()
            
            # Check for common model types
            if any(term in name_lower for term in ["classifier", "classification"]):
                return "classification"
            elif any(term in name_lower for term in ["regression", "regressor"]):
                return "regression"
            elif any(term in name_lower for term in ["generation", "generative", "gpt", "llm"]):
                return "text generation"
            elif any(term in name_lower for term in ["detection", "object", "yolo"]):
                return "object detection"
            elif any(term in name_lower for term in ["segmentation", "semantic"]):
                return "segmentation"
            elif any(term in desc_lower for term in ["sentiment", "classification"]):
                return "classification"
            elif any(term in desc_lower for term in ["predict", "forecast"]):
                return "regression"
            
            # Check metrics for clues
            metric_names = [m.name.lower() for m in model_card.evaluation_results]
            if any("accuracy" in name or "f1" in name for name in metric_names):
                return "classification"
            elif any("mse" in name or "mae" in name or "rmse" in name for name in metric_names):
                return "regression"
            elif any("bleu" in name or "rouge" in name for name in metric_names):
                return "text generation"
            
            return "machine learning"
            
        except Exception:
            return "machine learning"

    def _infer_domain(self, model_card: ModelCard, context: Dict[str, Any]) -> str:
        """Infer the application domain from available information."""
        try:
            # Check model name, description, and datasets for domain clues
            text_to_analyze = " ".join([
                model_card.model_details.name.lower(),
                (model_card.model_details.description or "").lower(),
                " ".join(model_card.model_details.datasets).lower(),
                " ".join(model_card.training_details.training_data).lower()
            ])
            
            domain_keywords = {
                "natural language processing": ["nlp", "text", "language", "sentiment", "translation"],
                "computer vision": ["vision", "image", "detection", "segmentation", "visual"],
                "healthcare": ["medical", "health", "clinical", "diagnosis", "patient"],
                "finance": ["financial", "trading", "credit", "fraud", "banking"],
                "e-commerce": ["product", "recommendation", "shopping", "retail"],
                "social media": ["social", "tweet", "post", "engagement"],
                "cybersecurity": ["security", "threat", "anomaly", "malware"],
                "autonomous systems": ["autonomous", "self-driving", "robotics"],
                "manufacturing": ["manufacturing", "quality", "defect", "production"]
            }
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in text_to_analyze for keyword in keywords):
                    return domain
            
            return "general purpose"
            
        except Exception:
            return "general purpose"

    def _summarize_performance(self, metrics: List[PerformanceMetric]) -> str:
        """Summarize model performance from metrics."""
        if not metrics:
            return "Performance metrics are being evaluated."
        
        try:
            # Find key performance indicators
            key_metrics = {}
            for metric in metrics:
                name = metric.name.lower()
                if any(key in name for key in ["accuracy", "f1", "auc", "precision", "recall"]):
                    key_metrics[name] = metric.value
            
            if not key_metrics:
                return f"Evaluated on {len(metrics)} metrics with comprehensive performance analysis."
            
            # Create performance summary
            best_metric = max(key_metrics.items(), key=lambda x: x[1])
            avg_performance = sum(key_metrics.values()) / len(key_metrics)
            
            performance_level = "excellent" if avg_performance >= 0.9 else \
                              "good" if avg_performance >= 0.8 else \
                              "moderate" if avg_performance >= 0.7 else "developing"
            
            return f"Achieves {performance_level} performance with {best_metric[0]} of {best_metric[1]:.3f}."
            
        except Exception:
            return "Demonstrates measurable performance across evaluation metrics."

    def _generate_use_cases(self, model_type: str, domain: str, performance_summary: str) -> Dict[str, str]:
        """Generate use case descriptions based on model characteristics."""
        strengths_map = {
            "classification": "categorization and decision support tasks",
            "regression": "numerical prediction and forecasting",
            "text generation": "content creation and language understanding",
            "object detection": "visual recognition and localization",
            "segmentation": "precise boundary detection and analysis"
        }
        
        applications_map = {
            "natural language processing": "text analysis, content moderation, and language applications",
            "computer vision": "image analysis, visual inspection, and automated monitoring",
            "healthcare": "clinical decision support and diagnostic assistance",
            "finance": "risk assessment, fraud detection, and market analysis",
            "e-commerce": "product recommendations and customer experience optimization"
        }
        
        return {
            "strengths": strengths_map.get(model_type, "pattern recognition and data analysis"),
            "applications": applications_map.get(domain, "data-driven decision making and automation")
        }

    def _analyze_performance_level(self, metrics: List[PerformanceMetric]) -> float:
        """Analyze overall performance level from metrics."""
        if not metrics:
            return 0.5  # Neutral performance when no metrics available
        
        try:
            # Weight different types of metrics
            weighted_scores = []
            for metric in metrics:
                name = metric.name.lower()
                value = metric.value
                
                # Apply weights based on metric importance
                if "accuracy" in name or "f1" in name:
                    weighted_scores.append(value * 1.0)  # High importance
                elif "precision" in name or "recall" in name:
                    weighted_scores.append(value * 0.8)  # Medium importance
                elif "auc" in name or "roc" in name:
                    weighted_scores.append(value * 0.9)  # High importance
                else:
                    weighted_scores.append(value * 0.5)  # Lower importance
            
            return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.5
            
        except Exception:
            return 0.5

    async def enhance_model_card_async(self, model_card: ModelCard, context: Dict[str, Any]) -> ModelCard:
        """Asynchronously enhance a model card with AI-generated content."""
        try:
            logger.info(f"Starting AI enhancement for {model_card.model_details.name}")
            
            # Run AI enhancements in parallel
            tasks = [
                self._enhance_description_async(model_card, context),
                self._enhance_limitations_async(model_card, context),
                self._enhance_ethical_considerations_async(model_card, context),
                self._enhance_use_cases_async(model_card, context)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Apply successful enhancements
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"AI enhancement task {i} failed: {result}")
                else:
                    logger.debug(f"AI enhancement task {i} completed successfully")
            
            # Add metadata about AI enhancement
            model_card.metadata["ai_enhanced"] = True
            model_card.metadata["ai_enhancement_timestamp"] = datetime.now().isoformat()
            model_card.metadata["ai_model"] = self.model_name
            
            logger.info(f"AI enhancement completed for {model_card.model_details.name}")
            return model_card
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return model_card

    async def _enhance_description_async(self, model_card: ModelCard, context: Dict[str, Any]) -> None:
        """Asynchronously enhance model description."""
        if not model_card.model_details.description:
            model_card.model_details.description = self.generate_description(model_card, context)

    async def _enhance_limitations_async(self, model_card: ModelCard, context: Dict[str, Any]) -> None:
        """Asynchronously enhance model limitations."""
        if not model_card.limitations.known_limitations:
            limitations = self.generate_limitations(model_card, context)
            model_card.limitations.known_limitations.extend(limitations)

    async def _enhance_ethical_considerations_async(self, model_card: ModelCard, context: Dict[str, Any]) -> None:
        """Asynchronously enhance ethical considerations."""
        ethical = self.generate_ethical_considerations(model_card, context)
        model_card.ethical_considerations.bias_risks.extend(ethical.get("bias_risks", []))
        model_card.ethical_considerations.bias_mitigation.extend(ethical.get("mitigation_strategies", []))

    async def _enhance_use_cases_async(self, model_card: ModelCard, context: Dict[str, Any]) -> None:
        """Asynchronously enhance use cases."""
        use_cases = self.generate_use_cases(model_card, context)
        model_card.limitations.out_of_scope_uses.extend(use_cases.get("inappropriate", []))
        model_card.limitations.recommendations.extend(use_cases.get("recommended", []))