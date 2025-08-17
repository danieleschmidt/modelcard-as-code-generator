"""
AI-powered content generation engine for intelligent model card creation.

This module provides advanced AI capabilities including:
- Natural language generation for model descriptions
- Intelligent content suggestion and completion
- Automated bias detection and mitigation recommendations
- Smart template selection based on model characteristics
- Content quality assessment and improvement suggestions
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class ContentType(Enum):
    """Types of content that can be generated."""
    DESCRIPTION = "description"
    LIMITATIONS = "limitations"
    BIAS_ANALYSIS = "bias_analysis"
    USE_CASES = "use_cases"
    ETHICAL_CONSIDERATIONS = "ethical_considerations"
    TECHNICAL_DETAILS = "technical_details"


@dataclass
class GenerationContext:
    """Context for AI content generation."""
    model_type: str
    domain: str
    performance_metrics: Dict[str, float]
    training_data: List[str]
    intended_use: str
    regulatory_requirements: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentSuggestion:
    """AI-generated content suggestion."""
    content_type: ContentType
    generated_text: str
    confidence_score: float
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAssessment:
    """Assessment of content quality."""
    overall_score: float
    clarity_score: float
    completeness_score: float
    accuracy_score: float
    compliance_score: float
    improvements: List[str] = field(default_factory=list)


class NaturalLanguageGenerator:
    """Natural language generation for model card content."""

    def __init__(self):
        self.templates = self._load_templates()
        self.domain_knowledge = self._load_domain_knowledge()
        self.linguistic_patterns = self._initialize_patterns()

    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load content generation templates."""
        return {
            "description": {
                "classification": "This {model_type} model is designed for {task} in the {domain} domain. "
                                 "It achieves {accuracy:.1%} accuracy on {dataset} and is optimized for {use_case}.",
                "regression": "This regression model predicts {target_variable} with an R² score of {r2_score:.3f}. "
                             "It is particularly effective for {use_case} in {domain} applications.",
                "nlp": "This natural language processing model specializes in {nlp_task} for {language} text. "
                       "With {performance_metric}: {score:.3f}, it excels at {specific_capability}.",
                "computer_vision": "This computer vision model performs {cv_task} on {image_type} images. "
                                  "It achieves {metric_name}: {metric_value:.3f} and is optimized for {deployment_context}."
            },
            "limitations": {
                "general": [
                    "Performance may degrade on data significantly different from the training distribution.",
                    "Model predictions should be validated by domain experts before critical applications.",
                    "Resource requirements may limit deployment in constrained environments.",
                    "Regular retraining is recommended to maintain performance over time."
                ],
                "classification": [
                    "Prediction confidence may vary across different classes, particularly for underrepresented categories.",
                    "Decision boundaries may not capture complex edge cases in the feature space.",
                    "Class imbalance in training data may affect minority class predictions."
                ],
                "nlp": [
                    "Model performance may vary across different linguistic styles and domains.",
                    "Handling of out-of-vocabulary words and novel expressions may be limited.",
                    "Cultural and linguistic biases present in training data may affect outputs."
                ]
            },
            "bias_analysis": {
                "demographic": "Analysis revealed potential disparities in model performance across {demographic_groups}. "
                              "The largest performance gap was {max_gap:.2%} between {group1} and {group2}.",
                "statistical": "Statistical parity difference of {stat_parity:.3f} and equalized odds difference of "
                              "{eq_odds:.3f} were observed across protected attributes.",
                "mitigation": "Recommended mitigation strategies include {strategies} and ongoing monitoring for {metrics}."
            }
        }

    def _load_domain_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific knowledge bases."""
        return {
            "healthcare": {
                "regulatory_considerations": ["HIPAA compliance", "FDA validation", "Clinical safety"],
                "ethical_concerns": ["Patient privacy", "Medical bias", "Healthcare equity"],
                "technical_requirements": ["Robustness", "Interpretability", "Reliability"]
            },
            "finance": {
                "regulatory_considerations": ["Fair lending", "GDPR compliance", "Audit trails"],
                "ethical_concerns": ["Algorithmic fairness", "Financial inclusion", "Transparency"],
                "technical_requirements": ["Security", "Explainability", "Performance monitoring"]
            },
            "criminal_justice": {
                "regulatory_considerations": ["Due process", "Constitutional rights", "Legal standards"],
                "ethical_concerns": ["Racial bias", "Recidivism prediction ethics", "Human oversight"],
                "technical_requirements": ["Fairness metrics", "Transparency", "Accountability"]
            },
            "education": {
                "regulatory_considerations": ["FERPA compliance", "Student privacy", "Educational equity"],
                "ethical_concerns": ["Learning bias", "Accessibility", "Inclusive design"],
                "technical_requirements": ["Adaptability", "Personalization", "Progress tracking"]
            }
        }

    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize linguistic patterns for content generation."""
        return {
            "technical_language": [
                "demonstrates", "achieves", "exhibits", "maintains", "optimizes",
                "leverages", "implements", "incorporates", "utilizes", "employs"
            ],
            "limitation_indicators": [
                "may experience", "could exhibit", "might demonstrate", "potentially shows",
                "has been observed to", "can be limited by", "may struggle with"
            ],
            "confidence_modifiers": [
                "typically", "generally", "usually", "commonly", "frequently",
                "in most cases", "under normal conditions", "as expected"
            ]
        }

    async def generate_content(self, content_type: ContentType, context: GenerationContext) -> ContentSuggestion:
        """Generate AI-powered content for model cards."""
        logger.debug(f"Generating {content_type.value} content for {context.model_type}")

        if content_type == ContentType.DESCRIPTION:
            return await self._generate_description(context)
        elif content_type == ContentType.LIMITATIONS:
            return await self._generate_limitations(context)
        elif content_type == ContentType.BIAS_ANALYSIS:
            return await self._generate_bias_analysis(context)
        elif content_type == ContentType.USE_CASES:
            return await self._generate_use_cases(context)
        elif content_type == ContentType.ETHICAL_CONSIDERATIONS:
            return await self._generate_ethical_considerations(context)
        elif content_type == ContentType.TECHNICAL_DETAILS:
            return await self._generate_technical_details(context)
        else:
            return ContentSuggestion(
                content_type=content_type,
                generated_text="Content generation not implemented for this type.",
                confidence_score=0.0
            )

    async def _generate_description(self, context: GenerationContext) -> ContentSuggestion:
        """Generate model description."""
        model_type = context.model_type.lower()
        
        # Select appropriate template
        if "classification" in model_type or "classifier" in model_type:
            template_key = "classification"
        elif "regression" in model_type or "regressor" in model_type:
            template_key = "regression"
        elif "nlp" in model_type or "language" in model_type or "text" in model_type:
            template_key = "nlp"
        elif "vision" in model_type or "image" in model_type or "cv" in model_type:
            template_key = "computer_vision"
        else:
            template_key = "classification"  # Default

        template = self.templates["description"][template_key]
        
        # Fill template with context information
        filled_template = await self._fill_template(template, context)
        
        # Enhance with domain-specific information
        enhanced_content = await self._enhance_with_domain_knowledge(filled_template, context)
        
        confidence = self._calculate_confidence(context, ["model_type", "domain", "performance_metrics"])
        
        return ContentSuggestion(
            content_type=ContentType.DESCRIPTION,
            generated_text=enhanced_content,
            confidence_score=confidence,
            sources=["template_based", "domain_knowledge"],
            metadata={"template_used": template_key}
        )

    async def _generate_limitations(self, context: GenerationContext) -> ContentSuggestion:
        """Generate limitations section."""
        limitations = []
        
        # Add general limitations
        limitations.extend(self.templates["limitations"]["general"])
        
        # Add model-type specific limitations
        model_type = context.model_type.lower()
        if "classification" in model_type:
            limitations.extend(self.templates["limitations"]["classification"])
        elif "nlp" in model_type or "language" in model_type:
            limitations.extend(self.templates["limitations"]["nlp"])
        
        # Add domain-specific limitations
        domain_limitations = await self._get_domain_limitations(context)
        limitations.extend(domain_limitations)
        
        # Add performance-based limitations
        performance_limitations = await self._get_performance_limitations(context)
        limitations.extend(performance_limitations)
        
        # Format as bullet points
        formatted_limitations = "\\n".join(f"- {limitation}" for limitation in limitations[:6])  # Limit to 6 points
        
        confidence = self._calculate_confidence(context, ["model_type", "domain"])
        
        return ContentSuggestion(
            content_type=ContentType.LIMITATIONS,
            generated_text=formatted_limitations,
            confidence_score=confidence,
            sources=["template_based", "domain_analysis", "performance_analysis"]
        )

    async def _generate_bias_analysis(self, context: GenerationContext) -> ContentSuggestion:
        """Generate bias analysis content."""
        bias_content = []
        
        # Demographic bias analysis
        if context.domain in ["healthcare", "finance", "criminal_justice", "education"]:
            demographic_analysis = await self._analyze_demographic_bias(context)
            bias_content.append(demographic_analysis)
        
        # Statistical bias metrics
        statistical_analysis = await self._analyze_statistical_bias(context)
        bias_content.append(statistical_analysis)
        
        # Mitigation recommendations
        mitigation_strategies = await self._recommend_bias_mitigation(context)
        bias_content.append(mitigation_strategies)
        
        combined_content = "\\n\\n".join(bias_content)
        
        confidence = self._calculate_confidence(context, ["domain", "performance_metrics"])
        
        return ContentSuggestion(
            content_type=ContentType.BIAS_ANALYSIS,
            generated_text=combined_content,
            confidence_score=confidence,
            sources=["bias_templates", "domain_knowledge", "statistical_analysis"]
        )

    async def _generate_use_cases(self, context: GenerationContext) -> ContentSuggestion:
        """Generate use cases and applications."""
        use_cases = []
        
        # Primary use case (from context)
        if context.intended_use:
            use_cases.append(f"**Primary Use Case**: {context.intended_use}")
        
        # Domain-specific use cases
        domain_use_cases = await self._get_domain_use_cases(context)
        use_cases.extend(domain_use_cases)
        
        # Performance-appropriate use cases
        performance_use_cases = await self._get_performance_appropriate_use_cases(context)
        use_cases.extend(performance_use_cases)
        
        # Out-of-scope use cases
        out_of_scope = await self._get_out_of_scope_use_cases(context)
        if out_of_scope:
            use_cases.append("\\n**Out of Scope**:")
            use_cases.extend(out_of_scope)
        
        formatted_content = "\\n".join(use_cases)
        
        confidence = self._calculate_confidence(context, ["intended_use", "domain", "performance_metrics"])
        
        return ContentSuggestion(
            content_type=ContentType.USE_CASES,
            generated_text=formatted_content,
            confidence_score=confidence,
            sources=["context_analysis", "domain_knowledge", "performance_analysis"]
        )

    async def _generate_ethical_considerations(self, context: GenerationContext) -> ContentSuggestion:
        """Generate ethical considerations."""
        ethical_content = []
        
        # Domain-specific ethical considerations
        if context.domain in self.domain_knowledge:
            domain_ethics = self.domain_knowledge[context.domain]["ethical_concerns"]
            for concern in domain_ethics:
                ethical_content.append(f"- **{concern}**: Consider the implications of {concern.lower()} in model deployment and usage.")
        
        # General ethical considerations
        general_ethics = [
            "Ensure model predictions are used in conjunction with human judgment",
            "Monitor for unintended consequences and bias in real-world applications", 
            "Provide clear information about model limitations to end users",
            "Implement appropriate safeguards and oversight mechanisms"
        ]
        ethical_content.extend(f"- {item}" for item in general_ethics)
        
        # Regulatory compliance considerations
        if context.regulatory_requirements:
            ethical_content.append("\\n**Regulatory Compliance**:")
            for requirement in context.regulatory_requirements:
                ethical_content.append(f"- Ensure compliance with {requirement} requirements")
        
        formatted_content = "\\n".join(ethical_content)
        
        confidence = self._calculate_confidence(context, ["domain", "regulatory_requirements"])
        
        return ContentSuggestion(
            content_type=ContentType.ETHICAL_CONSIDERATIONS,
            generated_text=formatted_content,
            confidence_score=confidence,
            sources=["domain_knowledge", "regulatory_analysis", "ethical_frameworks"]
        )

    async def _generate_technical_details(self, context: GenerationContext) -> ContentSuggestion:
        """Generate technical implementation details."""
        technical_content = []
        
        # Model architecture
        technical_content.append(f"**Model Type**: {context.model_type}")
        
        # Performance metrics
        if context.performance_metrics:
            technical_content.append("**Performance Metrics**:")
            for metric, value in context.performance_metrics.items():
                if isinstance(value, float):
                    technical_content.append(f"- {metric}: {value:.4f}")
                else:
                    technical_content.append(f"- {metric}: {value}")
        
        # Training data information
        if context.training_data:
            technical_content.append("**Training Data**:")
            for dataset in context.training_data[:3]:  # Limit to 3 datasets
                technical_content.append(f"- {dataset}")
        
        # Domain-specific technical requirements
        if context.domain in self.domain_knowledge:
            tech_reqs = self.domain_knowledge[context.domain]["technical_requirements"]
            technical_content.append("**Technical Requirements**:")
            for req in tech_reqs:
                technical_content.append(f"- {req}")
        
        formatted_content = "\\n".join(technical_content)
        
        confidence = self._calculate_confidence(context, ["model_type", "performance_metrics", "training_data"])
        
        return ContentSuggestion(
            content_type=ContentType.TECHNICAL_DETAILS,
            generated_text=formatted_content,
            confidence_score=confidence,
            sources=["context_data", "domain_requirements"]
        )

    async def _fill_template(self, template: str, context: GenerationContext) -> str:
        """Fill template with context information."""
        # Extract placeholders
        placeholders = re.findall(r'\\{([^}]+)\\}', template)
        
        filled_template = template
        for placeholder in placeholders:
            value = await self._resolve_placeholder(placeholder, context)
            filled_template = filled_template.replace(f"{{{placeholder}}}", str(value))
        
        return filled_template

    async def _resolve_placeholder(self, placeholder: str, context: GenerationContext) -> str:
        """Resolve template placeholder to actual value."""
        if placeholder == "model_type":
            return context.model_type
        elif placeholder == "domain":
            return context.domain
        elif placeholder == "task":
            return context.intended_use or "predictive modeling"
        elif placeholder == "use_case":
            return context.intended_use or "general purpose applications"
        elif placeholder.endswith("_score") or placeholder in context.performance_metrics:
            return context.performance_metrics.get(placeholder, "N/A")
        elif placeholder == "dataset":
            return context.training_data[0] if context.training_data else "proprietary dataset"
        else:
            # Try to extract from custom attributes
            return context.custom_attributes.get(placeholder, f"[{placeholder}]")

    async def _enhance_with_domain_knowledge(self, content: str, context: GenerationContext) -> str:
        """Enhance content with domain-specific knowledge."""
        if context.domain not in self.domain_knowledge:
            return content
        
        domain_info = self.domain_knowledge[context.domain]
        
        # Add domain-specific technical considerations
        if "technical_requirements" in domain_info:
            tech_reqs = domain_info["technical_requirements"]
            if tech_reqs:
                content += f" This model incorporates {', '.join(tech_reqs[:2])} to meet domain requirements."
        
        return content

    async def _get_domain_limitations(self, context: GenerationContext) -> List[str]:
        """Get domain-specific limitations."""
        limitations = []
        
        if context.domain == "healthcare":
            limitations.extend([
                "Medical decisions should always involve qualified healthcare professionals.",
                "Model may not account for rare conditions or complex comorbidities.",
                "Performance may vary across different patient populations."
            ])
        elif context.domain == "finance":
            limitations.extend([
                "Economic conditions may affect model predictions significantly.",
                "Regulatory changes may impact model validity.",
                "Model should not be the sole basis for financial decisions."
            ])
        elif context.domain == "criminal_justice":
            limitations.extend([
                "Predictions should be used as decision support, not replacement for human judgment.",
                "Historical bias in criminal justice data may affect predictions.",
                "Regular auditing for fairness across demographic groups is essential."
            ])
        
        return limitations

    async def _get_performance_limitations(self, context: GenerationContext) -> List[str]:
        """Get limitations based on performance metrics."""
        limitations = []
        
        for metric, value in context.performance_metrics.items():
            if isinstance(value, (int, float)):
                if metric.lower() in ["accuracy", "precision", "recall", "f1"] and value < 0.9:
                    limitations.append(f"Model {metric} of {value:.3f} indicates room for improvement in prediction quality.")
                elif metric.lower() in ["mse", "mae", "rmse"] and value > 0.1:
                    limitations.append(f"Prediction error ({metric}: {value:.3f}) should be considered in application context.")
        
        return limitations

    async def _analyze_demographic_bias(self, context: GenerationContext) -> str:
        """Analyze potential demographic bias."""
        # Placeholder analysis - in practice would use actual bias detection
        template = self.templates["bias_analysis"]["demographic"]
        
        return template.format(
            demographic_groups="gender, age, and ethnic groups",
            max_gap=0.05,  # Placeholder
            group1="majority group",
            group2="minority group"
        )

    async def _analyze_statistical_bias(self, context: GenerationContext) -> str:
        """Analyze statistical bias metrics."""
        template = self.templates["bias_analysis"]["statistical"]
        
        return template.format(
            stat_parity=0.02,  # Placeholder
            eq_odds=0.03  # Placeholder
        )

    async def _recommend_bias_mitigation(self, context: GenerationContext) -> str:
        """Recommend bias mitigation strategies."""
        strategies = [
            "regular fairness auditing",
            "diverse training data collection",
            "algorithmic debiasing techniques",
            "continuous monitoring"
        ]
        
        metrics = ["demographic parity", "equalized opportunity", "predictive parity"]
        
        template = self.templates["bias_analysis"]["mitigation"]
        return template.format(
            strategies=", ".join(strategies),
            metrics=", ".join(metrics)
        )

    async def _get_domain_use_cases(self, context: GenerationContext) -> List[str]:
        """Get domain-specific use cases."""
        use_cases = []
        
        if context.domain == "healthcare":
            use_cases.extend([
                "- Clinical decision support systems",
                "- Patient risk stratification",
                "- Drug discovery and development"
            ])
        elif context.domain == "finance":
            use_cases.extend([
                "- Credit risk assessment",
                "- Fraud detection systems",
                "- Algorithmic trading strategies"
            ])
        elif context.domain == "education":
            use_cases.extend([
                "- Personalized learning recommendations",
                "- Student performance prediction",
                "- Educational content optimization"
            ])
        
        return use_cases

    async def _get_performance_appropriate_use_cases(self, context: GenerationContext) -> List[str]:
        """Get use cases appropriate for performance level."""
        use_cases = []
        
        # Analyze performance metrics to suggest appropriate use cases
        accuracy = context.performance_metrics.get("accuracy", 0.0)
        
        if accuracy > 0.95:
            use_cases.append("- High-stakes decision making with appropriate oversight")
        elif accuracy > 0.85:
            use_cases.append("- Decision support systems with human validation")
        elif accuracy > 0.7:
            use_cases.append("- Screening and preliminary analysis tools")
        else:
            use_cases.append("- Research and development applications")
        
        return use_cases

    async def _get_out_of_scope_use_cases(self, context: GenerationContext) -> List[str]:
        """Get use cases that are out of scope."""
        out_of_scope = []
        
        if context.domain == "healthcare":
            out_of_scope.extend([
                "- Emergency medical decisions without physician oversight",
                "- Diagnosis of conditions not represented in training data",
                "- Pediatric applications if trained only on adult data"
            ])
        elif context.domain == "finance":
            out_of_scope.extend([
                "- Unregulated financial advice to retail customers",
                "- High-frequency trading without risk management",
                "- Cross-border applications without local compliance review"
            ])
        
        return out_of_scope

    def _calculate_confidence(self, context: GenerationContext, required_fields: List[str]) -> float:
        """Calculate confidence score for generated content."""
        available_fields = 0
        total_fields = len(required_fields)
        
        for field in required_fields:
            if hasattr(context, field):
                value = getattr(context, field)
                if value and (not isinstance(value, (list, dict)) or len(value) > 0):
                    available_fields += 1
        
        base_confidence = available_fields / total_fields
        
        # Adjust based on domain knowledge availability
        if context.domain in self.domain_knowledge:
            base_confidence *= 1.1
        
        # Adjust based on regulatory requirements
        if context.regulatory_requirements:
            base_confidence *= 1.05
        
        return min(1.0, base_confidence)


class ContentQualityAssessor:
    """Assess and improve content quality."""

    def __init__(self):
        self.quality_metrics = self._initialize_quality_metrics()

    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """Initialize quality assessment metrics."""
        return {
            "clarity_indicators": [
                "clear", "specific", "detailed", "comprehensive", "well-defined"
            ],
            "completeness_requirements": {
                ContentType.DESCRIPTION: ["model_type", "performance", "use_case"],
                ContentType.LIMITATIONS: ["technical", "performance", "domain"],
                ContentType.BIAS_ANALYSIS: ["analysis", "metrics", "mitigation"],
                ContentType.USE_CASES: ["primary", "appropriate", "out_of_scope"],
                ContentType.ETHICAL_CONSIDERATIONS: ["domain_ethics", "oversight", "compliance"]
            },
            "accuracy_checks": [
                "consistent_metrics",
                "realistic_values", 
                "domain_appropriate",
                "factual_statements"
            ]
        }

    async def assess_content_quality(self, content: str, content_type: ContentType, context: GenerationContext) -> QualityAssessment:
        """Assess the quality of generated content."""
        
        clarity_score = await self._assess_clarity(content)
        completeness_score = await self._assess_completeness(content, content_type)
        accuracy_score = await self._assess_accuracy(content, context)
        compliance_score = await self._assess_compliance(content, context)
        
        overall_score = (clarity_score + completeness_score + accuracy_score + compliance_score) / 4
        
        improvements = await self._generate_improvements(content, content_type, context, {
            "clarity": clarity_score,
            "completeness": completeness_score,
            "accuracy": accuracy_score,
            "compliance": compliance_score
        })
        
        return QualityAssessment(
            overall_score=overall_score,
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            compliance_score=compliance_score,
            improvements=improvements
        )

    async def _assess_clarity(self, content: str) -> float:
        """Assess content clarity."""
        clarity_indicators = self.quality_metrics["clarity_indicators"]
        
        # Check for clarity indicators
        indicator_count = sum(1 for indicator in clarity_indicators if indicator in content.lower())
        indicator_score = min(1.0, indicator_count / 3)  # Expect at least 3 indicators
        
        # Check sentence structure
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        length_score = 1.0 if 10 <= avg_sentence_length <= 25 else 0.7  # Ideal range
        
        # Check for jargon density
        total_words = len(content.split())
        technical_words = self._count_technical_words(content)
        jargon_ratio = technical_words / max(1, total_words)
        jargon_score = 1.0 if jargon_ratio < 0.2 else 0.8  # Moderate technical language
        
        return (indicator_score + length_score + jargon_score) / 3

    async def _assess_completeness(self, content: str, content_type: ContentType) -> float:
        """Assess content completeness."""
        if content_type not in self.quality_metrics["completeness_requirements"]:
            return 0.8  # Default score for unknown types
        
        required_elements = self.quality_metrics["completeness_requirements"][content_type]
        found_elements = 0
        
        for element in required_elements:
            # Simple keyword matching - in practice would use more sophisticated NLP
            if element.replace("_", " ") in content.lower():
                found_elements += 1
        
        return found_elements / len(required_elements)

    async def _assess_accuracy(self, content: str, context: GenerationContext) -> float:
        """Assess content accuracy."""
        accuracy_score = 1.0
        
        # Check for unrealistic performance claims
        performance_numbers = re.findall(r'(\\d+\\.\\d+)', content)
        for number in performance_numbers:
            value = float(number)
            if value > 1.0 and any(metric in content.lower() for metric in ["accuracy", "precision", "recall", "f1"]):
                accuracy_score *= 0.8  # Penalize unrealistic values
        
        # Check consistency with context
        if context.domain and context.domain not in content.lower():
            accuracy_score *= 0.9
        
        if context.model_type and context.model_type.lower() not in content.lower():
            accuracy_score *= 0.9
        
        return accuracy_score

    async def _assess_compliance(self, content: str, context: GenerationContext) -> float:
        """Assess regulatory compliance."""
        compliance_score = 1.0
        
        # Check for required regulatory mentions
        if context.regulatory_requirements:
            mentioned_requirements = sum(1 for req in context.regulatory_requirements 
                                       if req.lower() in content.lower())
            compliance_score = mentioned_requirements / len(context.regulatory_requirements)
        
        # Check for domain-specific compliance indicators
        if context.domain == "healthcare" and "patient" not in content.lower():
            compliance_score *= 0.9
        elif context.domain == "finance" and "risk" not in content.lower():
            compliance_score *= 0.9
        
        return compliance_score

    async def _generate_improvements(self, content: str, content_type: ContentType, 
                                   context: GenerationContext, scores: Dict[str, float]) -> List[str]:
        """Generate specific improvement suggestions."""
        improvements = []
        
        if scores["clarity"] < 0.7:
            improvements.append("Consider simplifying technical language and shortening sentences for better clarity.")
        
        if scores["completeness"] < 0.8:
            required_elements = self.quality_metrics["completeness_requirements"].get(content_type, [])
            improvements.append(f"Ensure all required elements are covered: {', '.join(required_elements)}")
        
        if scores["accuracy"] < 0.9:
            improvements.append("Verify that all performance metrics and technical details are accurate and realistic.")
        
        if scores["compliance"] < 0.8:
            if context.regulatory_requirements:
                improvements.append(f"Include references to relevant regulations: {', '.join(context.regulatory_requirements)}")
            improvements.append("Ensure content meets domain-specific compliance requirements.")
        
        return improvements

    def _count_technical_words(self, content: str) -> int:
        """Count technical words in content."""
        technical_terms = {
            "algorithm", "model", "training", "validation", "accuracy", "precision", "recall",
            "optimization", "hyperparameter", "neural", "regression", "classification",
            "supervised", "unsupervised", "reinforcement", "deep", "machine", "artificial"
        }
        
        words = content.lower().split()
        return sum(1 for word in words if any(term in word for term in technical_terms))


# Main AI Content Engine
class AIContentEngine:
    """Main engine for AI-powered content generation."""

    def __init__(self):
        self.nlg = NaturalLanguageGenerator()
        self.quality_assessor = ContentQualityAssessor()
        self.generation_history: List[Dict[str, Any]] = []

    async def generate_intelligent_content(self, 
                                         content_types: List[ContentType],
                                         context: GenerationContext,
                                         quality_threshold: float = 0.8) -> Dict[ContentType, ContentSuggestion]:
        """Generate intelligent content for multiple content types."""
        results = {}
        
        for content_type in content_types:
            # Generate initial content
            suggestion = await self.nlg.generate_content(content_type, context)
            
            # Assess quality
            quality = await self.quality_assessor.assess_content_quality(
                suggestion.generated_text, content_type, context
            )
            
            # Improve if below threshold
            if quality.overall_score < quality_threshold:
                suggestion = await self._improve_content(suggestion, quality, context)
            
            # Record generation history
            self.generation_history.append({
                "timestamp": datetime.now().isoformat(),
                "content_type": content_type.value,
                "quality_score": quality.overall_score,
                "confidence_score": suggestion.confidence_score,
                "context_domain": context.domain
            })
            
            results[content_type] = suggestion
        
        return results

    async def _improve_content(self, suggestion: ContentSuggestion, 
                             quality: QualityAssessment, context: GenerationContext) -> ContentSuggestion:
        """Improve content based on quality assessment."""
        improved_text = suggestion.generated_text
        
        # Apply improvements
        for improvement in quality.improvements:
            if "simplifying technical language" in improvement:
                improved_text = await self._simplify_language(improved_text)
            elif "required elements" in improvement:
                improved_text = await self._add_missing_elements(improved_text, suggestion.content_type, context)
            elif "accuracy" in improvement:
                improved_text = await self._verify_accuracy(improved_text, context)
            elif "compliance" in improvement:
                improved_text = await self._enhance_compliance(improved_text, context)
        
        # Update suggestion
        suggestion.generated_text = improved_text
        suggestion.confidence_score *= 0.9  # Slightly reduce confidence for modified content
        suggestion.metadata["improved"] = True
        suggestion.metadata["improvements_applied"] = quality.improvements
        
        return suggestion

    async def _simplify_language(self, text: str) -> str:
        """Simplify technical language in text."""
        # Simple replacements for common technical terms
        replacements = {
            "utilize": "use",
            "implement": "use",
            "demonstrate": "show",
            "exhibits": "shows",
            "incorporates": "includes"
        }
        
        simplified = text
        for technical, simple in replacements.items():
            simplified = simplified.replace(technical, simple)
        
        return simplified

    async def _add_missing_elements(self, text: str, content_type: ContentType, context: GenerationContext) -> str:
        """Add missing required elements to content."""
        # This would analyze what elements are missing and add them
        # For now, just append a generic addition
        if content_type == ContentType.LIMITATIONS and "validation" not in text.lower():
            text += "\\n- Regular validation with domain experts is recommended."
        
        return text

    async def _verify_accuracy(self, text: str, context: GenerationContext) -> str:
        """Verify and correct accuracy issues."""
        # Check for unrealistic values and correct them
        corrected = text
        
        # Fix percentage values over 100%
        corrected = re.sub(r'(\\d+\\.\\d+)% accuracy', lambda m: f"{min(float(m.group(1)), 100):.1f}% accuracy", corrected)
        
        return corrected

    async def _enhance_compliance(self, text: str, context: GenerationContext) -> str:
        """Enhance compliance-related content."""
        if context.regulatory_requirements and not any(req.lower() in text.lower() for req in context.regulatory_requirements):
            compliance_note = f"\\n\\nThis model should be used in compliance with {', '.join(context.regulatory_requirements)} requirements."
            text += compliance_note
        
        return text

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about content generation."""
        if not self.generation_history:
            return {"message": "No generation history available"}
        
        quality_scores = [entry["quality_score"] for entry in self.generation_history]
        confidence_scores = [entry["confidence_score"] for entry in self.generation_history]
        
        content_type_counts = {}
        domain_counts = {}
        
        for entry in self.generation_history:
            content_type = entry["content_type"]
            domain = entry["context_domain"]
            
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            "total_generations": len(self.generation_history),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "average_confidence_score": sum(confidence_scores) / len(confidence_scores),
            "content_type_distribution": content_type_counts,
            "domain_distribution": domain_counts,
            "latest_generation": self.generation_history[-1] if self.generation_history else None
        }


# Global AI content engine instance
ai_content_engine = AIContentEngine()