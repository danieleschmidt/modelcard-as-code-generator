"""Template library for model cards."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
from ..core.models import ModelCard, CardConfig, CardFormat
from ..formats.huggingface import HuggingFaceCard


class Template(ABC):
    """Base template class."""
    
    def __init__(self, name: str, required_sections: Optional[List[str]] = None):
        self.name = name
        self.required_sections = required_sections or []
    
    @abstractmethod
    def create(self, **kwargs) -> ModelCard:
        """Create a model card using this template."""
        pass
    
    def format_list(self, items: List[str]) -> str:
        """Format a list of items for display."""
        return "\n".join(f"- {item}" for item in items)


class NLPClassificationTemplate(Template):
    """Template for NLP classification models."""
    
    def __init__(self):
        super().__init__(
            name="nlp_classification",
            required_sections=[
                "model_details",
                "intended_use", 
                "training_details",
                "evaluation_results",
                "limitations"
            ]
        )
    
    def create(self, **kwargs) -> HuggingFaceCard:
        """Create NLP classification model card."""
        card = HuggingFaceCard()
        
        # Set model details
        card.set_model_details(
            name=kwargs.get("model_name", "nlp-classifier"),
            languages=kwargs.get("languages", ["en"]),
            license=kwargs.get("license", "apache-2.0"),
            finetuned_from=kwargs.get("base_model", "bert-base-uncased"),
            tags=["text-classification", "nlp"]
        )
        
        # Set pipeline tag
        card.set_pipeline_tag("text-classification")
        
        # Set intended uses
        card.uses(
            direct_use=kwargs.get("direct_use", "Text classification for various domains"),
            downstream_use=kwargs.get("downstream_use", "Feature extraction for other NLP tasks"), 
            out_of_scope=kwargs.get("out_of_scope", "Not suitable for medical or legal decision making")
        )
        
        # Set training data
        datasets = kwargs.get("training_data", ["custom-dataset"])
        card.training_data(
            datasets=datasets,
            preprocessing=kwargs.get("preprocessing", "Text tokenization and normalization")
        )
        
        # Set evaluation metrics
        metrics = kwargs.get("metrics", {
            "accuracy": 0.92,
            "f1_macro": 0.89,
            "precision": 0.91,
            "recall": 0.88
        })
        card.evaluation(metrics)
        
        # Add common limitations
        limitations = kwargs.get("limitations", [
            "Performance may degrade on out-of-domain text",
            "May exhibit biases present in training data",
            "Not evaluated on non-English text"
        ])
        for limitation in limitations:
            card.add_limitation(limitation)
        
        # Add ethical considerations
        if kwargs.get("include_bias_analysis", True):
            card.ethical_considerations.bias_risks = [
                "May perpetuate demographic biases",
                "Performance may vary across different text domains"
            ]
            card.ethical_considerations.bias_mitigation = [
                "Regular bias audits recommended",
                "Evaluate on diverse test sets"
            ]
        
        # Add widget example
        example_text = kwargs.get("example_text", "This is a great product!")
        card.add_widget_example(example_text, "Example classification")
        
        return card


class ComputerVisionTemplate(Template):
    """Template for computer vision models."""
    
    def __init__(self):
        super().__init__(
            name="computer_vision",
            required_sections=[
                "model_details",
                "intended_use",
                "training_details", 
                "evaluation_results",
                "limitations"
            ]
        )
    
    def create(self, **kwargs) -> HuggingFaceCard:
        """Create computer vision model card."""
        card = HuggingFaceCard()
        
        # Set model details
        card.set_model_details(
            name=kwargs.get("model_name", "vision-model"),
            license=kwargs.get("license", "apache-2.0"),
            finetuned_from=kwargs.get("base_model", "resnet-50"),
            tags=["computer-vision", "image-classification"]
        )
        
        # Set pipeline tag
        card.set_pipeline_tag("image-classification")
        
        # Set intended uses
        card.uses(
            direct_use=kwargs.get("direct_use", "Image classification for computer vision tasks"),
            downstream_use=kwargs.get("downstream_use", "Feature extraction for other vision tasks"),
            out_of_scope=kwargs.get("out_of_scope", "Not suitable for medical diagnosis or safety-critical applications")
        )
        
        # Set training data
        datasets = kwargs.get("training_data", ["imagenet"])
        card.training_data(
            datasets=datasets,
            preprocessing=kwargs.get("preprocessing", "Image resizing, normalization, and augmentation")
        )
        
        # Set evaluation metrics
        metrics = kwargs.get("metrics", {
            "accuracy": 0.85,
            "top_5_accuracy": 0.95,
            "inference_time_ms": 15
        })
        card.evaluation(metrics)
        
        # Add limitations
        limitations = kwargs.get("limitations", [
            "Performance varies with image quality and lighting conditions",
            "May not generalize well to images outside training distribution",
            "Requires high-resolution images for best performance"
        ])
        for limitation in limitations:
            card.add_limitation(limitation)
        
        return card


class LLMTemplate(Template):
    """Template for Large Language Models."""
    
    def __init__(self):
        super().__init__(
            name="llm",
            required_sections=[
                "model_details",
                "intended_use",
                "training_details",
                "evaluation_results", 
                "limitations",
                "ethical_considerations"
            ]
        )
    
    def create(self, **kwargs) -> HuggingFaceCard:
        """Create LLM model card."""
        card = HuggingFaceCard()
        
        # Set model details
        card.set_model_details(
            name=kwargs.get("model_name", "custom-llm"),
            languages=kwargs.get("languages", ["en"]),
            license=kwargs.get("license", "apache-2.0"),
            finetuned_from=kwargs.get("base_model", "llama-2-7b"),
            tags=["text-generation", "llm", "causal-lm"]
        )
        
        # Set pipeline tag
        card.set_pipeline_tag("text-generation")
        
        # Set intended uses
        card.uses(
            direct_use=kwargs.get("direct_use", "Text generation for creative writing and assistance"),
            downstream_use=kwargs.get("downstream_use", "Fine-tuning for specific domains"),
            out_of_scope=kwargs.get("out_of_scope", "Not for generating harmful, biased, or factually incorrect content")
        )
        
        # Set training data
        datasets = kwargs.get("training_data", ["custom-text-corpus"])
        card.training_data(
            datasets=datasets,
            preprocessing=kwargs.get("preprocessing", "Text tokenization and sequence formatting")
        )
        
        # Set evaluation metrics
        metrics = kwargs.get("metrics", {
            "perplexity": 15.2,
            "bleu": 0.35,
            "rouge_l": 0.42
        })
        card.evaluation(metrics)
        
        # Add LLM-specific limitations
        limitations = kwargs.get("limitations", [
            "May generate biased or harmful content",
            "Can produce factually incorrect information",
            "Performance degrades on very long contexts",
            "May exhibit inconsistent behavior across different prompts"
        ])
        for limitation in limitations:
            card.add_limitation(limitation)
        
        # Add ethical considerations
        card.ethical_considerations.bias_risks = kwargs.get("bias_risks", [
            "May perpetuate societal biases present in training data",
            "Can generate discriminatory content",
            "May exhibit cultural or demographic biases"
        ])
        
        card.ethical_considerations.bias_mitigation = kwargs.get("bias_mitigation", [
            "Implement content filtering and safety measures",
            "Regular bias evaluation and red team testing",
            "Provide clear usage guidelines and warnings"
        ])
        
        # Add sensitive use cases
        card.limitations.sensitive_use_cases = kwargs.get("sensitive_use_cases", [
            "Medical advice generation",
            "Legal consultation",
            "Financial decision making",
            "Content moderation"
        ])
        
        # Add widget example
        example_text = kwargs.get("example_text", "Once upon a time")
        card.add_widget_example(example_text, "Story generation example")
        
        return card


class TemplateLibrary:
    """Central library for managing model card templates."""
    
    _templates: Dict[str, Template] = {}
    
    @classmethod
    def register(cls, template: Template) -> None:
        """Register a new template."""
        cls._templates[template.name] = template
    
    @classmethod
    def get(cls, name: str) -> Template:
        """Get a template by name."""
        if name not in cls._templates:
            raise ValueError(f"Template '{name}' not found. Available: {list(cls._templates.keys())}")
        return cls._templates[name]
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List all available template names."""
        return list(cls._templates.keys())
    
    @classmethod
    def create_card(cls, template_name: str, **kwargs) -> ModelCard:
        """Create a model card using a template."""
        template = cls.get(template_name)
        return template.create(**kwargs)


# Register default templates
TemplateLibrary.register(NLPClassificationTemplate())
TemplateLibrary.register(ComputerVisionTemplate())
TemplateLibrary.register(LLMTemplate())