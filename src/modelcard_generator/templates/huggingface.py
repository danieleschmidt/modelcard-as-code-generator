"""
Hugging Face model card template.

Implements the Hugging Face model card format and structure
following their community standards and requirements.
"""

from typing import Dict, Any, List
from jinja2 import Template as Jinja2Template

from .registry import Template
from ..core.model_card import ModelCard
from ..core.config import CardConfig


class HuggingFaceTemplate(Template):
    """Template for Hugging Face model cards."""
    
    def __init__(self):
        super().__init__(
            name="huggingface",
            description="Hugging Face community model card format"
        )
        self.markdown_template = self._get_markdown_template()
    
    def render(self, model_card: ModelCard, output_format: str = "markdown") -> str:
        """Render model card in Hugging Face format."""
        if output_format == "markdown":
            return self._render_markdown(model_card)
        elif output_format == "yaml":
            return self._render_yaml_frontmatter(model_card)
        else:
            raise ValueError(f"Unsupported output format for Hugging Face: {output_format}")
    
    def enhance_model_card(
        self, 
        model_card: ModelCard, 
        collected_data: Dict[str, Any], 
        config: CardConfig
    ) -> None:
        """Enhance model card with Hugging Face specific content."""
        # Add Hugging Face specific sections
        self._add_model_details_section(model_card, collected_data)
        self._add_intended_uses_section(model_card, collected_data)
        self._add_training_details_section(model_card, collected_data)
        self._add_evaluation_section(model_card, collected_data)
        
        if config.include_ethical_considerations:
            self._add_ethical_considerations_section(model_card, collected_data)
        
        if config.include_limitations:
            self._add_limitations_section(model_card, collected_data)
    
    def get_required_sections(self) -> List[str]:
        """Get required sections for Hugging Face format."""
        return [
            "model_details",
            "intended_use",
            "training_data"
        ]
    
    def get_optional_sections(self) -> List[str]:
        """Get optional sections for Hugging Face format."""
        return [
            "metrics",
            "evaluation_data",
            "ethical_considerations",
            "caveats_and_recommendations",
            "bias_analysis",
            "environmental_impact"
        ]
    
    def _render_markdown(self, model_card: ModelCard) -> str:
        """Render model card as Markdown."""
        template = Jinja2Template(self.markdown_template)
        
        # Prepare template context
        context = {
            "model_card": model_card,
            "model_details": model_card.model_details,
            "intended_use": model_card.intended_use,
            "training_data": model_card.training_data,
            "evaluation_data": model_card.evaluation_data,
            "metrics": model_card.metrics,
            "ethical_considerations": model_card.ethical_considerations,
            "caveats_and_recommendations": model_card.caveats_and_recommendations,
            "custom_sections": model_card.custom_sections,
        }
        
        return template.render(**context)
    
    def _render_yaml_frontmatter(self, model_card: ModelCard) -> str:
        """Render YAML frontmatter for Hugging Face Hub."""
        yaml_data = {
            "language": model_card.model_details.languages or ["en"],
            "license": model_card.model_details.license or "apache-2.0",
            "tags": model_card.model_details.tags or [],
            "datasets": [ds.name for ds in model_card.training_data + model_card.evaluation_data],
            "metrics": [metric.name for metric in model_card.metrics],
        }
        
        # Add base model if specified
        if model_card.model_details.base_model:
            yaml_data["base_model"] = model_card.model_details.base_model
        
        # Add pipeline tag based on model type
        if model_card.model_details.architecture:
            yaml_data["pipeline_tag"] = self._get_pipeline_tag(model_card.model_details.architecture)
        
        import yaml
        return yaml.dump(yaml_data, default_flow_style=False)
    
    def _add_model_details_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add model details section."""
        if not model_card.model_details.description and 'config' in collected_data:
            config_data = collected_data['config']
            model_card.model_details.description = config_data.get('model', {}).get('description', 
                f"A {model_card.model_details.architecture or 'machine learning'} model")
        
        # Extract architecture information
        if 'config' in collected_data:
            config_data = collected_data['config']
            model_info = config_data.get('model', {})
            
            if not model_card.model_details.architecture:
                model_card.model_details.architecture = model_info.get('architecture') or model_info.get('model_type')
            
            if not model_card.model_details.parameters:
                model_card.model_details.parameters = model_info.get('num_parameters') or model_info.get('parameters')
    
    def _add_intended_uses_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add intended uses section."""
        if not model_card.intended_use:
            # Infer intended use from model type/architecture
            architecture = model_card.model_details.architecture or ""
            
            if "classification" in architecture.lower():
                intended_use = {
                    "primary_use": "Text classification",
                    "primary_users": "Researchers and developers",
                    "out_of_scope": "This model should not be used for making decisions that affect people's lives or safety."
                }
            elif "generation" in architecture.lower() or "gpt" in architecture.lower():
                intended_use = {
                    "primary_use": "Text generation",
                    "primary_users": "Researchers and developers",
                    "out_of_scope": "This model should not be used to generate harmful or misleading content."
                }
            else:
                intended_use = {
                    "primary_use": "Machine learning research and development",
                    "primary_users": "Researchers and developers",
                    "out_of_scope": "Use outside the intended research context is not recommended."
                }
            
            model_card.intended_use = intended_use
    
    def _add_training_details_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add training details section."""
        training_details = {}
        
        if 'training' in collected_data:
            training_data = collected_data['training']
            
            # Extract hyperparameters
            if 'hyperparameters' in training_data:
                training_details["hyperparameters"] = training_data['hyperparameters']
            
            # Extract training time
            if 'training_time' in training_data:
                training_details["training_time"] = training_data['training_time']
            
            # Extract final metrics
            if 'final_metrics' in training_data:
                training_details["final_metrics"] = training_data['final_metrics']
        
        if training_details:
            model_card.add_section("training_details", training_details)
    
    def _add_evaluation_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add evaluation section."""
        if 'evaluation' in collected_data:
            eval_data = collected_data['evaluation']
            
            evaluation_section = {
                "testing_data": "Standard evaluation datasets",
                "testing_factors": "Various demographic and linguistic factors",
                "testing_results": eval_data.get('metrics', {})
            }
            
            model_card.quantitative_analysis = evaluation_section
    
    def _add_ethical_considerations_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add ethical considerations section."""
        if not model_card.ethical_considerations:
            ethical_considerations = {
                "human_life": "This model should not be used in applications that directly impact human life or safety without proper validation.",
                "mitigations": "Users should be aware of potential biases and limitations of the model.",
                "risks_and_harms": "Potential risks include biased outputs and misuse for generating misleading information.",
                "use_cases": "Intended for research and development purposes with appropriate oversight."
            }
            
            model_card.ethical_considerations = ethical_considerations
    
    def _add_limitations_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add limitations and recommendations section."""
        if not model_card.caveats_and_recommendations:
            limitations = {
                "limitations": "This model may have biases present in the training data and should be used with caution.",
                "recommendations": "Users should evaluate the model's performance on their specific use case and implement appropriate safeguards."
            }
            
            model_card.caveats_and_recommendations = limitations
    
    def _get_pipeline_tag(self, architecture: str) -> str:
        """Get Hugging Face pipeline tag based on architecture."""
        architecture_lower = architecture.lower()
        
        if "classification" in architecture_lower:
            return "text-classification"
        elif "generation" in architecture_lower or "gpt" in architecture_lower:
            return "text-generation"
        elif "translation" in architecture_lower:
            return "translation"
        elif "summarization" in architecture_lower:
            return "summarization"
        elif "question" in architecture_lower:
            return "question-answering"
        elif "fill" in architecture_lower or "mask" in architecture_lower:
            return "fill-mask"
        else:
            return "text-classification"  # Default
    
    def _get_markdown_template(self) -> str:
        """Get the Markdown template for Hugging Face model cards."""
        return '''---
{{ model_card.metadata.get('yaml_frontmatter', '') }}
---

# {{ model_details.name }}

{{ model_details.description or "Model description not available." }}

## Model Details

### Model Description

- **Developed by:** {{ model_details.get('developers', 'Not specified') }}
- **Model type:** {{ model_details.architecture or 'Not specified' }}
- **Language(s) (NLP):** {{ model_details.languages | join(', ') if model_details.languages else 'Not specified' }}
- **License:** {{ model_details.license or 'Not specified' }}
- **Finetuned from model:** {{ model_details.finetuned_from or model_details.base_model or 'Not specified' }}

### Model Sources

- **Repository:** {{ model_card.metadata.get('repository', 'Not specified') }}
- **Paper:** {{ model_card.metadata.get('paper', 'Not specified') }}
- **Demo:** {{ model_card.metadata.get('demo', 'Not specified') }}

## Uses

### Direct Use

{{ intended_use.get('primary_use', 'Primary use case not specified.') }}

### Downstream Use

{{ intended_use.get('downstream_use', 'Downstream use cases not specified.') }}

### Out-of-Scope Use

{{ intended_use.get('out_of_scope', 'Out-of-scope uses not specified.') }}

## Bias, Risks, and Limitations

{{ ethical_considerations.get('risks_and_harms', 'Risks and limitations not specified.') }}

### Recommendations

{{ caveats_and_recommendations.get('recommendations', 'No specific recommendations provided.') }}

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("{{ model_details.name }}")
model = AutoModel.from_pretrained("{{ model_details.name }}")
```

## Training Details

### Training Data

{% for dataset in training_data %}
- **{{ dataset.name }}**: {{ dataset.description or 'No description available' }}
{% endfor %}

### Training Procedure

{% if model_card.get_section('training_details') %}
{% set training_details = model_card.get_section('training_details') %}
#### Preprocessing

{{ training_details.get('preprocessing', 'Preprocessing details not specified.') }}

#### Training Hyperparameters

{% for param, value in training_details.get('hyperparameters', {}).items() %}
- **{{ param }}:** {{ value }}
{% endfor %}

#### Speeds, Sizes, Times

{% if training_details.get('training_time') %}
- **Training time:** {{ training_details.training_time }} seconds
{% endif %}
{% endif %}

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

{% for dataset in evaluation_data %}
- **{{ dataset.name }}**: {{ dataset.description or 'No description available' }}
{% endfor %}

#### Metrics

{% for metric in metrics %}
- **{{ metric.name }}**: {{ metric.description or 'Performance metric' }}
{% endfor %}

### Results

{% for metric in metrics %}
- **{{ metric.name }}**: {{ metric.value }}{% if metric.unit %} {{ metric.unit }}{% endif %}
{% endfor %}

#### Summary

{{ model_card.quantitative_analysis.get('summary', 'No evaluation summary provided.') }}

## Model Examination

{{ model_card.get_section('model_examination', 'Model examination details not provided.') }}

## Environmental Impact

{% if model_card.get_section('environmental_impact') %}
{% set env_impact = model_card.get_section('environmental_impact') %}
- **Hardware Type:** {{ env_impact.get('hardware_type', 'Not specified') }}
- **Hours used:** {{ env_impact.get('hours_used', 'Not specified') }}
- **Cloud Provider:** {{ env_impact.get('cloud_provider', 'Not specified') }}
- **Compute Region:** {{ env_impact.get('compute_region', 'Not specified') }}
- **Carbon Emitted:** {{ env_impact.get('carbon_emitted', 'Not specified') }}
{% else %}
Environmental impact information not provided.
{% endif %}

## Technical Specifications

### Model Architecture and Objective

{{ model_details.architecture or 'Architecture details not specified.' }}

### Compute Infrastructure

#### Hardware

{{ model_card.get_section('compute_infrastructure', {}).get('hardware', 'Hardware details not specified.') }}

#### Software

{{ model_card.get_section('compute_infrastructure', {}).get('software', 'Software details not specified.') }}

## Citation

```bibtex
@misc{{{ model_details.name.lower().replace(' ', '_').replace('-', '_') }},
  title = { {{- model_details.name -}} },
  author = { {{- model_card.metadata.get('authors', 'Not specified') -}} },
  year = { {{- model_card.metadata.get('year', 'Not specified') -}} },
  url = { {{- model_card.metadata.get('url', 'Not specified') -}} }
}
```

## Glossary

{{ model_card.get_section('glossary', 'No glossary provided.') }}

## More Information

{{ model_card.get_section('more_information', 'No additional information provided.') }}

## Model Card Authors

{{ model_card.metadata.get('model_card_authors', 'Model card authors not specified.') }}

## Model Card Contact

{{ model_card.metadata.get('model_card_contact', 'Contact information not provided.') }}
'''