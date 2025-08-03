"""
Google Model Cards template.

Implements Google's Model Cards format following their specification
for structured model documentation.
"""

from typing import Dict, Any, List
import json
from datetime import datetime

from .registry import Template
from ..core.model_card import ModelCard
from ..core.config import CardConfig


class GoogleModelCardTemplate(Template):
    """Template for Google Model Cards format."""
    
    def __init__(self):
        super().__init__(
            name="google",
            description="Google Model Cards structured format"
        )
    
    def render(self, model_card: ModelCard, output_format: str = "json") -> str:
        """Render model card in Google format."""
        if output_format == "json":
            return self._render_json(model_card)
        elif output_format == "markdown":
            return self._render_markdown(model_card)
        else:
            raise ValueError(f"Unsupported output format for Google Model Cards: {output_format}")
    
    def enhance_model_card(
        self, 
        model_card: ModelCard, 
        collected_data: Dict[str, Any], 
        config: CardConfig
    ) -> None:
        """Enhance model card with Google Model Cards specific structure."""
        # Add Google-specific structured sections
        self._add_model_details_section(model_card, collected_data)
        self._add_model_parameters_section(model_card, collected_data)
        self._add_quantitative_analysis_section(model_card, collected_data)
        self._add_considerations_section(model_card, collected_data)
        
        # Add metadata following Google's schema
        self._add_google_metadata(model_card)
    
    def get_required_sections(self) -> List[str]:
        """Get required sections for Google Model Cards."""
        return [
            "model_details",
            "model_parameters", 
            "quantitative_analysis"
        ]
    
    def get_optional_sections(self) -> List[str]:
        """Get optional sections for Google Model Cards."""
        return [
            "considerations",
            "graphics",
            "references"
        ]
    
    def _render_json(self, model_card: ModelCard) -> str:
        """Render model card as structured JSON following Google's schema."""
        google_format = {
            "schema_version": "0.0.2",
            "model_details": self._build_model_details(model_card),
            "model_parameters": self._build_model_parameters(model_card),
            "quantitative_analysis": self._build_quantitative_analysis(model_card),
            "considerations": self._build_considerations(model_card),
        }
        
        # Add optional sections if they exist
        if model_card.get_section('graphics'):
            google_format["graphics"] = model_card.get_section('graphics')
        
        if model_card.get_section('references'):
            google_format["references"] = model_card.get_section('references')
        
        return json.dumps(google_format, indent=2, ensure_ascii=False)
    
    def _render_markdown(self, model_card: ModelCard) -> str:
        """Render model card as Markdown with Google structure."""
        sections = []
        
        # Title
        sections.append(f"# {model_card.model_details.name}")
        sections.append("")
        
        # Model Details
        sections.append("## Model Details")
        model_details = self._build_model_details(model_card)
        
        sections.append(f"**Name:** {model_details.get('name', 'Not specified')}")
        sections.append(f"**Version:** {model_details.get('version', 'Not specified')}")
        sections.append(f"**Date:** {model_details.get('date', 'Not specified')}")
        sections.append(f"**Type:** {model_details.get('type', 'Not specified')}")
        sections.append(f"**Paper:** {model_details.get('paper', 'Not specified')}")
        sections.append(f"**Citation:** {model_details.get('citation', 'Not specified')}")
        sections.append(f"**License:** {model_details.get('license', 'Not specified')}")
        sections.append("")
        
        # Model Parameters
        sections.append("## Model Parameters")
        model_params = self._build_model_parameters(model_card)
        
        if 'model_architecture' in model_params:
            sections.append(f"**Architecture:** {model_params['model_architecture']}")
        if 'input_format' in model_params:
            sections.append(f"**Input Format:** {model_params['input_format']}")
        if 'output_format' in model_params:
            sections.append(f"**Output Format:** {model_params['output_format']}")
        sections.append("")
        
        # Quantitative Analysis
        sections.append("## Quantitative Analysis")
        quant_analysis = self._build_quantitative_analysis(model_card)
        
        if 'performance_metrics' in quant_analysis:
            sections.append("### Performance Metrics")
            for metric in quant_analysis['performance_metrics']:
                metric_name = metric.get('type', 'Unknown metric')
                metric_value = metric.get('value', 'Not specified')
                sections.append(f"- **{metric_name}:** {metric_value}")
            sections.append("")
        
        # Considerations
        sections.append("## Considerations")
        considerations = self._build_considerations(model_card)
        
        if 'users' in considerations:
            sections.append("### Users")
            sections.append(considerations['users'])
            sections.append("")
        
        if 'use_cases' in considerations:
            sections.append("### Use Cases")
            sections.append(considerations['use_cases'])
            sections.append("")
        
        if 'limitations' in considerations:
            sections.append("### Limitations")
            sections.append(considerations['limitations'])
            sections.append("")
        
        if 'ethical_considerations' in considerations:
            sections.append("### Ethical Considerations")
            sections.append(considerations['ethical_considerations'])
            sections.append("")
        
        return "\n".join(sections)
    
    def _build_model_details(self, model_card: ModelCard) -> Dict[str, Any]:
        """Build model details section in Google format."""
        details = {
            "name": model_card.model_details.name,
            "version": {
                "name": model_card.model_details.version or "1.0",
                "date": datetime.now().strftime("%Y-%m-%d")
            },
            "type": model_card.model_details.architecture or "machine_learning_model",
            "paper": model_card.metadata.get('paper', ''),
            "citation": model_card.metadata.get('citation', ''),
            "license": model_card.model_details.license or '',
            "path": model_card.metadata.get('model_path', ''),
        }
        
        # Add owners if available
        if model_card.metadata.get('owners'):
            details["owners"] = model_card.metadata['owners']
        
        return details
    
    def _build_model_parameters(self, model_card: ModelCard) -> Dict[str, Any]:
        """Build model parameters section in Google format."""
        parameters = {}
        
        # Model architecture
        if model_card.model_details.architecture:
            parameters["model_architecture"] = model_card.model_details.architecture
        
        # Input/Output format
        if model_card.model_details.languages:
            parameters["input_format"] = f"Text in {', '.join(model_card.model_details.languages)}"
        
        # Data section
        if model_card.training_data or model_card.evaluation_data:
            data_info = {
                "training_data": [
                    {
                        "name": ds.name,
                        "description": ds.description or "",
                        "link": ds.url or ""
                    }
                    for ds in model_card.training_data
                ],
                "eval_data": [
                    {
                        "name": ds.name,
                        "description": ds.description or "",
                        "link": ds.url or ""
                    }
                    for ds in model_card.evaluation_data
                ]
            }
            parameters["data"] = data_info
        
        return parameters
    
    def _build_quantitative_analysis(self, model_card: ModelCard) -> Dict[str, Any]:
        """Build quantitative analysis section in Google format."""
        analysis = {}
        
        # Performance metrics
        if model_card.metrics:
            performance_metrics = []
            
            for metric in model_card.metrics:
                metric_entry = {
                    "type": metric.name,
                    "value": str(metric.value)
                }
                
                if metric.dataset:
                    metric_entry["slice"] = metric.dataset
                
                if metric.confidence_interval:
                    metric_entry["confidence_interval"] = {
                        "lower_bound": metric.confidence_interval[0],
                        "upper_bound": metric.confidence_interval[1]
                    }
                
                performance_metrics.append(metric_entry)
            
            analysis["performance_metrics"] = performance_metrics
        
        # Graphics (placeholder for visualization data)
        graphics = model_card.get_section('graphics')
        if graphics:
            analysis["graphics"] = graphics
        
        return analysis
    
    def _build_considerations(self, model_card: ModelCard) -> Dict[str, Any]:
        """Build considerations section in Google format."""
        considerations = {}
        
        # Users
        intended_use = model_card.intended_use
        if intended_use and intended_use.get('primary_users'):
            considerations["users"] = intended_use['primary_users']
        
        # Use cases
        if intended_use and intended_use.get('primary_use'):
            considerations["use_cases"] = intended_use['primary_use']
        
        # Limitations
        limitations = []
        if model_card.caveats_and_recommendations:
            if 'limitations' in model_card.caveats_and_recommendations:
                limitations.append(model_card.caveats_and_recommendations['limitations'])
        
        if intended_use and intended_use.get('out_of_scope'):
            limitations.append(f"Out of scope: {intended_use['out_of_scope']}")
        
        if limitations:
            considerations["limitations"] = "; ".join(limitations)
        
        # Ethical considerations
        if model_card.ethical_considerations:
            ethical_text = []
            for key, value in model_card.ethical_considerations.items():
                if isinstance(value, str):
                    ethical_text.append(f"{key}: {value}")
            
            if ethical_text:
                considerations["ethical_considerations"] = "; ".join(ethical_text)
        
        # Fairness assessment
        if model_card.get_section('fairness_assessment'):
            considerations["fairness_assessment"] = model_card.get_section('fairness_assessment')
        
        return considerations
    
    def _add_model_details_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add Google-specific model details."""
        # Enhance with configuration data
        if 'config' in collected_data:
            config_data = collected_data['config']
            model_info = config_data.get('model', {})
            
            # Set default values if not present
            if not model_card.model_details.description:
                model_card.model_details.description = (
                    f"A {model_card.model_details.architecture or 'machine learning'} model "
                    f"for {model_info.get('task', 'various tasks')}"
                )
    
    def _add_model_parameters_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add model parameters section."""
        parameters = {}
        
        if 'config' in collected_data:
            config_data = collected_data['config']
            
            # Extract model configuration
            model_config = config_data.get('model', {})
            training_config = config_data.get('training', {})
            
            if model_config:
                parameters.update(model_config)
            
            if training_config:
                parameters['training_parameters'] = training_config
        
        if parameters:
            model_card.add_section('model_parameters', parameters)
    
    def _add_quantitative_analysis_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add quantitative analysis section."""
        # This is already handled by the core model card metrics
        # But we can enhance with Google-specific structure
        if 'evaluation' in collected_data:
            eval_data = collected_data['evaluation']
            
            # Add evaluation methodology
            quant_analysis = {
                "evaluation_methodology": "Standard evaluation on held-out test sets",
                "metrics_description": "Performance measured using standard metrics for the task"
            }
            
            if eval_data.get('metadata'):
                quant_analysis.update(eval_data['metadata'])
            
            model_card.quantitative_analysis.update(quant_analysis)
    
    def _add_considerations_section(self, model_card: ModelCard, collected_data: Dict[str, Any]) -> None:
        """Add considerations section."""
        considerations = {}
        
        # Add default considerations if not present
        if not model_card.intended_use:
            considerations["use_cases"] = "General purpose machine learning applications"
            considerations["users"] = "Machine learning researchers and practitioners"
        
        # Add limitations
        if not model_card.caveats_and_recommendations:
            considerations["limitations"] = (
                "Model performance may vary across different datasets and domains. "
                "Users should evaluate the model on their specific use case."
            )
        
        if considerations:
            model_card.add_section('google_considerations', considerations)
    
    def _add_google_metadata(self, model_card: ModelCard) -> None:
        """Add Google Model Cards specific metadata."""
        model_card.metadata.update({
            "schema_version": "0.0.2",
            "model_card_version": "1.0",
            "date_created": datetime.now().isoformat(),
            "format": "google_model_cards"
        })