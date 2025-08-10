"""Hugging Face model card format implementation."""

try:
    import yaml
except ImportError:
    yaml = None
from typing import Dict, List, Optional, Any
from ..core.models import ModelCard, CardConfig, CardFormat


class HuggingFaceCard(ModelCard):
    """Hugging Face specific model card implementation."""
    
    def __init__(self, config: Optional[CardConfig] = None):
        if config is None:
            config = CardConfig(format=CardFormat.HUGGINGFACE)
        super().__init__(config)
        
        # HF-specific metadata
        self.hf_metadata = {
            "license": None,
            "library_name": None,
            "tags": [],
            "datasets": [],
            "language": [],
            "pipeline_tag": None,
            "widget": []
        }
    
    def set_model_details(
        self,
        name: str,
        languages: Optional[List[str]] = None,
        license: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        **kwargs
    ) -> None:
        """Set model details section for HF format."""
        self.model_details.name = name
        
        if languages:
            self.model_details.language = languages
            self.hf_metadata["language"] = languages
        
        if license:
            self.model_details.license = license
            self.hf_metadata["license"] = license
        
        if finetuned_from:
            self.model_details.base_model = finetuned_from
        
        # Handle additional HF-specific fields
        for key, value in kwargs.items():
            if key in ["library_name", "pipeline_tag"]:
                self.hf_metadata[key] = value
            elif key == "tags":
                self.model_details.tags = value if isinstance(value, list) else [value]
                self.hf_metadata["tags"] = self.model_details.tags
    
    def uses(
        self,
        direct_use: Optional[str] = None,
        downstream_use: Optional[str] = None,
        out_of_scope: Optional[str] = None
    ) -> None:
        """Set intended uses section."""
        uses_content = []
        
        if direct_use:
            uses_content.append(f"**Direct Use**: {direct_use}")
        
        if downstream_use:
            uses_content.append(f"**Downstream Use**: {downstream_use}")
        
        if out_of_scope:
            uses_content.append(f"**Out-of-Scope Use**: {out_of_scope}")
            self.limitations.out_of_scope_uses.append(out_of_scope)
        
        self.intended_use = "\n\n".join(uses_content)
    
    def training_data(
        self,
        datasets: List[str],
        preprocessing: Optional[str] = None
    ) -> None:
        """Set training data section."""
        self.training_details.training_data = datasets
        self.model_details.datasets = datasets
        self.hf_metadata["datasets"] = datasets
        
        if preprocessing:
            self.training_details.preprocessing = preprocessing
    
    def evaluation(self, metrics: Dict[str, Any]) -> None:
        """Set evaluation section with metrics."""
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                # Handle complex metric structures
                value = metric_value.get("value", metric_value.get("score", 0))
                confidence_interval = metric_value.get("confidence_interval")
                dataset = metric_value.get("dataset")
                
                self.add_metric(
                    metric_name, 
                    float(value),
                    confidence_interval=confidence_interval,
                    dataset=dataset
                )
            else:
                self.add_metric(metric_name, float(metric_value))
    
    def add_widget_example(self, text: str, example_title: Optional[str] = None) -> None:
        """Add inference widget example."""
        widget_example = {"text": text}
        if example_title:
            widget_example["example_title"] = example_title
        
        self.hf_metadata["widget"].append(widget_example)
    
    def set_pipeline_tag(self, tag: str) -> None:
        """Set the pipeline tag for model categorization."""
        self.hf_metadata["pipeline_tag"] = tag
    
    def render(self, format_type: str = "markdown") -> str:
        """Render HF model card with YAML frontmatter."""
        if format_type == "markdown":
            return self._render_hf_markdown()
        else:
            return super().render(format_type)
    
    def _render_hf_markdown(self) -> str:
        """Render as HF-compatible markdown with YAML frontmatter."""
        # Build YAML frontmatter
        frontmatter = {}
        
        if self.hf_metadata["license"]:
            frontmatter["license"] = self.hf_metadata["license"]
        
        if self.hf_metadata["library_name"]:
            frontmatter["library_name"] = self.hf_metadata["library_name"]
        
        if self.hf_metadata["tags"]:
            frontmatter["tags"] = self.hf_metadata["tags"]
        
        if self.hf_metadata["datasets"]:
            frontmatter["datasets"] = self.hf_metadata["datasets"]
        
        if self.hf_metadata["language"]:
            frontmatter["language"] = self.hf_metadata["language"]
        
        if self.hf_metadata["pipeline_tag"]:
            frontmatter["pipeline_tag"] = self.hf_metadata["pipeline_tag"]
        
        if self.hf_metadata["widget"]:
            frontmatter["widget"] = self.hf_metadata["widget"]
        
        if self.model_details.base_model:
            frontmatter["base_model"] = self.model_details.base_model
        
        # Add metrics to frontmatter
        if self.evaluation_results:
            model_index = {
                "name": self.model_details.name,
                "results": []
            }
            
            # Group metrics by dataset if available
            dataset_metrics = {}
            for metric in self.evaluation_results:
                dataset = metric.dataset or "default"
                if dataset not in dataset_metrics:
                    dataset_metrics[dataset] = {}
                dataset_metrics[dataset][metric.name] = metric.value
            
            for dataset, metrics in dataset_metrics.items():
                result = {
                    "task": {
                        "type": self.hf_metadata.get("pipeline_tag", "unknown")
                    },
                    "metrics": [
                        {"type": name, "value": value}
                        for name, value in metrics.items()
                    ]
                }
                if dataset != "default":
                    result["dataset"] = {"name": dataset}
                
                model_index["results"].append(result)
            
            frontmatter["model-index"] = [model_index]
        
        # Generate YAML frontmatter
        yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        
        # Generate markdown content
        markdown_content = self._render_hf_content()
        
        # Combine frontmatter and content
        full_content = f"---\n{yaml_content}---\n\n{markdown_content}"
        
        return full_content
    
    def _render_hf_content(self) -> str:
        """Render the main content section."""
        lines = []
        
        # Title
        lines.append(f"# {self.model_details.name}")
        
        # Description
        if self.model_details.description:
            lines.append(f"\n{self.model_details.description}")
        
        # Model Details
        lines.append("\n## Model Details")
        lines.append(f"\n### Model Description")
        
        if self.model_details.description:
            lines.append(f"- **Developed by:** {', '.join(self.model_details.authors) if self.model_details.authors else 'Not specified'}")
        
        if self.model_details.base_model:
            lines.append(f"- **Model type:** Fine-tuned from {self.model_details.base_model}")
        
        if self.model_details.language:
            lines.append(f"- **Language(s) (NLP):** {', '.join(self.model_details.language)}")
        
        if self.model_details.license:
            lines.append(f"- **License:** {self.model_details.license}")
        
        if self.model_details.base_model:
            lines.append(f"- **Finetuned from model:** {self.model_details.base_model}")
        
        # Intended Uses
        if self.intended_use:
            lines.append("\n## Intended uses & limitations")
            lines.append(f"\n{self.intended_use}")
        
        # Training Details
        lines.append("\n## Training Details")
        
        if self.training_details.training_data:
            lines.append("\n### Training Data")
            lines.append(f"This model was trained on the following datasets:")
            for dataset in self.training_details.training_data:
                lines.append(f"- {dataset}")
        
        if self.training_details.preprocessing:
            lines.append("\n### Training Procedure")
            lines.append(f"\n**Preprocessing:** {self.training_details.preprocessing}")
        
        if self.training_details.hyperparameters:
            lines.append("\n**Training Hyperparameters:**")
            for param, value in self.training_details.hyperparameters.items():
                lines.append(f"- {param}: {value}")
        
        # Evaluation
        if self.evaluation_results:
            lines.append("\n## Evaluation")
            
            # Group by dataset
            dataset_metrics = {}
            for metric in self.evaluation_results:
                dataset = metric.dataset or "Test Set"
                if dataset not in dataset_metrics:
                    dataset_metrics[dataset] = []
                dataset_metrics[dataset].append(metric)
            
            for dataset, metrics in dataset_metrics.items():
                lines.append(f"\n### {dataset}")
                for metric in metrics:
                    if metric.confidence_interval:
                        ci_str = f" (95% CI: {metric.confidence_interval})"
                    else:
                        ci_str = ""
                    lines.append(f"- **{metric.name}:** {metric.value:.4f}{ci_str}")
        
        # Environmental Impact
        if self.config.include_carbon_footprint:
            lines.append("\n## Environmental Impact")
            lines.append("Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).")
            
            if self.training_details.hardware:
                lines.append(f"- **Hardware Type:** {self.training_details.hardware}")
            if self.training_details.training_time:
                lines.append(f"- **Training Time:** {self.training_details.training_time}")
        
        # Bias, Risks, and Limitations
        if (self.limitations.known_limitations or 
            self.ethical_considerations.bias_risks or 
            self.config.include_ethical_considerations):
            
            lines.append("\n## Bias, Risks, and Limitations")
            
            if self.limitations.known_limitations:
                for limitation in self.limitations.known_limitations:
                    lines.append(f"- {limitation}")
            
            if self.ethical_considerations.bias_risks:
                lines.append("\n### Bias Analysis")
                for risk in self.ethical_considerations.bias_risks:
                    lines.append(f"- {risk}")
            
            if self.limitations.out_of_scope_uses:
                lines.append("\n### Out-of-Scope Use")
                for use_case in self.limitations.out_of_scope_uses:
                    lines.append(f"- {use_case}")
        
        # Recommendations
        if (self.limitations.recommendations or 
            self.ethical_considerations.bias_mitigation):
            
            lines.append("\n## Recommendations")
            
            if self.limitations.recommendations:
                for rec in self.limitations.recommendations:
                    lines.append(f"- {rec}")
            
            if self.ethical_considerations.bias_mitigation:
                lines.append("\n### Bias Mitigation")
                for mitigation in self.ethical_considerations.bias_mitigation:
                    lines.append(f"- {mitigation}")
        
        # How to Get Started with the Model
        if self.hf_metadata["widget"]:
            lines.append("\n## How to Get Started with the Model")
            lines.append("\nUse the code below to get started with the model.")
            lines.append("\n```python")
            lines.append("from transformers import AutoTokenizer, AutoModel")
            lines.append("")
            lines.append(f'tokenizer = AutoTokenizer.from_pretrained("{self.model_details.name}")')
            lines.append(f'model = AutoModel.from_pretrained("{self.model_details.name}")')
            lines.append("```")
        
        # Custom sections
        for section_name, content in self.custom_sections.items():
            lines.append(f"\n## {section_name}")
            lines.append(content)
        
        # Citation
        lines.append("\n## Citation")
        lines.append("\n**BibTeX:**")
        lines.append("\n```bibtex")
        lines.append("@misc{" + self.model_details.name.replace("-", "_") + ",")
        lines.append(f'  title={{{self.model_details.name}}},')
        if self.model_details.authors:
            lines.append(f'  author={{{", ".join(self.model_details.authors)}}},')
        lines.append(f'  year={{2024}},')
        lines.append(f'  url={{https://huggingface.co/{self.model_details.name}}}')
        lines.append("}")
        lines.append("```")
        
        return "\n".join(lines)
    
    def export_to_hub(self, repo_id: str, token: Optional[str] = None) -> None:
        """Export model card to Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi, login
        except ImportError:
            raise ImportError("huggingface_hub required for Hub export")
        
        if token:
            login(token=token)
        
        api = HfApi()
        
        # Generate model card content
        card_content = self.render("markdown")
        
        # Upload to Hub
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload model card for {self.model_details.name}"
        )
        
        print(f"Model card uploaded to https://huggingface.co/{repo_id}")


def create_hf_card_from_config(config_path: str) -> HuggingFaceCard:
    """Create HF model card from a configuration file."""
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    
    if config_file.suffix == '.json':
        with open(config_file) as f:
            config = json.load(f)
    elif config_file.suffix in ['.yaml', '.yml']:
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_file.suffix}")
    
    card = HuggingFaceCard()
    
    # Map config to model card fields
    if 'model' in config:
        model_config = config['model']
        card.model_details(
            name=model_config.get('name', 'unnamed-model'),
            languages=model_config.get('languages'),
            license=model_config.get('license'),
            finetuned_from=model_config.get('base_model'),
            **{k: v for k, v in model_config.items() 
               if k not in ['name', 'languages', 'license', 'base_model']}
        )
    
    if 'uses' in config:
        uses_config = config['uses']
        card.uses(
            direct_use=uses_config.get('direct'),
            downstream_use=uses_config.get('downstream'),
            out_of_scope=uses_config.get('out_of_scope')
        )
    
    if 'training' in config:
        training_config = config['training']
        card.training_data(
            datasets=training_config.get('datasets', []),
            preprocessing=training_config.get('preprocessing')
        )
    
    if 'evaluation' in config:
        card.evaluation(config['evaluation'])
    
    return card