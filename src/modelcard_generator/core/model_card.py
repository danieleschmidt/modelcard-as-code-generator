"""
Core ModelCard class representing a model card document.

This module provides the central ModelCard class that represents a structured
model card document with methods for manipulation, validation, and export.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import json
import yaml
from pathlib import Path
import hashlib


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    name: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    description: Optional[str] = None
    dataset: Optional[str] = None
    threshold: Optional[float] = None
    confidence_interval: Optional[List[float]] = None


@dataclass
class Dataset:
    """Represents a dataset used in model training or evaluation."""
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    url: Optional[str] = None
    size: Optional[int] = None
    license: Optional[str] = None
    preprocessing: Optional[str] = None


@dataclass
class ModelDetails:
    """Core model information."""
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    architecture: Optional[str] = None
    parameters: Optional[int] = None
    languages: List[str] = field(default_factory=list)
    license: Optional[str] = None
    base_model: Optional[str] = None
    finetuned_from: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ModelCard:
    """
    Core ModelCard class representing a structured model card document.
    
    This class provides a unified interface for creating, manipulating, and
    exporting model cards in various formats while maintaining consistency
    across different standards.
    """
    
    # Core sections (required)
    model_details: ModelDetails = field(default_factory=lambda: ModelDetails(name=""))
    
    # Optional sections
    intended_use: Dict[str, Any] = field(default_factory=dict)
    factors: Dict[str, Any] = field(default_factory=dict)
    metrics: List[MetricValue] = field(default_factory=list)
    evaluation_data: List[Dataset] = field(default_factory=list)
    training_data: List[Dataset] = field(default_factory=list)
    quantitative_analysis: Dict[str, Any] = field(default_factory=dict)
    ethical_considerations: Dict[str, Any] = field(default_factory=dict)
    caveats_and_recommendations: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance sections
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    technical_robustness: Dict[str, Any] = field(default_factory=dict)
    
    # Custom sections
    custom_sections: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metadata after object creation."""
        if not self.metadata:
            self.metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "generated_by": "modelcard-as-code-generator",
                "version": "1.0",
                "schema_version": "1.0"
            }
    
    # Core manipulation methods
    
    def add_metric(self, metric: MetricValue) -> None:
        """Add a metric to the model card."""
        self.metrics.append(metric)
    
    def update_metric(self, name: str, value: Union[int, float, str], **kwargs) -> None:
        """Update or add a metric value."""
        # Find existing metric
        for metric in self.metrics:
            if metric.name == name:
                metric.value = value
                for key, val in kwargs.items():
                    if hasattr(metric, key):
                        setattr(metric, key, val)
                return
        
        # Add new metric if not found
        self.add_metric(MetricValue(name=name, value=value, **kwargs))
    
    def get_metric(self, name: str) -> Optional[MetricValue]:
        """Get a metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
    
    def add_dataset(self, dataset: Dataset, dataset_type: str = "training") -> None:
        """Add a dataset to training or evaluation data."""
        if dataset_type == "training":
            self.training_data.append(dataset)
        elif dataset_type == "evaluation":
            self.evaluation_data.append(dataset)
        else:
            raise ValueError("dataset_type must be 'training' or 'evaluation'")
    
    def add_section(self, section_name: str, content: Any) -> None:
        """Add a custom section to the model card."""
        self.custom_sections[section_name] = content
    
    def get_section(self, section_name: str) -> Optional[Any]:
        """Get a section from the model card."""
        # Check standard sections first
        if hasattr(self, section_name):
            return getattr(self, section_name)
        
        # Check custom sections
        return self.custom_sections.get(section_name)
    
    def update_section(self, section_name: str, content: Any) -> None:
        """Update a section in the model card."""
        if hasattr(self, section_name):
            setattr(self, section_name, content)
        else:
            self.custom_sections[section_name] = content
    
    # Export methods
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary representation."""
        data = {
            "model_details": {
                "name": self.model_details.name,
                "version": self.model_details.version,
                "description": self.model_details.description,
                "architecture": self.model_details.architecture,
                "parameters": self.model_details.parameters,
                "languages": self.model_details.languages,
                "license": self.model_details.license,
                "base_model": self.model_details.base_model,
                "finetuned_from": self.model_details.finetuned_from,
                "tags": self.model_details.tags,
            },
            "intended_use": self.intended_use,
            "factors": self.factors,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "description": metric.description,
                    "dataset": metric.dataset,
                    "threshold": metric.threshold,
                    "confidence_interval": metric.confidence_interval,
                }
                for metric in self.metrics
            ],
            "evaluation_data": [
                {
                    "name": dataset.name,
                    "description": dataset.description,
                    "version": dataset.version,
                    "url": dataset.url,
                    "size": dataset.size,
                    "license": dataset.license,
                    "preprocessing": dataset.preprocessing,
                }
                for dataset in self.evaluation_data
            ],
            "training_data": [
                {
                    "name": dataset.name,
                    "description": dataset.description,
                    "version": dataset.version,
                    "url": dataset.url,
                    "size": dataset.size,
                    "license": dataset.license,
                    "preprocessing": dataset.preprocessing,
                }
                for dataset in self.training_data
            ],
            "quantitative_analysis": self.quantitative_analysis,
            "ethical_considerations": self.ethical_considerations,
            "caveats_and_recommendations": self.caveats_and_recommendations,
            "risk_assessment": self.risk_assessment,
            "technical_robustness": self.technical_robustness,
            "custom_sections": self.custom_sections,
            "metadata": self.metadata,
        }
        
        # Remove None values and empty collections
        return self._clean_dict(data)
    
    def _clean_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values and empty collections from dictionary."""
        cleaned = {}
        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, dict):
                cleaned_dict = self._clean_dict(value)
                if cleaned_dict:
                    cleaned[key] = cleaned_dict
            elif isinstance(value, list):
                if value:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned
    
    def to_json(self, indent: int = 2) -> str:
        """Export model card as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_yaml(self) -> str:
        """Export model card as YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True, indent=2)
    
    def save_json(self, file_path: Path) -> None:
        """Save model card as JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def save_yaml(self, file_path: Path) -> None:
        """Save model card as YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_yaml())
    
    def save(self, file_path: Path, format: Optional[str] = None) -> None:
        """Save model card to file with auto-detected or specified format."""
        file_path = Path(file_path)
        
        if format is None:
            format = file_path.suffix.lower().lstrip('.')
        
        if format in ['json']:
            self.save_json(file_path)
        elif format in ['yaml', 'yml']:
            self.save_yaml(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: json, yaml, yml")
    
    # Import methods
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        """Create ModelCard from dictionary."""
        card = cls()
        
        # Model details
        if "model_details" in data:
            model_data = data["model_details"]
            card.model_details = ModelDetails(
                name=model_data.get("name", ""),
                version=model_data.get("version"),
                description=model_data.get("description"),
                architecture=model_data.get("architecture"),
                parameters=model_data.get("parameters"),
                languages=model_data.get("languages", []),
                license=model_data.get("license"),
                base_model=model_data.get("base_model"),
                finetuned_from=model_data.get("finetuned_from"),
                tags=model_data.get("tags", []),
            )
        
        # Standard sections
        card.intended_use = data.get("intended_use", {})
        card.factors = data.get("factors", {})
        card.quantitative_analysis = data.get("quantitative_analysis", {})
        card.ethical_considerations = data.get("ethical_considerations", {})
        card.caveats_and_recommendations = data.get("caveats_and_recommendations", {})
        card.risk_assessment = data.get("risk_assessment", {})
        card.technical_robustness = data.get("technical_robustness", {})
        card.custom_sections = data.get("custom_sections", {})
        card.metadata = data.get("metadata", {})
        
        # Metrics
        if "metrics" in data:
            card.metrics = [
                MetricValue(
                    name=metric.get("name", ""),
                    value=metric.get("value"),
                    unit=metric.get("unit"),
                    description=metric.get("description"),
                    dataset=metric.get("dataset"),
                    threshold=metric.get("threshold"),
                    confidence_interval=metric.get("confidence_interval"),
                )
                for metric in data["metrics"]
            ]
        
        # Datasets
        for dataset_type in ["training_data", "evaluation_data"]:
            if dataset_type in data:
                datasets = [
                    Dataset(
                        name=ds.get("name", ""),
                        description=ds.get("description"),
                        version=ds.get("version"),
                        url=ds.get("url"),
                        size=ds.get("size"),
                        license=ds.get("license"),
                        preprocessing=ds.get("preprocessing"),
                    )
                    for ds in data[dataset_type]
                ]
                setattr(card, dataset_type, datasets)
        
        return card
    
    @classmethod
    def from_json(cls, json_str: str) -> "ModelCard":
        """Create ModelCard from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ModelCard":
        """Create ModelCard from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def load(cls, file_path: Path) -> "ModelCard":
        """Load ModelCard from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model card file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_path.suffix.lower() == '.json':
            return cls.from_json(content)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            return cls.from_yaml(content)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Utility methods
    
    def get_hash(self) -> str:
        """Get hash of model card content for change detection."""
        content = self.to_json()
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_completeness_score(self) -> float:
        """Calculate completeness score based on filled sections."""
        total_fields = 0
        filled_fields = 0
        
        # Model details
        detail_fields = ["name", "version", "description", "architecture", "license"]
        for field in detail_fields:
            total_fields += 1
            if getattr(self.model_details, field):
                filled_fields += 1
        
        # Standard sections
        sections = [
            self.intended_use, self.factors, self.quantitative_analysis,
            self.ethical_considerations, self.caveats_and_recommendations
        ]
        
        for section in sections:
            total_fields += 1
            if section:
                filled_fields += 1
        
        # Metrics and datasets
        total_fields += 2
        if self.metrics:
            filled_fields += 1
        if self.training_data or self.evaluation_data:
            filled_fields += 1
        
        return filled_fields / total_fields if total_fields > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about the model card."""
        return {
            "model_name": self.model_details.name,
            "model_version": self.model_details.version,
            "metrics_count": len(self.metrics),
            "training_datasets": len(self.training_data),
            "evaluation_datasets": len(self.evaluation_data),
            "custom_sections": len(self.custom_sections),
            "completeness_score": self.get_completeness_score(),
            "content_hash": self.get_hash(),
            "last_updated": self.metadata.get("created_at"),
        }
    
    def validate_basic(self) -> List[str]:
        """Perform basic validation and return list of issues."""
        issues = []
        
        # Model name is required
        if not self.model_details.name:
            issues.append("Model name is required")
        
        # At least one metric should be present
        if not self.metrics:
            issues.append("At least one metric should be provided")
        
        # At least one dataset should be specified
        if not self.training_data and not self.evaluation_data:
            issues.append("At least one training or evaluation dataset should be specified")
        
        return issues