"""Core data models for model cards."""

import json

try:
    import yaml
except ImportError:
    yaml = None
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class CardFormat(Enum):
    """Supported model card formats."""
    HUGGINGFACE = "huggingface"
    GOOGLE = "google" 
    EU_CRA = "eu_cra"
    CUSTOM = "custom"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A validation issue found in a model card."""
    severity: ValidationSeverity
    message: str
    path: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass 
class ValidationResult:
    """Result of model card validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_sections: List[str] = field(default_factory=list)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]


@dataclass
class CardConfig:
    """Configuration for model card generation."""
    format: CardFormat = CardFormat.HUGGINGFACE
    include_ethical_considerations: bool = True
    include_carbon_footprint: bool = True
    include_bias_analysis: bool = True
    regulatory_standard: Optional[str] = None
    template_name: Optional[str] = None
    auto_populate: bool = True
    validation_strict: bool = False
    output_format: str = "markdown"  # markdown, json, html


@dataclass
class ModelDetails:
    """Model details section."""
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    license: Optional[str] = None
    base_model: Optional[str] = None
    language: List[str] = field(default_factory=list)
    library_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetric:
    """A single performance metric."""
    name: str
    value: float
    confidence_interval: Optional[List[float]] = None
    dataset: Optional[str] = None
    slice_name: Optional[str] = None


@dataclass
class TrainingDetails:
    """Training process details."""
    framework: Optional[str] = None
    model_architecture: Optional[str] = None
    training_data: List[str] = field(default_factory=list)
    preprocessing: Optional[str] = None
    hardware: Optional[str] = None
    training_time: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EthicalConsiderations:
    """Ethical considerations and bias analysis."""
    bias_risks: List[str] = field(default_factory=list)
    bias_mitigation: List[str] = field(default_factory=list)
    fairness_constraints: List[str] = field(default_factory=list)
    sensitive_attributes: List[str] = field(default_factory=list)
    fairness_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Limitations:
    """Model limitations and recommendations."""
    known_limitations: List[str] = field(default_factory=list)
    sensitive_use_cases: List[str] = field(default_factory=list)
    out_of_scope_uses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ModelCard:
    """Main model card class containing all sections."""
    
    def __init__(self, config: Optional[CardConfig] = None):
        self.config = config or CardConfig()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Core sections
        self.model_details = ModelDetails(name="")
        self.intended_use = ""
        self.training_details = TrainingDetails()
        self.evaluation_results: List[PerformanceMetric] = []
        self.ethical_considerations = EthicalConsiderations()
        self.limitations = Limitations()
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.custom_sections: Dict[str, str] = {}
        
        # Compliance tracking
        self.compliance_info: Dict[str, Any] = {}
        self.audit_trail: List[Dict[str, Any]] = []
    
    def add_metric(self, name: str, value: float, **kwargs) -> None:
        """Add a performance metric."""
        metric = PerformanceMetric(name=name, value=value, **kwargs)
        self.evaluation_results.append(metric)
        self._update_timestamp()
    
    def update_metric(self, name: str, value: float, reason: Optional[str] = None) -> None:
        """Update an existing metric value."""
        for metric in self.evaluation_results:
            if metric.name == name:
                old_value = metric.value
                metric.value = value
                self._log_change(f"Updated metric {name}", {
                    "old_value": old_value,
                    "new_value": value,
                    "reason": reason
                })
                break
        else:
            self.add_metric(name, value)
        self._update_timestamp()
    
    def add_section(self, name: str, content: str) -> None:
        """Add a custom section."""
        self.custom_sections[name] = content
        self._log_change(f"Added section {name}", {"content_length": len(content)})
        self._update_timestamp()
    
    def add_limitation(self, limitation: str) -> None:
        """Add a model limitation."""
        self.limitations.known_limitations.append(limitation)
        self._log_change("Added limitation", {"limitation": limitation})
        self._update_timestamp()
    
    def set_compliance_info(self, standard: str, status: str, details: Dict[str, Any]) -> None:
        """Set compliance information for a regulatory standard."""
        self.compliance_info[standard] = {
            "status": status,
            "details": details,
            "checked_at": datetime.now().isoformat()
        }
        self._update_timestamp()
    
    def render(self, format_type: str = "markdown") -> str:
        """Render the model card in the specified format."""
        if format_type == "markdown":
            return self._render_markdown()
        elif format_type == "json":
            return self._render_json()
        elif format_type == "html":
            return self._render_html()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the model card to a file."""
        path = Path(path)
        content = self.render(self.config.output_format)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        
        self._log_change("Saved model card", {"path": str(path)})
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'ModelCard':
        """Load model card from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model card file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse based on file extension and content
        if path.suffix.lower() == '.json':
            return cls._parse_json_card(content)
        elif path.suffix.lower() in ['.md', '.txt']:
            return cls._parse_markdown_card(content)
        else:
            # Try to detect format
            if content.strip().startswith('{'):
                return cls._parse_json_card(content)
            else:
                return cls._parse_markdown_card(content)
    
    @classmethod
    def _parse_json_card(cls, content: str) -> 'ModelCard':
        """Parse JSON format model card."""
        import json
        data = json.loads(content)
        
        card = cls()
        
        # Parse model details
        if 'model_details' in data:
            details = data['model_details']
            card.model_details.name = details.get('name', '')
            card.model_details.version = details.get('version', '')
            card.model_details.description = details.get('description', '')
            card.model_details.authors = details.get('authors', [])
            card.model_details.license = details.get('license', '')
            card.model_details.base_model = details.get('base_model', '')
            card.model_details.language = details.get('language', [])
            card.model_details.tags = details.get('tags', [])
            card.model_details.datasets = details.get('datasets', [])
        
        # Parse intended use
        if 'intended_use' in data:
            card.intended_use = data['intended_use']
        
        # Parse training details
        if 'training_details' in data:
            training = data['training_details']
            card.training_details.framework = training.get('framework', '')
            card.training_details.model_architecture = training.get('model_architecture', '')
            card.training_details.training_data = training.get('training_data', [])
            card.training_details.preprocessing = training.get('preprocessing', '')
            card.training_details.hyperparameters = training.get('hyperparameters', {})
        
        # Parse evaluation results
        if 'evaluation_results' in data:
            for metric_data in data['evaluation_results']:
                if isinstance(metric_data, dict) and 'name' in metric_data and 'value' in metric_data:
                    card.add_metric(
                        metric_data['name'],
                        metric_data['value'],
                        confidence_interval=metric_data.get('confidence_interval')
                    )
        
        # Parse limitations
        if 'limitations' in data:
            limitations = data['limitations']
            if 'known_limitations' in limitations:
                for limitation in limitations['known_limitations']:
                    card.add_limitation(limitation)
        
        return card
    
    @classmethod
    def _parse_markdown_card(cls, content: str) -> 'ModelCard':
        """Parse Markdown format model card."""
        import re
        
        card = cls()
        
        # Extract title (model name)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            card.model_details.name = title_match.group(1).strip()
        
        # Extract model details section
        model_details_section = cls._extract_section(content, "Model Details")
        if model_details_section:
            # Parse version - look for both bullet point and regular formats
            version_match = re.search(r'-\s*\*\*Version\*\*:\s*(.+)', model_details_section, re.IGNORECASE) or \
                           re.search(r'Version["\s]*:?["\s]*(.*?)(?:\n|$)', model_details_section, re.IGNORECASE)
            if version_match:
                card.model_details.version = version_match.group(1).strip(' *-')
            
            # Parse authors - look for both bullet point and regular formats
            authors_match = re.search(r'-\s*\*\*Authors?\*\*:\s*(.+)', model_details_section, re.IGNORECASE) or \
                           re.search(r'Authors?["\s]*:?["\s]*(.*?)(?:\n|$)', model_details_section, re.IGNORECASE)
            if authors_match:
                authors_str = authors_match.group(1).strip(' *-')
                if authors_str:
                    card.model_details.authors = [a.strip() for a in authors_str.split(',')]
            
            # Parse license - look for both bullet point and regular formats
            license_match = re.search(r'-\s*\*\*License\*\*:\s*(.+)', model_details_section, re.IGNORECASE) or \
                           re.search(r'License["\s]*:?["\s]*(.*?)(?:\n|$)', model_details_section, re.IGNORECASE)
            if license_match:
                card.model_details.license = license_match.group(1).strip(' *-')
        
        # Extract intended use section
        intended_use_section = cls._extract_section(content, "Intended Use")
        if intended_use_section:
            # Clean up the section text (remove markdown formatting)
            cleaned_text = re.sub(r'^#+\s*', '', intended_use_section, flags=re.MULTILINE)
            cleaned_text = cleaned_text.strip()
            if cleaned_text:
                card.intended_use = cleaned_text
        
        # Extract evaluation results
        eval_section = cls._extract_section(content, "Evaluation Results")
        if eval_section:
            # Find metric lines like "- **accuracy**: 0.95"
            metric_matches = re.findall(r'-\s*\*\*([^*]+)\*\*:\s*([\d.]+)', eval_section)
            for name, value in metric_matches:
                try:
                    card.add_metric(name.strip(), float(value))
                except ValueError:
                    continue
        
        # Extract limitations
        limitations_section = cls._extract_section(content, "Limitations")
        if limitations_section:
            # Find limitation lines like "- Some limitation"
            limitation_matches = re.findall(r'-\s*(.+)$', limitations_section, re.MULTILINE)
            for limitation in limitation_matches:
                if limitation.strip():
                    card.add_limitation(limitation.strip())
        
        # Extract training data from training details section
        training_section = cls._extract_section(content, "Training Details")
        if training_section:
            # Look for dataset mentions
            dataset_matches = re.findall(r'dataset[s]?[:\s]*([^\n]+)', training_section, re.IGNORECASE)
            for dataset in dataset_matches:
                card.training_details.training_data.append(dataset.strip())
        
        return card
    
    @staticmethod
    def _extract_section(content: str, section_name: str) -> Optional[str]:
        """Extract a section from markdown content."""
        import re
        
        # Pattern to match section header and content until next header or end of file
        pattern = rf'^#+\s*{re.escape(section_name)}\s*\n(.*?)(?=\n#+|\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return None
    
    def export_jsonld(self, path: Union[str, Path]) -> None:
        """Export as JSON-LD for machine reading."""
        jsonld_data = {
            "@context": "https://schema.org/",
            "@type": "SoftwareApplication",
            "name": self.model_details.name,
            "version": self.model_details.version,
            "description": self.model_details.description,
            "author": self.model_details.authors,
            "license": self.model_details.license,
            "applicationCategory": "MachineLearningModel",
            "operatingSystem": "Any",
            "dateCreated": self.created_at.isoformat(),
            "dateModified": self.updated_at.isoformat(),
            "metrics": [
                {
                    "@type": "PropertyValue",
                    "name": metric.name,
                    "value": metric.value
                }
                for metric in self.evaluation_results
            ]
        }
        
        Path(path).write_text(json.dumps(jsonld_data, indent=2), encoding="utf-8")
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get the audit trail of changes."""
        return self.audit_trail.copy()
    
    def enable_audit_trail(self) -> None:
        """Enable audit trail tracking."""
        self.metadata["audit_enabled"] = True
    
    def _render_markdown(self) -> str:
        """Render as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# {self.model_details.name}")
        if self.model_details.description:
            lines.append(f"\n{self.model_details.description}")
        
        # Model Details
        lines.append("\n## Model Details")
        if self.model_details.version:
            lines.append(f"- **Version**: {self.model_details.version}")
        if self.model_details.authors:
            lines.append(f"- **Authors**: {', '.join(self.model_details.authors)}")
        if self.model_details.license:
            lines.append(f"- **License**: {self.model_details.license}")
        if self.model_details.base_model:
            lines.append(f"- **Base Model**: {self.model_details.base_model}")
        
        # Intended Use
        if self.intended_use:
            lines.append("\n## Intended Use")
            lines.append(self.intended_use)
        
        # Training Details
        lines.append("\n## Training Details")
        if self.training_details.framework:
            lines.append(f"- **Framework**: {self.training_details.framework}")
        if self.training_details.training_data:
            lines.append(f"- **Training Data**: {', '.join(self.training_details.training_data)}")
        if self.training_details.hyperparameters:
            lines.append("- **Hyperparameters**:")
            for key, value in self.training_details.hyperparameters.items():
                lines.append(f"  - {key}: {value}")
        
        # Evaluation Results
        if self.evaluation_results:
            lines.append("\n## Evaluation Results")
            for metric in self.evaluation_results:
                if metric.confidence_interval:
                    ci_str = f" (CI: {metric.confidence_interval})"
                else:
                    ci_str = ""
                lines.append(f"- **{metric.name}**: {metric.value}{ci_str}")
        
        # Ethical Considerations
        if self.config.include_ethical_considerations and (
            self.ethical_considerations.bias_risks or 
            self.ethical_considerations.fairness_metrics
        ):
            lines.append("\n## Ethical Considerations")
            if self.ethical_considerations.bias_risks:
                lines.append("### Bias Risks")
                for risk in self.ethical_considerations.bias_risks:
                    lines.append(f"- {risk}")
            if self.ethical_considerations.fairness_metrics:
                lines.append("### Fairness Metrics")
                for metric, value in self.ethical_considerations.fairness_metrics.items():
                    lines.append(f"- **{metric}**: {value}")
        
        # Limitations
        if self.limitations.known_limitations:
            lines.append("\n## Limitations")
            for limitation in self.limitations.known_limitations:
                lines.append(f"- {limitation}")
        
        # Custom sections
        for section_name, content in self.custom_sections.items():
            lines.append(f"\n## {section_name}")
            lines.append(content)
        
        # Compliance info
        if self.compliance_info:
            lines.append("\n## Compliance Information")
            for standard, info in self.compliance_info.items():
                lines.append(f"- **{standard}**: {info['status']}")
        
        return "\n".join(lines)
    
    def _render_json(self) -> str:
        """Render as JSON."""
        data = {
            "model_details": {
                "name": self.model_details.name,
                "version": self.model_details.version,
                "description": self.model_details.description,
                "authors": self.model_details.authors,
                "license": self.model_details.license,
                "base_model": self.model_details.base_model,
                "language": self.model_details.language,
                "tags": self.model_details.tags,
                "datasets": self.model_details.datasets
            },
            "intended_use": self.intended_use,
            "training_details": {
                "framework": self.training_details.framework,
                "model_architecture": self.training_details.model_architecture,
                "training_data": self.training_details.training_data,
                "preprocessing": self.training_details.preprocessing,
                "hardware": self.training_details.hardware,
                "training_time": self.training_details.training_time,
                "hyperparameters": self.training_details.hyperparameters
            },
            "evaluation_results": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "confidence_interval": metric.confidence_interval,
                    "dataset": metric.dataset,
                    "slice": metric.slice_name
                }
                for metric in self.evaluation_results
            ],
            "ethical_considerations": {
                "bias_risks": self.ethical_considerations.bias_risks,
                "bias_mitigation": self.ethical_considerations.bias_mitigation,
                "fairness_constraints": self.ethical_considerations.fairness_constraints,
                "sensitive_attributes": self.ethical_considerations.sensitive_attributes,
                "fairness_metrics": self.ethical_considerations.fairness_metrics
            },
            "limitations": {
                "known_limitations": self.limitations.known_limitations,
                "sensitive_use_cases": self.limitations.sensitive_use_cases,
                "out_of_scope_uses": self.limitations.out_of_scope_uses,
                "recommendations": self.limitations.recommendations
            },
            "custom_sections": self.custom_sections,
            "compliance_info": self.compliance_info,
            "metadata": {
                **self.metadata,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat()
            }
        }
        return json.dumps(data, indent=2)
    
    def _render_html(self) -> str:
        """Render as HTML."""
        import html
        
        lines = ["<!DOCTYPE html>", "<html>", "<head>"]
        lines.append(f"<title>{html.escape(self.model_details.name)}</title>")
        lines.append("<style>")
        lines.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        lines.append("h1 { color: #333; }")
        lines.append("h2 { color: #666; border-bottom: 1px solid #ddd; }")
        lines.append("ul { margin-left: 20px; }")
        lines.append("</style>")
        lines.append("</head><body>")
        
        # Convert markdown to HTML (simplified)
        markdown_content = self._render_markdown()
        html_content = markdown_content.replace("\n# ", "\n<h1>").replace("</h1>", "</h1>")
        html_content = html_content.replace("\n## ", "\n<h2>").replace("</h2>", "</h2>")
        html_content = html_content.replace("\n- ", "\n<li>").replace("</li>", "</li>")
        html_content = html_content.replace("\n\n", "</p><p>")
        html_content = f"<p>{html_content}</p>"
        
        lines.append(html_content)
        lines.append("</body></html>")
        
        return "\n".join(lines)
    
    def _update_timestamp(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = datetime.now()
    
    def _log_change(self, action: str, details: Dict[str, Any]) -> None:
        """Log a change to the audit trail."""
        if self.metadata.get("audit_enabled", False):
            self.audit_trail.append({
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details,
                "author": "system"  # Could be enhanced to track actual user
            })


@dataclass
class DriftMetricChange:
    """A change in a metric value indicating drift."""
    metric_name: str
    old_value: float
    new_value: float
    delta: float
    threshold: float
    is_significant: bool


@dataclass
class DriftReport:
    """Report on detected model card drift."""
    has_drift: bool
    changes: List[DriftMetricChange] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def significant_changes(self) -> List[DriftMetricChange]:
        """Get only significant changes that exceed thresholds."""
        return [change for change in self.changes if change.is_significant]