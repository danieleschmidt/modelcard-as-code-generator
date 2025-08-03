"""Google Model Cards format implementation."""

import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..core.models import ModelCard, CardConfig, CardFormat, PerformanceMetric


@dataclass
class Graphic:
    """Graphic representation for Google Model Cards."""
    name: str
    image: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Owner:
    """Model owner information."""
    name: str
    contact: str


@dataclass
class QuantitativeAnalysisMetric:
    """Quantitative analysis metric for Google format."""
    type: str
    value: float
    confidence_interval: Optional[List[float]] = None
    slice: str = "overall"
    description: Optional[str] = None


@dataclass
class ModelDetails:
    """Google Model Cards model details."""
    name: str
    owners: List[Owner] = field(default_factory=list)
    version: Optional[str] = None
    date: Optional[str] = None
    type: Optional[str] = None
    paper: Optional[str] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    path: Optional[str] = None


class GoogleModelCard(ModelCard):
    """Google Model Cards format implementation."""
    
    def __init__(self, config: Optional[CardConfig] = None):
        if config is None:
            config = CardConfig(format=CardFormat.GOOGLE)
        super().__init__(config)
        
        # Google-specific schema
        self.schema_version = "1.0"
        self.google_model_details = ModelDetails(name="")
        self.considerations = {
            "users": [],
            "use_cases": [],
            "limitations": [],
            "tradeoffs": [],
            "ethical_considerations": []
        }
        self.quantitative_analysis = {
            "performance_metrics": [],
            "graphics": []
        }
    
    def set_model_details(
        self,
        name: str,
        owners: List[Dict[str, str]],
        version: Optional[str] = None,
        model_type: Optional[str] = None,
        paper: Optional[str] = None,
        citation: Optional[str] = None,
        license: Optional[str] = None,
        path: Optional[str] = None
    ) -> None:
        """Set Google-specific model details."""
        self.google_model_details.name = name
        self.google_model_details.version = version
        self.google_model_details.date = datetime.now().isoformat()
        self.google_model_details.type = model_type
        self.google_model_details.paper = paper
        self.google_model_details.citation = citation
        self.google_model_details.license = license
        self.google_model_details.path = path
        
        # Convert owner dicts to Owner objects
        self.google_model_details.owners = [
            Owner(name=owner.get("name", ""), contact=owner.get("contact", ""))
            for owner in owners
        ]
        
        # Update base model details
        self.model_details.name = name
        self.model_details.version = version
        self.model_details.license = license
        if owners:
            self.model_details.authors = [owner.get("name", "") for owner in owners]
    
    def add_intended_users(self, users: List[str]) -> None:
        """Add intended users."""
        self.considerations["users"].extend(users)
    
    def add_use_cases(self, use_cases: List[str]) -> None:
        """Add intended use cases."""
        self.considerations["use_cases"].extend(use_cases)
        
        # Update base intended use
        if not self.intended_use:
            self.intended_use = "Intended for the following use cases:\n" + "\n".join(f"- {uc}" for uc in use_cases)
    
    def add_limitations(self, limitations: List[str]) -> None:
        """Add model limitations."""
        self.considerations["limitations"].extend(limitations)
        self.limitations.known_limitations.extend(limitations)
    
    def add_tradeoffs(self, tradeoffs: List[str]) -> None:
        """Add model tradeoffs."""
        self.considerations["tradeoffs"].extend(tradeoffs)
    
    def add_ethical_considerations(self, considerations: List[str]) -> None:
        """Add ethical considerations."""
        self.considerations["ethical_considerations"].extend(considerations)
        self.ethical_considerations.bias_risks.extend(considerations)
    
    def add_performance_metric(
        self,
        metric_type: str,
        value: float,
        confidence_interval: Optional[List[float]] = None,
        slice_name: str = "overall",
        description: Optional[str] = None
    ) -> None:
        """Add performance metric in Google format."""
        metric = QuantitativeAnalysisMetric(
            type=metric_type,
            value=value,
            confidence_interval=confidence_interval,
            slice=slice_name,
            description=description
        )
        self.quantitative_analysis["performance_metrics"].append(metric)
        
        # Also add to base metrics
        self.add_metric(metric_type, value, confidence_interval=confidence_interval)
    
    def add_graphic(self, name: str, image_path: Optional[str] = None, description: Optional[str] = None) -> None:
        """Add graphic representation."""
        graphic = Graphic(name=name, image=image_path, description=description)
        self.quantitative_analysis["graphics"].append(graphic)
    
    def render(self, format_type: str = "json") -> str:
        """Render Google Model Card."""
        if format_type == "json":
            return self._render_google_json()
        elif format_type == "proto":
            return self._render_google_proto()
        elif format_type == "markdown":
            return self._render_google_markdown()
        else:
            return super().render(format_type)
    
    def _render_google_json(self) -> str:
        """Render as Google Model Cards JSON format."""
        data = {
            "schema_version": self.schema_version,
            "model_details": {
                "name": self.google_model_details.name,
                "owners": [
                    {"name": owner.name, "contact": owner.contact}
                    for owner in self.google_model_details.owners
                ],
                "version": self.google_model_details.version,
                "date": self.google_model_details.date,
                "type": self.google_model_details.type,
                "paper": self.google_model_details.paper,
                "citation": self.google_model_details.citation,
                "license": self.google_model_details.license,
                "path": self.google_model_details.path
            },
            "considerations": {
                "users": self.considerations["users"],
                "use_cases": self.considerations["use_cases"],
                "limitations": self.considerations["limitations"],
                "tradeoffs": self.considerations["tradeoffs"],
                "ethical_considerations": self.considerations["ethical_considerations"]
            },
            "quantitative_analysis": {
                "performance_metrics": [
                    {
                        "type": metric.type,
                        "value": metric.value,
                        "confidence_interval": metric.confidence_interval,
                        "slice": metric.slice,
                        "description": metric.description
                    }
                    for metric in self.quantitative_analysis["performance_metrics"]
                ],
                "graphics": [
                    {
                        "name": graphic.name,
                        "image": graphic.image,
                        "description": graphic.description
                    }
                    for graphic in self.quantitative_analysis["graphics"]
                ]
            }
        }
        
        # Remove None values
        data = self._remove_none_values(data)
        
        return json.dumps(data, indent=2)
    
    def _render_google_proto(self) -> str:
        """Render as Protocol Buffer text format."""
        lines = []
        
        lines.append(f'schema_version: "{self.schema_version}"')
        
        # Model details
        lines.append("model_details {")
        lines.append(f'  name: "{self.google_model_details.name}"')
        
        for owner in self.google_model_details.owners:
            lines.append("  owners {")
            lines.append(f'    name: "{owner.name}"')
            lines.append(f'    contact: "{owner.contact}"')
            lines.append("  }")
        
        if self.google_model_details.version:
            lines.append(f'  version: "{self.google_model_details.version}"')
        if self.google_model_details.date:
            lines.append(f'  date: "{self.google_model_details.date}"')
        if self.google_model_details.type:
            lines.append(f'  type: "{self.google_model_details.type}"')
        if self.google_model_details.license:
            lines.append(f'  license: "{self.google_model_details.license}"')
        
        lines.append("}")
        
        # Considerations
        lines.append("considerations {")
        
        for user in self.considerations["users"]:
            lines.append(f'  users: "{user}"')
        
        for use_case in self.considerations["use_cases"]:
            lines.append(f'  use_cases: "{use_case}"')
        
        for limitation in self.considerations["limitations"]:
            lines.append(f'  limitations: "{limitation}"')
        
        for tradeoff in self.considerations["tradeoffs"]:
            lines.append(f'  tradeoffs: "{tradeoff}"')
        
        for consideration in self.considerations["ethical_considerations"]:
            lines.append(f'  ethical_considerations: "{consideration}"')
        
        lines.append("}")
        
        # Quantitative analysis
        lines.append("quantitative_analysis {")
        
        for metric in self.quantitative_analysis["performance_metrics"]:
            lines.append("  performance_metrics {")
            lines.append(f'    type: "{metric.type}"')
            lines.append(f'    value: {metric.value}')
            if metric.confidence_interval:
                lines.append(f'    confidence_interval: {metric.confidence_interval}')
            lines.append(f'    slice: "{metric.slice}"')
            if metric.description:
                lines.append(f'    description: "{metric.description}"')
            lines.append("  }")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _render_google_markdown(self) -> str:
        """Render as Markdown for Google Model Cards."""
        lines = []
        
        # Title
        lines.append(f"# {self.google_model_details.name}")
        
        # Model Details
        lines.append("\n## Model Details")
        
        if self.google_model_details.owners:
            lines.append("### Owners")
            for owner in self.google_model_details.owners:
                lines.append(f"- **{owner.name}** ({owner.contact})")
        
        if self.google_model_details.version:
            lines.append(f"\n**Version:** {self.google_model_details.version}")
        
        if self.google_model_details.date:
            lines.append(f"**Date:** {self.google_model_details.date}")
        
        if self.google_model_details.type:
            lines.append(f"**Type:** {self.google_model_details.type}")
        
        if self.google_model_details.paper:
            lines.append(f"**Paper:** {self.google_model_details.paper}")
        
        if self.google_model_details.citation:
            lines.append(f"**Citation:** {self.google_model_details.citation}")
        
        if self.google_model_details.license:
            lines.append(f"**License:** {self.google_model_details.license}")
        
        # Considerations
        lines.append("\n## Considerations")
        
        if self.considerations["users"]:
            lines.append("\n### Intended Users")
            for user in self.considerations["users"]:
                lines.append(f"- {user}")
        
        if self.considerations["use_cases"]:
            lines.append("\n### Use Cases")
            for use_case in self.considerations["use_cases"]:
                lines.append(f"- {use_case}")
        
        if self.considerations["limitations"]:
            lines.append("\n### Limitations")
            for limitation in self.considerations["limitations"]:
                lines.append(f"- {limitation}")
        
        if self.considerations["tradeoffs"]:
            lines.append("\n### Tradeoffs")
            for tradeoff in self.considerations["tradeoffs"]:
                lines.append(f"- {tradeoff}")
        
        if self.considerations["ethical_considerations"]:
            lines.append("\n### Ethical Considerations")
            for consideration in self.considerations["ethical_considerations"]:
                lines.append(f"- {consideration}")
        
        # Quantitative Analysis
        if self.quantitative_analysis["performance_metrics"]:
            lines.append("\n## Quantitative Analysis")
            
            # Group metrics by slice
            slice_metrics = {}
            for metric in self.quantitative_analysis["performance_metrics"]:
                slice_name = metric.slice or "overall"
                if slice_name not in slice_metrics:
                    slice_metrics[slice_name] = []
                slice_metrics[slice_name].append(metric)
            
            for slice_name, metrics in slice_metrics.items():
                lines.append(f"\n### {slice_name.title()} Performance")
                
                for metric in metrics:
                    value_str = f"{metric.value:.4f}"
                    if metric.confidence_interval:
                        ci = metric.confidence_interval
                        value_str += f" (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])"
                    
                    lines.append(f"- **{metric.type}:** {value_str}")
                    if metric.description:
                        lines.append(f"  - {metric.description}")
        
        # Graphics
        if self.quantitative_analysis["graphics"]:
            lines.append("\n### Visualizations")
            for graphic in self.quantitative_analysis["graphics"]:
                lines.append(f"\n#### {graphic.name}")
                if graphic.description:
                    lines.append(graphic.description)
                if graphic.image:
                    lines.append(f"\n![{graphic.name}]({graphic.image})")
        
        return "\n".join(lines)
    
    def export_proto(self, path: str) -> None:
        """Export as Protocol Buffer file."""
        content = self._render_google_proto()
        with open(path, 'w') as f:
            f.write(content)
    
    def export_json(self, path: str) -> None:
        """Export as JSON file."""
        content = self._render_google_json()
        with open(path, 'w') as f:
            f.write(content)
    
    def _remove_none_values(self, data: Any) -> Any:
        """Recursively remove None values from data structure."""
        if isinstance(data, dict):
            return {k: self._remove_none_values(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self._remove_none_values(item) for item in data if item is not None]
        else:
            return data
    
    def validate_google_schema(self) -> Dict[str, Any]:
        """Validate against Google Model Cards schema."""
        issues = []
        
        # Required fields
        if not self.google_model_details.name:
            issues.append("Missing required field: model_details.name")
        
        if not self.google_model_details.owners:
            issues.append("Missing required field: model_details.owners")
        
        # Check owners have required fields
        for i, owner in enumerate(self.google_model_details.owners):
            if not owner.name:
                issues.append(f"Missing name for owner {i}")
            if not owner.contact:
                issues.append(f"Missing contact for owner {i}")
        
        # Performance metrics validation
        if not self.quantitative_analysis["performance_metrics"]:
            issues.append("No performance metrics provided")
        
        for i, metric in enumerate(self.quantitative_analysis["performance_metrics"]):
            if not metric.type:
                issues.append(f"Missing type for metric {i}")
            if metric.value is None:
                issues.append(f"Missing value for metric {i}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "schema_version": self.schema_version
        }


def create_google_card_from_template(template_data: Dict[str, Any]) -> GoogleModelCard:
    """Create Google Model Card from template data."""
    card = GoogleModelCard()
    
    # Set model details
    if "model_details" in template_data:
        details = template_data["model_details"]
        card.set_model_details(
            name=details.get("name", ""),
            owners=details.get("owners", []),
            version=details.get("version"),
            model_type=details.get("type"),
            paper=details.get("paper"),
            citation=details.get("citation"),
            license=details.get("license"),
            path=details.get("path")
        )
    
    # Set considerations
    if "considerations" in template_data:
        considerations = template_data["considerations"]
        
        if "users" in considerations:
            card.add_intended_users(considerations["users"])
        
        if "use_cases" in considerations:
            card.add_use_cases(considerations["use_cases"])
        
        if "limitations" in considerations:
            card.add_limitations(considerations["limitations"])
        
        if "tradeoffs" in considerations:
            card.add_tradeoffs(considerations["tradeoffs"])
        
        if "ethical_considerations" in considerations:
            card.add_ethical_considerations(considerations["ethical_considerations"])
    
    # Set quantitative analysis
    if "quantitative_analysis" in template_data:
        qa = template_data["quantitative_analysis"]
        
        if "performance_metrics" in qa:
            for metric_data in qa["performance_metrics"]:
                card.add_performance_metric(
                    metric_type=metric_data.get("type", ""),
                    value=metric_data.get("value", 0.0),
                    confidence_interval=metric_data.get("confidence_interval"),
                    slice_name=metric_data.get("slice", "overall"),
                    description=metric_data.get("description")
                )
        
        if "graphics" in qa:
            for graphic_data in qa["graphics"]:
                card.add_graphic(
                    name=graphic_data.get("name", ""),
                    image_path=graphic_data.get("image"),
                    description=graphic_data.get("description")
                )
    
    return card