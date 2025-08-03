"""
Configuration management for model card generation.

Provides centralized configuration for all aspects of model card generation,
including format options, validation settings, and compliance requirements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml


@dataclass
class CardConfig:
    """Configuration for model card generation."""
    
    # Format settings
    format: str = "huggingface"  # huggingface, google, eu_cra, custom
    template_path: Optional[str] = None
    output_format: str = "markdown"  # markdown, json, html, pdf
    
    # Content settings
    include_ethical_considerations: bool = True
    include_carbon_footprint: bool = True
    include_bias_analysis: bool = True
    include_limitations: bool = True
    include_usage_guidelines: bool = True
    
    # Compliance settings
    regulatory_standard: Optional[str] = None  # gdpr, eu_ai_act, ccpa, iso_23053
    compliance_level: str = "standard"  # minimal, standard, comprehensive
    
    # Validation settings
    validate_schema: bool = True
    validate_content: bool = True
    validate_completeness: bool = True
    min_completeness_score: float = 0.8
    
    # Generation settings
    auto_populate: bool = True
    extract_from_logs: bool = True
    extract_from_config: bool = True
    extract_from_code: bool = False
    
    # Security settings
    scan_for_secrets: bool = True
    redact_sensitive_info: bool = True
    allow_internal_refs: bool = False
    
    # Output settings
    pretty_print: bool = True
    include_metadata: bool = True
    include_generation_info: bool = True
    
    # Advanced settings
    custom_sections: Dict[str, Any] = field(default_factory=dict)
    custom_validators: List[str] = field(default_factory=list)
    plugins: List[str] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, config_path: Path) -> "CardConfig":
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**data)
    
    def to_file(self, config_path: Path) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "format": self.format,
            "template_path": self.template_path,
            "output_format": self.output_format,
            "include_ethical_considerations": self.include_ethical_considerations,
            "include_carbon_footprint": self.include_carbon_footprint,
            "include_bias_analysis": self.include_bias_analysis,
            "include_limitations": self.include_limitations,
            "include_usage_guidelines": self.include_usage_guidelines,
            "regulatory_standard": self.regulatory_standard,
            "compliance_level": self.compliance_level,
            "validate_schema": self.validate_schema,
            "validate_content": self.validate_content,
            "validate_completeness": self.validate_completeness,
            "min_completeness_score": self.min_completeness_score,
            "auto_populate": self.auto_populate,
            "extract_from_logs": self.extract_from_logs,
            "extract_from_config": self.extract_from_config,
            "extract_from_code": self.extract_from_code,
            "scan_for_secrets": self.scan_for_secrets,
            "redact_sensitive_info": self.redact_sensitive_info,
            "allow_internal_refs": self.allow_internal_refs,
            "pretty_print": self.pretty_print,
            "include_metadata": self.include_metadata,
            "include_generation_info": self.include_generation_info,
            "custom_sections": self.custom_sections,
            "custom_validators": self.custom_validators,
            "plugins": self.plugins,
        }
    
    def update(self, **kwargs) -> "CardConfig":
        """Return a new config with updated values."""
        data = self.to_dict()
        data.update(kwargs)
        return CardConfig(**data)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        # Validate format
        supported_formats = ["huggingface", "google", "eu_cra", "custom"]
        if self.format not in supported_formats:
            raise ValueError(f"Unsupported format: {self.format}. Supported: {supported_formats}")
        
        # Validate output format
        supported_outputs = ["markdown", "json", "html", "pdf"]
        if self.output_format not in supported_outputs:
            raise ValueError(f"Unsupported output format: {self.output_format}. Supported: {supported_outputs}")
        
        # Validate compliance level
        supported_levels = ["minimal", "standard", "comprehensive"]
        if self.compliance_level not in supported_levels:
            raise ValueError(f"Unsupported compliance level: {self.compliance_level}. Supported: {supported_levels}")
        
        # Validate completeness score
        if not 0.0 <= self.min_completeness_score <= 1.0:
            raise ValueError("min_completeness_score must be between 0.0 and 1.0")
        
        # Validate regulatory standard if provided
        if self.regulatory_standard:
            supported_standards = ["gdpr", "eu_ai_act", "ccpa", "iso_23053"]
            if self.regulatory_standard not in supported_standards:
                raise ValueError(f"Unsupported regulatory standard: {self.regulatory_standard}. Supported: {supported_standards}")


@dataclass
class ValidationConfig:
    """Configuration for validation settings."""
    
    schema_validation: bool = True
    content_validation: bool = True
    compliance_validation: bool = True
    security_validation: bool = True
    
    # Schema validation
    strict_schema: bool = False
    allow_extra_fields: bool = True
    
    # Content validation
    min_section_length: int = 10
    required_sections: List[str] = field(default_factory=list)
    check_language_quality: bool = True
    
    # Compliance validation
    enforce_compliance: bool = False
    compliance_standards: List[str] = field(default_factory=list)
    
    # Security validation
    scan_secrets: bool = True
    check_pii: bool = True
    validate_urls: bool = True


@dataclass
class GenerationContext:
    """Context information for model card generation."""
    
    # Source information
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    
    # Generation metadata
    generated_by: str = "modelcard-as-code-generator"
    generation_timestamp: Optional[str] = None
    config_used: Optional[CardConfig] = None
    
    # Source files
    source_files: Dict[str, str] = field(default_factory=dict)
    
    # Processing information
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0