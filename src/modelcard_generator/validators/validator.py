"""
Core validation system for model cards.

Provides comprehensive validation including schema validation,
content quality checks, and compliance verification.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import json
import jsonschema
import logging
from pathlib import Path

from ..core.model_card import ModelCard


@dataclass
class ValidationResult:
    """Result of model card validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class Validator:
    """Main validator for model cards."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schema_validators = {}
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """Load validation schemas for different formats."""
        # For now, we'll define schemas inline
        # In production, these would be loaded from files
        
        self.schema_validators["huggingface"] = {
            "type": "object",
            "required": ["model_details", "intended_use"],
            "properties": {
                "model_details": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "version": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "architecture": {"type": ["string", "null"]},
                        "license": {"type": ["string", "null"]},
                        "languages": {"type": "array", "items": {"type": "string"}},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "intended_use": {"type": "object"},
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "value"],
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": ["number", "string"]},
                            "unit": {"type": ["string", "null"]},
                            "description": {"type": ["string", "null"]}
                        }
                    }
                }
            }
        }
        
        self.schema_validators["google"] = {
            "type": "object",
            "required": ["model_details", "model_parameters", "quantitative_analysis"],
            "properties": {
                "model_details": {
                    "type": "object",
                    "required": ["name", "version"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "version": {"type": "object"},
                        "type": {"type": "string"},
                        "paper": {"type": ["string", "null"]},
                        "license": {"type": ["string", "null"]}
                    }
                },
                "model_parameters": {"type": "object"},
                "quantitative_analysis": {"type": "object"}
            }
        }
        
        self.schema_validators["eu_cra"] = {
            "type": "object",
            "required": ["intended_purpose", "risk_assessment", "technical_robustness"],
            "properties": {
                "intended_purpose": {
                    "type": "object",
                    "required": ["description"],
                    "properties": {
                        "description": {"type": "string", "minLength": 10},
                        "deployment_context": {"type": "string"},
                        "geographic_restrictions": {"type": "array"}
                    }
                },
                "risk_assessment": {
                    "type": "object",
                    "required": ["risk_level"],
                    "properties": {
                        "risk_level": {
                            "type": "string",
                            "enum": ["minimal", "limited", "high", "unacceptable"]
                        },
                        "mitigation_measures": {"type": "array"},
                        "risk_factors": {"type": "array"}
                    }
                },
                "technical_robustness": {"type": "object"}
            }
        }
    
    def validate(self, model_card: ModelCard, format_name: str = "huggingface") -> ValidationResult:
        """
        Validate model card against specified format.
        
        Args:
            model_card: ModelCard to validate
            format_name: Format to validate against
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        details = {}
        
        try:
            # 1. Schema validation
            schema_result = self._validate_schema(model_card, format_name)
            errors.extend(schema_result["errors"])
            warnings.extend(schema_result["warnings"])
            details["schema"] = schema_result
            
            # 2. Content validation
            content_result = self._validate_content(model_card)
            errors.extend(content_result["errors"])
            warnings.extend(content_result["warnings"])
            details["content"] = content_result
            
            # 3. Completeness validation
            completeness_result = self._validate_completeness(model_card, format_name)
            warnings.extend(completeness_result["warnings"])
            details["completeness"] = completeness_result
            
            # 4. Quality validation
            quality_result = self._validate_quality(model_card)
            warnings.extend(quality_result["warnings"])
            details["quality"] = quality_result
            
            # Calculate overall score
            score = self._calculate_score(details)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                score=score,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                details={}
            )
    
    def _validate_schema(self, model_card: ModelCard, format_name: str) -> Dict[str, Any]:
        """Validate against JSON schema."""
        errors = []
        warnings = []
        
        if format_name not in self.schema_validators:
            warnings.append(f"No schema available for format: {format_name}")
            return {"errors": errors, "warnings": warnings, "valid": True}
        
        schema = self.schema_validators[format_name]
        data = model_card.to_dict()
        
        try:
            # Validate against schema
            jsonschema.validate(data, schema)
            return {"errors": errors, "warnings": warnings, "valid": True}
            
        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation error: {e.message}"
            if e.absolute_path:
                error_msg += f" at path: {'.'.join(str(p) for p in e.absolute_path)}"
            errors.append(error_msg)
            
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        
        return {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}
    
    def _validate_content(self, model_card: ModelCard) -> Dict[str, Any]:
        """Validate content quality and consistency."""
        errors = []
        warnings = []
        
        # Check required model name
        if not model_card.model_details.name:
            errors.append("Model name is required")
        elif len(model_card.model_details.name.strip()) < 3:
            warnings.append("Model name is very short")
        
        # Check metrics consistency
        if model_card.metrics:
            metric_names = [m.name for m in model_card.metrics]
            if len(metric_names) != len(set(metric_names)):
                warnings.append("Duplicate metric names found")
            
            for metric in model_card.metrics:
                if not metric.name:
                    errors.append("Metric name cannot be empty")
                if metric.value is None:
                    errors.append(f"Metric '{metric.name}' has no value")
        
        # Check dataset information
        if not model_card.training_data and not model_card.evaluation_data:
            warnings.append("No training or evaluation datasets specified")
        
        # Check for basic description
        if not model_card.model_details.description:
            warnings.append("Model description is missing")
        elif len(model_card.model_details.description.strip()) < 20:
            warnings.append("Model description is very brief")
        
        return {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}
    
    def _validate_completeness(self, model_card: ModelCard, format_name: str) -> Dict[str, Any]:
        """Validate completeness based on format requirements."""
        warnings = []
        completeness_score = model_card.get_completeness_score()
        
        # Format-specific completeness checks
        if format_name == "huggingface":
            required_sections = ["model_details", "intended_use"]
            optional_sections = ["metrics", "ethical_considerations", "caveats_and_recommendations"]
            
        elif format_name == "google":
            required_sections = ["model_details", "quantitative_analysis"]
            optional_sections = ["considerations"]
            
        elif format_name == "eu_cra":
            required_sections = ["intended_purpose", "risk_assessment", "technical_robustness"]
            optional_sections = ["data_governance", "transparency", "human_oversight"]
            
        else:
            required_sections = ["model_details"]
            optional_sections = ["metrics", "intended_use"]
        
        # Check required sections
        missing_required = []
        for section in required_sections:
            if not model_card.get_section(section):
                missing_required.append(section)
        
        if missing_required:
            warnings.append(f"Missing required sections: {', '.join(missing_required)}")
        
        # Check optional sections
        missing_optional = []
        for section in optional_sections:
            if not model_card.get_section(section):
                missing_optional.append(section)
        
        if missing_optional:
            warnings.append(f"Missing recommended sections: {', '.join(missing_optional)}")
        
        # Overall completeness warning
        if completeness_score < 0.7:
            warnings.append(f"Low completeness score: {completeness_score:.2f}")
        
        return {
            "warnings": warnings,
            "score": completeness_score,
            "missing_required": missing_required,
            "missing_optional": missing_optional
        }
    
    def _validate_quality(self, model_card: ModelCard) -> Dict[str, Any]:
        """Validate quality aspects of the model card."""
        warnings = []
        quality_issues = []
        
        # Check text quality
        text_fields = [
            model_card.model_details.description,
            model_card.intended_use.get('primary_use') if model_card.intended_use else None,
        ]
        
        for text in text_fields:
            if text and isinstance(text, str):
                # Check for placeholder text
                placeholders = ['todo', 'tbd', 'placeholder', 'not specified', 'unknown']
                if any(placeholder in text.lower() for placeholder in placeholders):
                    quality_issues.append("Contains placeholder text")
                
                # Check for very short descriptions
                if len(text.strip()) < 10:
                    quality_issues.append("Very brief descriptions")
        
        # Check metric quality
        if model_card.metrics:
            for metric in model_card.metrics:
                # Check for reasonable metric values
                if isinstance(metric.value, (int, float)):
                    # Common metric ranges
                    if metric.name.lower() in ['accuracy', 'precision', 'recall', 'f1']:
                        if not 0 <= metric.value <= 1 and not 0 <= metric.value <= 100:
                            quality_issues.append(f"Unusual value for {metric.name}: {metric.value}")
                    
                    # Check for suspiciously perfect metrics
                    if metric.value == 1.0 or metric.value == 100.0:
                        warnings.append(f"Perfect score for {metric.name} - please verify")
        
        # Check consistency
        if model_card.model_details.languages:
            if len(model_card.model_details.languages) > 10:
                warnings.append("Large number of languages specified - please verify")
        
        if quality_issues:
            warnings.extend(quality_issues)
        
        return {
            "warnings": warnings,
            "quality_issues": quality_issues,
            "text_quality": "good" if not quality_issues else "needs_improvement"
        }
    
    def _calculate_score(self, details: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        score = 1.0
        
        # Schema validation (most important)
        if not details.get("schema", {}).get("valid", True):
            score -= 0.4
        
        # Content validation
        if not details.get("content", {}).get("valid", True):
            score -= 0.3
        
        # Completeness score
        completeness = details.get("completeness", {}).get("score", 1.0)
        score *= completeness
        
        # Quality penalties
        quality_issues = len(details.get("quality", {}).get("quality_issues", []))
        if quality_issues > 0:
            score -= min(0.2, quality_issues * 0.05)
        
        return max(0.0, score)
    
    def validate_dict(self, data: Dict[str, Any], format_name: str = "huggingface") -> ValidationResult:
        """Validate dictionary data directly."""
        try:
            model_card = ModelCard.from_dict(data)
            return self.validate(model_card, format_name)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Failed to create model card from data: {str(e)}"],
                warnings=[],
                details={}
            )
    
    def validate_file(self, file_path: Path, format_name: str = "huggingface") -> ValidationResult:
        """Validate model card from file."""
        try:
            model_card = ModelCard.load(file_path)
            return self.validate(model_card, format_name)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=[f"Failed to load model card from file: {str(e)}"],
                warnings=[],
                details={}
            )
    
    def get_validation_suggestions(self, result: ValidationResult) -> List[str]:
        """Get suggestions for improving validation score."""
        suggestions = []
        
        if not result.is_valid:
            suggestions.append("Fix validation errors first")
        
        if result.score < 0.8:
            suggestions.append("Improve completeness by adding missing sections")
        
        if result.warnings:
            warning_types = set()
            for warning in result.warnings:
                if "missing" in warning.lower():
                    warning_types.add("completeness")
                elif "placeholder" in warning.lower():
                    warning_types.add("content_quality")
                elif "brief" in warning.lower():
                    warning_types.add("detail")
            
            if "completeness" in warning_types:
                suggestions.append("Add missing required and recommended sections")
            if "content_quality" in warning_types:
                suggestions.append("Replace placeholder text with actual content")
            if "detail" in warning_types:
                suggestions.append("Provide more detailed descriptions")
        
        return suggestions