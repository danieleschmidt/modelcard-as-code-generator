"""Model card validation and compliance checking."""

import re
from enum import Enum
from typing import Any, Dict, List

from .models import ModelCard, ValidationIssue, ValidationResult, ValidationSeverity


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    EU_CRA = "eu_cra"
    GDPR = "gdpr"
    EU_AI_ACT = "eu_ai_act"
    ISO_23053 = "iso_23053"


class Validator:
    """Validate model cards for completeness, quality, and compliance."""

    def __init__(self):
        self.schemas = self._load_schemas()
        self.compliance_rules = self._load_compliance_rules()

    def validate_schema(self, card: ModelCard, schema: str = "huggingface") -> ValidationResult:
        """Validate card against a specific schema."""
        issues = []
        missing_sections = []

        # Get schema requirements
        schema_rules = self.schemas.get(schema, {})
        required_sections = schema_rules.get("required_sections", [])
        optional_sections = schema_rules.get("optional_sections", [])

        # Check required sections
        for section in required_sections:
            if not self._has_section(card, section):
                missing_sections.append(section)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required section: {section}",
                    path=section,
                    suggestion=f"Add {section} section to your model card"
                ))

        # Check field requirements
        field_rules = schema_rules.get("field_requirements", {})
        issues.extend(self._validate_fields(card, field_rules))

        # Check format-specific requirements
        if schema == "huggingface":
            issues.extend(self._validate_huggingface_specific(card))
        elif schema == "google":
            issues.extend(self._validate_google_specific(card))
        elif schema == "eu_cra":
            issues.extend(self._validate_eu_cra_specific(card))

        # Calculate score
        total_checks = len(required_sections) + len(optional_sections) + len(field_rules)
        passed_checks = total_checks - len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        score = passed_checks / total_checks if total_checks > 0 else 1.0

        is_valid = len([i for i in issues if i.severity == ValidationSeverity.ERROR]) == 0

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            missing_sections=missing_sections
        )

    def check_completeness(self, card: ModelCard, min_score: float = 0.8) -> ValidationResult:
        """Check card completeness against best practices."""
        issues = []
        missing_sections = []

        # Essential sections
        essential_sections = [
            "model_details.name",
            "model_details.version",
            "intended_use",
            "evaluation_results",
            "limitations.known_limitations"
        ]

        for section in essential_sections:
            if not self._has_nested_field(card, section):
                missing_sections.append(section)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing recommended section: {section}",
                    path=section,
                    suggestion=f"Consider adding {section} for better documentation"
                ))

        # Check metrics quality
        if card.evaluation_results:
            issues.extend(self._validate_metrics_quality(card.evaluation_results))
        else:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No evaluation metrics found",
                suggestion="Add performance metrics to demonstrate model capabilities"
            ))

        # Check ethical considerations
        if not card.ethical_considerations.bias_risks and not card.ethical_considerations.fairness_metrics:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No ethical considerations documented",
                suggestion="Add bias analysis and fairness considerations"
            ))

        # Calculate completeness score
        total_checks = len(essential_sections) + 3  # +3 for metrics, ethics, etc.
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])

        # Errors count as -1, warnings as -0.5
        score = max(0, (total_checks - error_count - 0.5 * warning_count) / total_checks)

        return ValidationResult(
            is_valid=score >= min_score,
            score=score,
            issues=issues,
            missing_sections=missing_sections
        )

    def check_quality(self, card: ModelCard) -> ValidationResult:
        """Check content quality and best practices."""
        issues = []

        # Check for placeholder text
        issues.extend(self._check_placeholder_content(card))

        # Check description quality
        if card.model_details.description:
            if len(card.model_details.description) < 50:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Model description is very short",
                    path="model_details.description",
                    suggestion="Provide a more detailed description of the model's purpose and capabilities"
                ))

        # Check intended use clarity
        if card.intended_use:
            if len(card.intended_use.split()) < 10:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Intended use description is very brief",
                    path="intended_use",
                    suggestion="Provide more detailed information about intended use cases"
                ))

        # Check for sufficient limitations
        if len(card.limitations.known_limitations) < 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Very few limitations documented",
                path="limitations.known_limitations",
                suggestion="Document additional known limitations and edge cases"
            ))

        # Check metric naming conventions
        issues.extend(self._validate_metric_names(card.evaluation_results))

        # Calculate quality score
        total_checks = 5
        warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])

        score = max(0, (total_checks - error_count - 0.3 * warning_count) / total_checks)

        return ValidationResult(
            is_valid=error_count == 0,
            score=score,
            issues=issues
        )

    def validate_compliance(self, card: ModelCard, standard: ComplianceStandard) -> ValidationResult:
        """Validate compliance with regulatory standards."""
        issues = []
        missing_sections = []

        rules = self.compliance_rules.get(standard.value, {})

        if standard == ComplianceStandard.GDPR:
            issues.extend(self._validate_gdpr_compliance(card))
        elif standard == ComplianceStandard.EU_AI_ACT:
            issues.extend(self._validate_eu_ai_act_compliance(card))
        elif standard == ComplianceStandard.EU_CRA:
            issues.extend(self._validate_eu_cra_compliance(card))

        # Check required fields for standard
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if not self._has_nested_field(card, field):
                missing_sections.append(field)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required field for {standard.value}: {field}",
                    path=field
                ))

        # Calculate compliance score
        total_requirements = len(required_fields) + len(rules.get("additional_checks", []))
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])

        score = max(0, (total_requirements - error_count) / total_requirements) if total_requirements > 0 else 1.0

        return ValidationResult(
            is_valid=error_count == 0,
            score=score,
            issues=issues,
            missing_sections=missing_sections
        )

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load validation schemas."""
        return {
            "huggingface": {
                "required_sections": [
                    "model_details.name",
                    "model_details.description",
                    "intended_use",
                    "training_details.training_data",
                    "evaluation_results"
                ],
                "optional_sections": [
                    "model_details.license",
                    "model_details.language",
                    "limitations.known_limitations",
                    "ethical_considerations.bias_risks"
                ],
                "field_requirements": {
                    "model_details.name": {"min_length": 1},
                    "model_details.description": {"min_length": 20},
                    "intended_use": {"min_length": 10}
                }
            },
            "google": {
                "required_sections": [
                    "model_details.name",
                    "model_details.version",
                    "model_details.description",
                    "intended_use",
                    "evaluation_results",
                    "ethical_considerations",
                    "limitations.known_limitations"
                ],
                "field_requirements": {
                    "model_details.version": {"pattern": r"^\d+\.\d+\.\d+$"},
                    "model_details.description": {"min_length": 50}
                }
            },
            "eu_cra": {
                "required_sections": [
                    "model_details.name",
                    "model_details.version",
                    "intended_use",
                    "training_details",
                    "evaluation_results",
                    "ethical_considerations.bias_risks",
                    "limitations.known_limitations",
                    "limitations.sensitive_use_cases"
                ],
                "field_requirements": {
                    "intended_use": {"must_include": ["purpose", "context"]},
                    "limitations.sensitive_use_cases": {"min_items": 1}
                }
            }
        }

    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance validation rules."""
        return {
            "gdpr": {
                "required_fields": [
                    "training_details.training_data",
                    "ethical_considerations.sensitive_attributes",
                    "limitations.known_limitations"
                ],
                "additional_checks": ["data_protection", "consent_mechanism"]
            },
            "eu_ai_act": {
                "required_fields": [
                    "intended_use",
                    "ethical_considerations.bias_risks",
                    "limitations.sensitive_use_cases",
                    "evaluation_results"
                ],
                "additional_checks": ["risk_assessment", "human_oversight"]
            },
            "eu_cra": {
                "required_fields": [
                    "model_details.version",
                    "training_details",
                    "evaluation_results",
                    "limitations.known_limitations"
                ],
                "additional_checks": ["technical_robustness", "security_measures"]
            }
        }

    def _has_section(self, card: ModelCard, section: str) -> bool:
        """Check if card has a specific section."""
        return self._has_nested_field(card, section)

    def _has_nested_field(self, card: ModelCard, field_path: str) -> bool:
        """Check if card has a nested field using dot notation."""
        parts = field_path.split(".")
        obj = card

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
                if obj is None:
                    return False
            else:
                return False

        # Check if field has meaningful content
        if isinstance(obj, str):
            return bool(obj.strip())
        elif isinstance(obj, list):
            return len(obj) > 0
        elif isinstance(obj, dict):
            return len(obj) > 0
        else:
            return obj is not None

    def _validate_fields(self, card: ModelCard, field_rules: Dict[str, Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate fields against specific rules."""
        issues = []

        for field_path, rules in field_rules.items():
            if not self._has_nested_field(card, field_path):
                continue

            value = self._get_nested_field(card, field_path)

            # Check minimum length
            if "min_length" in rules and isinstance(value, str):
                if len(value) < rules["min_length"]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Field {field_path} is too short (minimum {rules['min_length']} characters)",
                        path=field_path,
                        suggestion=f"Provide more detailed information for {field_path}"
                    ))

            # Check pattern matching
            if "pattern" in rules and isinstance(value, str):
                if not re.match(rules["pattern"], value):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Field {field_path} does not match required pattern",
                        path=field_path,
                        suggestion=f"Ensure {field_path} follows the correct format"
                    ))

            # Check required content
            if "must_include" in rules and isinstance(value, str):
                missing_terms = [term for term in rules["must_include"] if term.lower() not in value.lower()]
                if missing_terms:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Field {field_path} should include: {', '.join(missing_terms)}",
                        path=field_path,
                        suggestion=f"Consider including {', '.join(missing_terms)} in {field_path}"
                    ))

            # Check minimum items for lists
            if "min_items" in rules and isinstance(value, list):
                if len(value) < rules["min_items"]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Field {field_path} should have at least {rules['min_items']} items",
                        path=field_path,
                        suggestion=f"Add more items to {field_path}"
                    ))

        return issues

    def _get_nested_field(self, card: ModelCard, field_path: str) -> Any:
        """Get nested field value using dot notation."""
        parts = field_path.split(".")
        obj = card

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None

        return obj

    def _validate_huggingface_specific(self, card: ModelCard) -> List[ValidationIssue]:
        """Validate Hugging Face specific requirements."""
        issues = []

        # Check for license
        if not card.model_details.license:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="License not specified",
                path="model_details.license",
                suggestion="Specify a license for your model"
            ))

        # Check for language tags
        if not card.model_details.language:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Language not specified",
                path="model_details.language",
                suggestion="Add language tags for better discoverability"
            ))

        return issues

    def _validate_google_specific(self, card: ModelCard) -> List[ValidationIssue]:
        """Validate Google Model Cards specific requirements."""
        issues = []

        # Check for quantitative analysis
        if not card.evaluation_results:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No quantitative analysis provided",
                path="evaluation_results",
                suggestion="Add performance metrics and evaluation results"
            ))

        # Check for ethical considerations
        if not card.ethical_considerations.bias_risks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No bias analysis provided",
                path="ethical_considerations.bias_risks",
                suggestion="Include bias analysis and fairness considerations"
            ))

        return issues

    def _validate_eu_cra_specific(self, card: ModelCard) -> List[ValidationIssue]:
        """Validate EU CRA specific requirements."""
        issues = []

        # Check for risk assessment
        if not card.limitations.sensitive_use_cases:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No sensitive use cases documented",
                path="limitations.sensitive_use_cases",
                suggestion="Document use cases where the model should not be used"
            ))

        # Check for technical robustness
        technical_metrics = ["accuracy", "precision", "recall", "f1"]
        found_metrics = [m.name for m in card.evaluation_results if m.name in technical_metrics]
        if not found_metrics:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No technical robustness metrics found",
                path="evaluation_results",
                suggestion="Include accuracy, precision, recall, or F1 score"
            ))

        return issues

    def _validate_gdpr_compliance(self, card: ModelCard) -> List[ValidationIssue]:
        """Validate GDPR compliance."""
        issues = []

        # Check for data protection measures
        if not card.ethical_considerations.sensitive_attributes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No sensitive attributes documented",
                path="ethical_considerations.sensitive_attributes",
                suggestion="Document which sensitive attributes the model may process"
            ))

        # Check for training data documentation
        if not card.training_details.training_data:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Training data not documented",
                path="training_details.training_data",
                suggestion="Document the source and nature of training data"
            ))

        return issues

    def _validate_eu_ai_act_compliance(self, card: ModelCard) -> List[ValidationIssue]:
        """Validate EU AI Act compliance."""
        issues = []

        # Check for bias risks
        if not card.ethical_considerations.bias_risks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="No bias risks documented",
                path="ethical_considerations.bias_risks",
                suggestion="Document potential bias risks and mitigation measures"
            ))

        # Check for human oversight considerations
        if "human oversight" not in card.intended_use.lower():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Human oversight not mentioned",
                path="intended_use",
                suggestion="Consider documenting human oversight requirements"
            ))

        return issues

    def _check_placeholder_content(self, card: ModelCard) -> List[ValidationIssue]:
        """Check for placeholder or template content."""
        issues = []

        placeholder_patterns = [
            r"(?i)\[.*\]",  # [placeholder text]
            r"(?i)todo",
            r"(?i)fixme",
            r"(?i)placeholder",
            r"(?i)example",
            r"(?i)lorem ipsum"
        ]

        # Check various text fields
        text_fields = [
            ("model_details.description", card.model_details.description),
            ("intended_use", card.intended_use),
            ("training_details.preprocessing", card.training_details.preprocessing)
        ]

        for field_path, text in text_fields:
            if text:
                for pattern in placeholder_patterns:
                    if re.search(pattern, text):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Placeholder content detected in {field_path}",
                            path=field_path,
                            suggestion=f"Replace placeholder text with actual content in {field_path}"
                        ))
                        break

        return issues

    def _validate_metrics_quality(self, metrics: List) -> List[ValidationIssue]:
        """Validate quality of evaluation metrics."""
        issues = []

        if not metrics:
            return issues

        # Check for reasonable metric values
        for metric in metrics:
            if hasattr(metric, "name") and hasattr(metric, "value"):
                # Check for suspicious values
                if metric.name.lower() in ["accuracy", "precision", "recall", "f1"] and metric.value > 1.0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Metric {metric.name} has value > 1.0, which may be incorrect",
                        path=f"evaluation_results.{metric.name}",
                        suggestion="Verify metric values are in correct range (0-1 for most metrics)"
                    ))

                if metric.value < 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Metric {metric.name} has negative value",
                        path=f"evaluation_results.{metric.name}",
                        suggestion="Metrics should have non-negative values"
                    ))

        return issues

    def _validate_metric_names(self, metrics: List) -> List[ValidationIssue]:
        """Validate metric naming conventions."""
        issues = []

        standard_names = {
            "acc": "accuracy",
            "prec": "precision",
            "rec": "recall",
            "f1": "f1_score",
            "auc": "roc_auc"
        }

        for metric in metrics:
            if hasattr(metric, "name"):
                name = metric.name.lower()
                if name in standard_names:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Consider using standard name '{standard_names[name]}' instead of '{metric.name}'",
                        path=f"evaluation_results.{metric.name}",
                        suggestion=f"Use standardized metric name: {standard_names[name]}"
                    ))

        return issues


class ContentValidator:
    """Advanced content validation for model cards."""

    def check_completeness(self, card: ModelCard, min_score: float = 0.8) -> ValidationResult:
        """Check card completeness with detailed scoring."""
        validator = Validator()
        return validator.check_completeness(card, min_score)

    def check_quality(self, card: ModelCard) -> ValidationResult:
        """Check content quality with readability and completeness metrics."""
        validator = Validator()
        return validator.check_quality(card)


class ComplianceChecker:
    """Check compliance with multiple regulatory standards."""

    def __init__(self):
        self.validator = Validator()

    def check(self, card: ModelCard, standard: str) -> ValidationResult:
        """Check compliance with a specific standard."""
        try:
            compliance_standard = ComplianceStandard(standard)
            return self.validator.validate_compliance(card, compliance_standard)
        except ValueError:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Unknown compliance standard: {standard}",
                    suggestion="Use one of: gdpr, eu_ai_act, eu_cra, iso_23053"
                )]
            )
