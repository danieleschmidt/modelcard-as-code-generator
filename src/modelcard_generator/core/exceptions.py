"""Custom exceptions for model card generator."""

from typing import Any, Dict, List, Optional


class ModelCardError(Exception):
    """Base exception for model card generator."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(ModelCardError):
    """Exception raised when model card validation fails."""

    def __init__(self, message: str, issues: Optional[List[str]] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.issues = issues or []


class FormatError(ModelCardError):
    """Exception raised when format-specific operations fail."""

    def __init__(self, format_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Format '{format_name}': {message}", details)
        self.format_name = format_name


class TemplateError(ModelCardError):
    """Exception raised when template operations fail."""

    def __init__(self, template_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Template '{template_name}': {message}", details)
        self.template_name = template_name


class DataSourceError(ModelCardError):
    """Exception raised when data source operations fail."""

    def __init__(self, source_path: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Data source '{source_path}': {message}", details)
        self.source_path = source_path


class ComplianceError(ModelCardError):
    """Exception raised when compliance checks fail."""

    def __init__(self, standard: str, message: str, violations: Optional[List[str]] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Compliance '{standard}': {message}", details)
        self.standard = standard
        self.violations = violations or []


class DriftError(ModelCardError):
    """Exception raised when drift detection fails."""

    def __init__(self, message: str, metric_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.metric_name = metric_name


class SecurityError(ModelCardError):
    """Exception raised when security checks fail."""

    def __init__(self, message: str, vulnerability_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.vulnerability_type = vulnerability_type


class ConfigurationError(ModelCardError):
    """Exception raised when configuration is invalid."""

    def __init__(self, config_key: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Configuration '{config_key}': {message}", details)
        self.config_key = config_key


class ResourceError(ModelCardError):
    """Exception raised when resource operations fail."""

    def __init__(self, resource_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Resource '{resource_type}': {message}", details)
        self.resource_type = resource_type


class IntegrationError(ModelCardError):
    """Exception raised when external integration fails."""

    def __init__(self, integration_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Integration '{integration_name}': {message}", details)
        self.integration_name = integration_name


class RenderingError(ModelCardError):
    """Exception raised when rendering operations fail."""

    def __init__(self, format_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Rendering '{format_type}': {message}", details)
        self.format_type = format_type
