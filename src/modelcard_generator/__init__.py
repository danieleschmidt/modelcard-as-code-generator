"""
Model Card as Code Generator

Automated generation of Model Cards as executable, versioned artifacts.
Converts evaluation results, dataset statistics, and training logs into standardized
documentation that satisfies regulatory requirements and enables "card drift" detection in CI/CD.
"""

from .core.generator import ModelCardGenerator
from .core.config import CardConfig
from .core.model_card import ModelCard
from .templates.registry import TemplateRegistry
from .validators.validator import Validator
from .drift.detector import DriftDetector
from .compliance.checker import ComplianceChecker

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "team@terragonlabs.com"

__all__ = [
    "ModelCardGenerator",
    "CardConfig", 
    "ModelCard",
    "TemplateRegistry",
    "Validator",
    "DriftDetector",
    "ComplianceChecker",
]

# Package metadata
PACKAGE_NAME = "modelcard-as-code-generator"
SUPPORTED_FORMATS = ["huggingface", "google", "eu_cra", "custom"]
SUPPORTED_STANDARDS = ["gdpr", "eu_ai_act", "ccpa", "iso_23053"]