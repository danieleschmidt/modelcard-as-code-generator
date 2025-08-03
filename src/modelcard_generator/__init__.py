"""
Model Card as Code Generator

Automated generation of Model Cards as executable, versioned artifacts.
Converts evaluation results, dataset statistics, and training logs into
standardized documentation that satisfies regulatory requirements.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__license__ = "Apache-2.0"

from .core.generator import ModelCardGenerator
from .core.models import ModelCard, CardConfig
from .core.validator import Validator
from .core.drift_detector import DriftDetector
from .formats import HuggingFaceCard, GoogleModelCard, EUCRAModelCard
from .templates import TemplateLibrary

__all__ = [
    "ModelCardGenerator",
    "ModelCard", 
    "CardConfig",
    "Validator",
    "DriftDetector",
    "HuggingFaceCard",
    "GoogleModelCard", 
    "EUCRAModelCard",
    "TemplateLibrary",
]