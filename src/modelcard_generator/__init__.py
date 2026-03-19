"""
modelcard-as-code-generator
============================
Generate standardised ML Model Cards from YAML/JSON config files.

Quick start::

    from modelcard_generator import ModelCardGenerator

    gen = ModelCardGenerator()
    card = gen.from_config("mymodel.yaml")
    card.save("MODEL_CARD.md")
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"
__license__ = "MIT"

from .core.drift_detector import DriftDetector
from .core.generator import ModelCardGenerator
from .core.models import CardConfig, CardFormat, ModelCard
from .core.validator import Validator

__all__ = [
    "ModelCardGenerator",
    "ModelCard",
    "CardConfig",
    "CardFormat",
    "Validator",
    "DriftDetector",
]
