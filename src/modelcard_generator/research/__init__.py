"""Research and AI-powered enhancement modules for model card generation."""

from .ai_content_generator import AIContentGenerator
from .algorithm_optimizer import AlgorithmOptimizer
from .insight_engine import InsightEngine

# Conditionally import modules that require optional dependencies
try:
    from .research_analyzer import ResearchAnalyzer
    RESEARCH_ANALYZER_AVAILABLE = True
except ImportError:
    RESEARCH_ANALYZER_AVAILABLE = False

try:
    from .novelty_detector import NoveltyDetector
    NOVELTY_DETECTOR_AVAILABLE = True
except ImportError:
    NOVELTY_DETECTOR_AVAILABLE = False

__all__ = [
    "AIContentGenerator",
    "AlgorithmOptimizer", 
    "InsightEngine"
]

if RESEARCH_ANALYZER_AVAILABLE:
    __all__.append("ResearchAnalyzer")

if NOVELTY_DETECTOR_AVAILABLE:
    __all__.append("NoveltyDetector")