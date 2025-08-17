"""
Monitoring and observability module for Model Card Generator.

This module provides comprehensive monitoring capabilities including:
- Application metrics collection
- Health checks
- Performance monitoring
- Error tracking
- Usage analytics
"""

from .health import HealthChecker
from .metrics import MetricsCollector

# Import from core logging if available
try:
    from ..core.logging_config import get_logger, setup_logging
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def setup_logging(level="INFO"):
        logging.basicConfig(level=level)

__all__ = [
    "HealthChecker", 
    "MetricsCollector",
    "get_logger",
    "setup_logging",
]
