"""Logging configuration and utilities."""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager

from .exceptions import ModelCardError


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add custom fields from extra
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add context if available
        if self.include_context:
            context = getattr(record, 'context', {})
            if context:
                log_entry["context"] = context
        
        return json.dumps(log_entry, ensure_ascii=False)


class ModelCardLogger:
    """Enhanced logger for model card operations."""
    
    def __init__(self, name: str = "modelcard_generator"):
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
        self._configured = False
    
    def configure(
        self,
        level: Union[str, int] = logging.INFO,
        log_file: Optional[str] = None,
        structured: bool = False,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """Configure logger with specified settings."""
        if self._configured:
            return
        
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            if structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
            
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)
        
        self._configured = True
    
    def set_context(self, **kwargs) -> None:
        """Set context fields for all subsequent log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context fields."""
        self._context.clear()
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary context fields."""
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context
    
    def _log_with_context(self, level: int, message: str, extra_fields: Optional[Dict[str, Any]] = None) -> None:
        """Log message with context and extra fields."""
        combined_extra = {**self._context}
        if extra_fields:
            combined_extra.update(extra_fields)
        
        # Create extra dict for logger
        extra = {'extra_fields': combined_extra} if combined_extra else {}
        
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, kwargs)
    
    def exception(self, message: str, exc_info=True, **kwargs) -> None:
        """Log exception with traceback."""
        combined_extra = {**self._context, **kwargs}
        extra = {'extra_fields': combined_extra} if combined_extra else {}
        self.logger.exception(message, exc_info=exc_info, extra=extra)
    
    def log_operation_start(self, operation: str, **kwargs) -> None:
        """Log the start of an operation."""
        self.info(f"Starting {operation}", operation=operation, status="started", **kwargs)
    
    def log_operation_success(self, operation: str, duration_ms: Optional[float] = None, **kwargs) -> None:
        """Log successful completion of an operation."""
        extra = {"operation": operation, "status": "completed", **kwargs}
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        self.info(f"Successfully completed {operation}", **extra)
    
    def log_operation_failure(self, operation: str, error: Exception, duration_ms: Optional[float] = None, **kwargs) -> None:
        """Log failure of an operation."""
        extra = {
            "operation": operation,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs
        }
        
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        
        # Add additional context for ModelCardError
        if isinstance(error, ModelCardError):
            extra["error_details"] = error.details
        
        self.error(f"Failed to complete {operation}: {error}", **extra)
    
    def log_validation_result(self, is_valid: bool, score: float, issues: Optional[list] = None, **kwargs) -> None:
        """Log validation results."""
        extra = {
            "validation_valid": is_valid,
            "validation_score": score,
            **kwargs
        }
        
        if issues:
            extra["validation_issues"] = len(issues)
        
        if is_valid:
            self.info(f"Validation passed with score {score:.2%}", **extra)
        else:
            self.warning(f"Validation failed with score {score:.2%}", **extra)
    
    def log_metric_change(self, metric_name: str, old_value: float, new_value: float, is_significant: bool, **kwargs) -> None:
        """Log metric changes for drift detection."""
        delta = new_value - old_value
        extra = {
            "metric_name": metric_name,
            "metric_old_value": old_value,
            "metric_new_value": new_value,
            "metric_delta": delta,
            "metric_significant_change": is_significant,
            **kwargs
        }
        
        if is_significant:
            self.warning(f"Significant change in {metric_name}: {old_value:.4f} → {new_value:.4f} (Δ{delta:+.4f})", **extra)
        else:
            self.info(f"Minor change in {metric_name}: {old_value:.4f} → {new_value:.4f} (Δ{delta:+.4f})", **extra)
    
    def log_security_check(self, check_name: str, passed: bool, details: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log security check results."""
        extra = {
            "security_check": check_name,
            "security_passed": passed,
            **kwargs
        }
        
        if details:
            extra["security_details"] = details
        
        if passed:
            self.info(f"Security check '{check_name}' passed", **extra)
        else:
            self.error(f"Security check '{check_name}' failed", **extra)


# Global logger instance
logger = ModelCardLogger()


def get_logger(name: Optional[str] = None) -> ModelCardLogger:
    """Get logger instance."""
    if name:
        return ModelCardLogger(name)
    return logger


def configure_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    structured: bool = False,
    **kwargs
) -> None:
    """Configure global logging settings."""
    logger.configure(level=level, log_file=log_file, structured=structured, **kwargs)