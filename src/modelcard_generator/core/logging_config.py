"""Minimal logging configuration."""

import logging
from typing import Optional


class _Logger:
    """Thin wrapper around stdlib logging."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def debug(self, msg: str, **kw) -> None:
        self._log.debug(msg)

    def info(self, msg: str, **kw) -> None:
        self._log.info(msg)

    def warning(self, msg: str, **kw) -> None:
        self._log.warning(msg)

    def error(self, msg: str, **kw) -> None:
        self._log.error(msg)

    def set_context(self, **kw) -> None:
        pass

    def log_operation_start(self, op: str, **kw) -> None:
        self._log.debug(f"Starting: {op}")

    def log_operation_success(self, op: str, duration_ms: float = 0, **kw) -> None:
        self._log.debug(f"Done: {op} ({duration_ms:.0f}ms)")

    def log_operation_failure(self, op: str, exc: Exception, duration_ms: float = 0) -> None:
        self._log.error(f"Failed: {op} — {exc}")

    def log_validation_result(self, valid: bool, score: float, issues) -> None:
        self._log.debug(f"Validation: valid={valid} score={score:.2f}")

    def log_security_check(self, check: str, passed: bool, details=None) -> None:
        self._log.debug(f"Security check {check}: passed={passed}")

    def context(self, **kw):
        from contextlib import contextmanager
        @contextmanager
        def _ctx():
            yield
        return _ctx()


def get_logger(name: str) -> _Logger:
    return _Logger(name)


def configure_logging(level: str = "INFO", log_file: Optional[str] = None, structured: bool = False) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=numeric, handlers=handlers,
                        format="%(levelname)s %(name)s: %(message)s")
