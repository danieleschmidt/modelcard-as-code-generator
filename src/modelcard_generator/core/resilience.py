"""Advanced resilience patterns for model card generation."""

import asyncio
import random
import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .exceptions import ModelCardError, ResourceError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_enabled: bool = True
    timeout_seconds: float = 30.0
    bulkhead_capacity: int = 10
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout: int = 60


class AdaptiveTimeout:
    """Adaptive timeout that adjusts based on historical performance."""

    def __init__(self, initial_timeout: float = 30.0, min_timeout: float = 5.0, max_timeout: float = 300.0):
        self.initial_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.recent_times = []
        self.current_timeout = initial_timeout

    def record_execution_time(self, execution_time: float, success: bool) -> None:
        """Record execution time for adaptive adjustment."""
        if success:
            self.recent_times.append(execution_time)
            # Keep only recent measurements
            if len(self.recent_times) > 50:
                self.recent_times = self.recent_times[-50:]

            # Adjust timeout based on percentile
            if len(self.recent_times) >= 10:
                p95_time = sorted(self.recent_times)[int(len(self.recent_times) * 0.95)]
                self.current_timeout = max(
                    self.min_timeout,
                    min(self.max_timeout, p95_time * 2.0)
                )

    def get_timeout(self) -> float:
        """Get current adaptive timeout."""
        return self.current_timeout


class Bulkhead:
    """Bulkhead pattern to isolate resource pools."""

    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.semaphore = asyncio.Semaphore(capacity)
        self.active_requests = 0
        self.total_requests = 0
        self.rejected_requests = 0

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """Acquire resource from bulkhead with optional timeout."""
        try:
            acquired = False
            if timeout:
                acquired = await asyncio.wait_for(
                    self.semaphore.acquire(),
                    timeout=timeout
                )
            else:
                await self.semaphore.acquire()
                acquired = True

            if acquired:
                self.active_requests += 1
                self.total_requests += 1
                try:
                    yield
                finally:
                    self.active_requests -= 1
                    self.semaphore.release()
            else:
                self.rejected_requests += 1
                raise ResourceError("bulkhead_full", f"Bulkhead {self.name} is at capacity")

        except asyncio.TimeoutError:
            self.rejected_requests += 1
            raise ResourceError("bulkhead_timeout", f"Timeout acquiring resource from bulkhead {self.name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": (self.total_requests - self.rejected_requests) / max(1, self.total_requests)
        }


class GracefulDegradation:
    """Graceful degradation manager for progressive feature reduction."""

    def __init__(self):
        self.degradation_levels = {
            0: "normal",
            1: "reduced_features",
            2: "essential_only",
            3: "minimal"
        }
        self.current_level = 0
        self.error_counts = defaultdict(int)
        self.last_degradation = datetime.now()

    def record_error(self, error_type: str, severity: str = "medium") -> None:
        """Record error for degradation decision making."""
        weight = {"low": 1, "medium": 2, "high": 3}.get(severity, 2)
        self.error_counts[error_type] += weight

        # Check if we should degrade further
        total_errors = sum(self.error_counts.values())
        time_since_degradation = (datetime.now() - self.last_degradation).total_seconds()

        if total_errors > 10 and time_since_degradation > 30:  # 30 seconds cooldown
            self.increase_degradation()

    def record_success(self) -> None:
        """Record successful operation for potential recovery."""
        # Reduce error counts slightly on success
        for error_type in list(self.error_counts.keys()):
            self.error_counts[error_type] = max(0, self.error_counts[error_type] - 1)
            if self.error_counts[error_type] == 0:
                del self.error_counts[error_type]

        # Consider recovering if we've been stable
        if not self.error_counts and self.current_level > 0:
            time_since_degradation = (datetime.now() - self.last_degradation).total_seconds()
            if time_since_degradation > 300:  # 5 minutes stable
                self.decrease_degradation()

    def increase_degradation(self) -> None:
        """Increase degradation level."""
        if self.current_level < len(self.degradation_levels) - 1:
            self.current_level += 1
            self.last_degradation = datetime.now()
            logger.warning(f"Degradation increased to level {self.current_level}: {self.degradation_levels[self.current_level]}")

    def decrease_degradation(self) -> None:
        """Decrease degradation level (recovery)."""
        if self.current_level > 0:
            self.current_level -= 1
            self.last_degradation = datetime.now()
            logger.info(f"Degradation decreased to level {self.current_level}: {self.degradation_levels[self.current_level]}")

    def get_level(self) -> int:
        """Get current degradation level."""
        return self.current_level

    def should_skip_feature(self, feature: str) -> bool:
        """Check if a feature should be skipped at current degradation level."""
        feature_requirements = {
            "auto_validation": 1,
            "security_scanning": 1,
            "performance_monitoring": 2,
            "advanced_metrics": 2,
            "drift_detection": 3,
            "batch_processing": 3
        }

        required_level = feature_requirements.get(feature, 0)
        return self.current_level >= required_level


class HealthMonitor:
    """System health monitoring and alerting."""

    def __init__(self):
        self.metrics = {
            "success_rate": 1.0,
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "memory_usage": 0.0,
            "active_connections": 0
        }
        self.history = []
        self.alerts = []

    def update_metrics(self, **kwargs) -> None:
        """Update health metrics."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key] = value

        # Store history
        self.history.append({
            "timestamp": datetime.now(),
            **self.metrics.copy()
        })

        # Keep only recent history
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        # Check for alerts
        self._check_alerts()

    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        current_time = datetime.now()

        # Clear old alerts
        self.alerts = [alert for alert in self.alerts
                      if (current_time - alert["timestamp"]).total_seconds() < 3600]

        # Check alert conditions
        if self.metrics["success_rate"] < 0.9:
            self._add_alert("low_success_rate", f"Success rate dropped to {self.metrics['success_rate']:.2%}")

        if self.metrics["avg_response_time"] > 10.0:
            self._add_alert("high_response_time", f"Average response time: {self.metrics['avg_response_time']:.2f}s")

        if self.metrics["error_rate"] > 0.1:
            self._add_alert("high_error_rate", f"Error rate: {self.metrics['error_rate']:.2%}")

    def _add_alert(self, alert_type: str, message: str) -> None:
        """Add alert if not recently fired."""
        recent_alerts = [alert for alert in self.alerts
                        if alert["type"] == alert_type and
                        (datetime.now() - alert["timestamp"]).total_seconds() < 600]  # 10 min cooldown

        if not recent_alerts:
            alert = {
                "type": alert_type,
                "message": message,
                "timestamp": datetime.now(),
                "severity": "warning"
            }
            self.alerts.append(alert)
            logger.warning(f"Health alert: {alert_type} - {message}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "status": "healthy" if self.metrics["success_rate"] > 0.95 else "degraded",
            "metrics": self.metrics.copy(),
            "active_alerts": len(self.alerts),
            "recent_alerts": self.alerts[-5:] if self.alerts else []
        }


# Global instances
bulkheads = {
    "api_calls": Bulkhead("api_calls", 10),
    "file_operations": Bulkhead("file_operations", 20),
    "validation": Bulkhead("validation", 5),
    "generation": Bulkhead("generation", 15)
}

degradation_manager = GracefulDegradation()
health_monitor = HealthMonitor()


def resilient_operation(
    operation_name: str,
    config: Optional[ResilienceConfig] = None,
    bulkhead_name: Optional[str] = None,
    adaptive_timeout: bool = True
):
    """Decorator for resilient operation execution."""
    if config is None:
        config = ResilienceConfig()

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            timeout_manager = AdaptiveTimeout() if adaptive_timeout else None
            start_time = time.time()
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    # Use bulkhead if specified
                    if bulkhead_name and bulkhead_name in bulkheads:
                        async with bulkheads[bulkhead_name].acquire(timeout=config.timeout_seconds):
                            result = await _execute_with_timeout(
                                func(*args, **kwargs),
                                timeout_manager.get_timeout() if timeout_manager else config.timeout_seconds
                            )
                    else:
                        result = await _execute_with_timeout(
                            func(*args, **kwargs),
                            timeout_manager.get_timeout() if timeout_manager else config.timeout_seconds
                        )

                    # Success - record metrics
                    execution_time = time.time() - start_time
                    if timeout_manager:
                        timeout_manager.record_execution_time(execution_time, True)

                    degradation_manager.record_success()
                    health_monitor.update_metrics(
                        success_rate=1.0,
                        avg_response_time=execution_time
                    )

                    return result

                except Exception as e:
                    last_exception = e
                    execution_time = time.time() - start_time

                    if timeout_manager:
                        timeout_manager.record_execution_time(execution_time, False)

                    degradation_manager.record_error(
                        type(e).__name__,
                        "high" if isinstance(e, (ResourceError, ModelCardError)) else "medium"
                    )

                    # Don't retry on certain exceptions
                    if isinstance(e, (ValueError, TypeError)) or attempt == config.max_retries:
                        break

                    # Calculate backoff delay
                    delay = config.base_delay * (config.backoff_multiplier ** attempt)
                    if config.jitter_enabled:
                        delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
                    delay = min(delay, config.max_delay)

                    logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)

            # All retries exhausted
            health_monitor.update_metrics(
                error_rate=1.0,
                avg_response_time=time.time() - start_time
            )

            raise ModelCardError(
                f"Operation {operation_name} failed after {config.max_retries + 1} attempts",
                details={"last_exception": str(last_exception)}
            )

        return wrapper
    return decorator


async def _execute_with_timeout(coro: Awaitable[Any], timeout: float) -> Any:
    """Execute coroutine with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise ResourceError("operation_timeout", f"Operation timed out after {timeout}s")


def get_resilience_status() -> Dict[str, Any]:
    """Get comprehensive resilience status."""
    return {
        "bulkheads": {name: bulkhead.get_stats() for name, bulkhead in bulkheads.items()},
        "degradation_level": degradation_manager.get_level(),
        "health_status": health_monitor.get_health_status(),
        "active_alerts": len(health_monitor.alerts)
    }
