"""Circuit breaker pattern implementation for fault tolerance."""

import asyncio
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from .exceptions import ResourceError
from .logging_config import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures to open circuit
    success_threshold: int = 3          # Number of successes to close circuit
    timeout_seconds: float = 60.0       # Time to wait before trying half-open
    monitor_window_seconds: int = 60    # Window for counting failures
    slow_call_threshold_ms: float = 5000.0  # Calls slower than this are considered failures
    max_half_open_calls: int = 3        # Max concurrent calls in half-open state


class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.slow_calls = 0
        self.state_changes = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.recent_calls: List[tuple] = []  # (timestamp, success, duration)

    def record_call(self, success: bool, duration_ms: float) -> None:
        """Record a call result."""
        now = time.time()
        self.total_calls += 1

        if success:
            self.successful_calls += 1
            self.last_success_time = now
        else:
            self.failed_calls += 1
            self.last_failure_time = now

        if duration_ms > 5000:  # Consider slow calls
            self.slow_calls += 1

        # Add to recent calls window
        self.recent_calls.append((now, success, duration_ms))

        # Clean old entries
        cutoff = now - self.window_size
        self.recent_calls = [call for call in self.recent_calls if call[0] > cutoff]

    def get_failure_rate(self) -> float:
        """Get current failure rate in the monitoring window."""
        if not self.recent_calls:
            return 0.0

        failures = sum(1 for _, success, _ in self.recent_calls if not success)
        return failures / len(self.recent_calls)

    def get_slow_call_rate(self) -> float:
        """Get current slow call rate."""
        if not self.recent_calls:
            return 0.0

        slow_calls = sum(1 for _, _, duration in self.recent_calls if duration > 5000)
        return slow_calls / len(self.recent_calls)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "slow_calls": self.slow_calls,
            "failure_rate": self.get_failure_rate(),
            "slow_call_rate": self.get_slow_call_rate(),
            "state_changes": self.state_changes,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "recent_calls_count": len(self.recent_calls)
        }


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats(self.config.monitor_window_seconds)
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if we should transition states
            await self._check_state_transition()

            # Fast fail if circuit is open
            if self.state == CircuitState.OPEN:
                raise ResourceError(
                    "circuit_breaker",
                    f"Circuit breaker '{self.name}' is OPEN - failing fast"
                )

            # Limit concurrent calls in half-open state
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.max_half_open_calls:
                    raise ResourceError(
                        "circuit_breaker",
                        f"Circuit breaker '{self.name}' is HALF_OPEN - too many concurrent calls"
                    )
                self.half_open_calls += 1

        # Execute the function
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Check if call was too slow
            is_slow = duration_ms > self.config.slow_call_threshold_ms
            success = not is_slow

            await self._record_success(duration_ms, is_slow)

            if is_slow:
                logger.warning(f"Slow call in circuit breaker '{self.name}': {duration_ms:.1f}ms")

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._record_failure(e, duration_ms)
            raise
        finally:
            if self.state == CircuitState.HALF_OPEN:
                async with self._lock:
                    self.half_open_calls -= 1

    async def _record_success(self, duration_ms: float, is_slow: bool) -> None:
        """Record a successful call."""
        async with self._lock:
            self.stats.record_call(success=not is_slow, duration_ms=duration_ms)

            if not is_slow:
                self.consecutive_failures = 0
                self.consecutive_successes += 1

                # Transition to closed if we have enough successes in half-open
                if (self.state == CircuitState.HALF_OPEN and
                    self.consecutive_successes >= self.config.success_threshold):
                    await self._transition_to_closed()
            else:
                # Slow calls count as failures for state management
                await self._handle_failure()

    async def _record_failure(self, error: Exception, duration_ms: float) -> None:
        """Record a failed call."""
        async with self._lock:
            self.stats.record_call(success=False, duration_ms=duration_ms)
            await self._handle_failure()

            logger.error(f"Circuit breaker '{self.name}' recorded failure",
                        error=str(error), duration_ms=duration_ms)

    async def _handle_failure(self) -> None:
        """Handle a failure (internal method, assumes lock is held)."""
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        # Transition to open if we exceed failure threshold
        if (self.state == CircuitState.CLOSED and
            self.consecutive_failures >= self.config.failure_threshold):
            await self._transition_to_open()
        elif (self.state == CircuitState.HALF_OPEN and
              self.consecutive_failures >= 1):
            # Immediately open if we fail in half-open state
            await self._transition_to_open()

    async def _check_state_transition(self) -> None:
        """Check if we should transition from open to half-open."""
        if (self.state == CircuitState.OPEN and
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.timeout_seconds):
            await self._transition_to_half_open()

    async def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.stats.state_changes += 1

        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to CLOSED")

    async def _transition_to_open(self) -> None:
        """Transition to open state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        self.stats.state_changes += 1

        logger.warning(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to OPEN")

    async def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.half_open_calls = 0
        self.stats.state_changes += 1

        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to HALF_OPEN")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time,
            "stats": self.stats.to_dict(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "slow_call_threshold_ms": self.config.slow_call_threshold_ms
            }
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config)
            return self.breakers[name]

    async def call_with_breaker(self, name: str, func: Callable[..., Awaitable[Any]],
                               *args, config: Optional[CircuitBreakerConfig] = None, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        breaker = await self.get_breaker(name, config)
        return await breaker.call(func, *args, **kwargs)

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }

    def reset_breaker(self, name: str) -> bool:
        """Reset a circuit breaker to closed state."""
        if name in self.breakers:
            breaker = self.breakers[name]
            breaker.state = CircuitState.CLOSED
            breaker.consecutive_failures = 0
            breaker.consecutive_successes = 0
            breaker.last_failure_time = None
            logger.info(f"Circuit breaker '{name}' manually reset to CLOSED")
            return True
        return False


# Global circuit breaker registry
registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await registry.call_with_breaker(name, func, *args, config=config, **kwargs)

        # Attach utility methods
        wrapper.get_breaker_status = lambda: registry.breakers.get(name, {}).get_status() if name in registry.breakers else None
        wrapper.reset_breaker = lambda: registry.reset_breaker(name)

        return wrapper
    return decorator


# Common circuit breaker configurations
API_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    success_threshold=2,
    timeout_seconds=30.0,
    slow_call_threshold_ms=5000.0
)

FILE_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=3,
    timeout_seconds=10.0,
    slow_call_threshold_ms=2000.0
)

VALIDATION_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=2,
    success_threshold=1,
    timeout_seconds=60.0,
    slow_call_threshold_ms=10000.0
)
