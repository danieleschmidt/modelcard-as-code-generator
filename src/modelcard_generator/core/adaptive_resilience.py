"""Advanced adaptive resilience patterns with self-healing capabilities."""

import asyncio
import json
import math
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar

import numpy as np
from scipy import stats

from .logging_config import get_logger
from .exceptions import ModelCardError, ValidationError

logger = get_logger(__name__)

T = TypeVar('T')


class ResiliencePattern(Enum):
    """Types of resilience patterns."""
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    RETRY = "retry"
    TIMEOUT = "timeout"
    RATE_LIMITER = "rate_limiter"
    CACHE_ASIDE = "cache_aside"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    HEALTH_CHECK = "health_check"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ResilienceMetrics:
    """Metrics for resilience monitoring."""
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    circuit_breaker_trips: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    availability: float = 1.0
    recovery_time: float = 0.0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    healthy: bool
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns."""
    enabled_patterns: Set[ResiliencePattern] = field(default_factory=set)
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    timeout_seconds: float = 30.0
    rate_limit_requests_per_second: float = 100.0
    health_check_interval: float = 30.0
    graceful_degradation_enabled: bool = True
    bulkhead_pool_size: int = 10
    adaptive_thresholds: bool = True


class AdaptiveCircuitBreaker:
    """Self-tuning circuit breaker with machine learning capabilities."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        adaptive_learning: bool = True
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.adaptive_learning = adaptive_learning
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        # Adaptive learning parameters
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptation_window = 100  # Number of requests to consider
        
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
        except Exception as e:
            self._record_failure(time.time() - start_time)
            raise e
    
    def _record_success(self, duration: float) -> None:
        """Record successful operation."""
        self.success_count += 1
        self.failure_count = 0
        
        self.performance_history.append({
            "success": True,
            "duration": duration,
            "timestamp": datetime.now()
        })
        
        if self.state == "HALF_OPEN" and self.success_count >= self.success_threshold:
            self.state = "CLOSED"
            logger.info("Circuit breaker CLOSED - service recovered")
        
        self._adapt_parameters()
    
    def _record_failure(self, duration: float) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = datetime.now()
        
        self.performance_history.append({
            "success": False,
            "duration": duration,
            "timestamp": datetime.now()
        })
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures")
        
        self._adapt_parameters()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return False
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _adapt_parameters(self) -> None:
        """Adapt circuit breaker parameters based on performance history."""
        if not self.adaptive_learning or len(self.performance_history) < self.adaptation_window:
            return
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(minutes=30)
        self.performance_history = [
            h for h in self.performance_history[-self.adaptation_window:]
            if h["timestamp"] > cutoff_time
        ]
        
        if len(self.performance_history) < 10:
            return
        
        # Calculate recent failure rate
        recent_failures = sum(1 for h in self.performance_history[-20:] if not h["success"])
        failure_rate = recent_failures / min(20, len(self.performance_history))
        
        # Adapt thresholds based on failure patterns
        if failure_rate > 0.5:  # High failure rate
            self.failure_threshold = max(3, self.failure_threshold - 1)
            self.recovery_timeout = min(300, self.recovery_timeout * 1.5)
        elif failure_rate < 0.1:  # Low failure rate
            self.failure_threshold = min(10, self.failure_threshold + 1)
            self.recovery_timeout = max(30, self.recovery_timeout * 0.8)
        
        logger.debug(f"Adapted circuit breaker: threshold={self.failure_threshold}, timeout={self.recovery_timeout}")


class BulkheadPattern:
    """Resource isolation pattern to prevent cascade failures."""
    
    def __init__(self, pool_size: int = 10, timeout: float = 30.0):
        self.pool_size = pool_size
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(pool_size)
        self.active_requests = 0
        self.queue_size = 0
        
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function within bulkhead constraints."""
        self.queue_size += 1
        try:
            async with asyncio.wait_for(self.semaphore.acquire(), timeout=self.timeout):
                self.active_requests += 1
                self.queue_size -= 1
                try:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                    return result
                finally:
                    self.active_requests -= 1
                    self.semaphore.release()
        except asyncio.TimeoutError:
            self.queue_size -= 1
            raise BulkheadTimeoutError(f"Bulkhead timeout after {self.timeout}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current bulkhead metrics."""
        return {
            "active_requests": self.active_requests,
            "queue_size": self.queue_size,
            "pool_utilization": self.active_requests / self.pool_size,
            "available_capacity": self.pool_size - self.active_requests
        }


class AdaptiveRetry:
    """Intelligent retry mechanism with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    async def execute(
        self,
        func: Callable[..., T],
        *args,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> T:
        """Execute function with adaptive retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Operation succeeded on attempt {attempt + 1}")
                return result
            
            except retryable_exceptions as e:
                last_exception = e
                if attempt == self.max_attempts - 1:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        logger.error(f"All {self.max_attempts} attempts failed")
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        import random
        
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter ±25%
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)


class GracefulDegradation:
    """Graceful degradation handler for maintaining service availability."""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
        self.feature_flags: Dict[str, bool] = {}
        self.degradation_levels: Dict[str, int] = {}
        
    def register_fallback(self, service_name: str, fallback_func: Callable) -> None:
        """Register a fallback handler for a service."""
        self.fallback_handlers[service_name] = fallback_func
        logger.info(f"Registered fallback for service: {service_name}")
    
    def set_feature_flag(self, feature_name: str, enabled: bool) -> None:
        """Set feature flag for dynamic feature toggling."""
        self.feature_flags[feature_name] = enabled
        logger.info(f"Feature flag {feature_name}: {'enabled' if enabled else 'disabled'}")
    
    def set_degradation_level(self, service_name: str, level: int) -> None:
        """Set degradation level (0=full, 1=limited, 2=minimal, 3=disabled)."""
        self.degradation_levels[service_name] = level
        logger.info(f"Service {service_name} degradation level: {level}")
    
    async def execute_with_fallback(
        self,
        service_name: str,
        primary_func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute primary function with fallback on failure."""
        # Check feature flags
        if not self.feature_flags.get(service_name, True):
            return await self._execute_fallback(service_name, *args, **kwargs)
        
        # Check degradation level
        degradation_level = self.degradation_levels.get(service_name, 0)
        if degradation_level >= 3:  # Service disabled
            return await self._execute_fallback(service_name, *args, **kwargs)
        
        try:
            # Execute primary function with potential limitations
            if degradation_level >= 2:  # Minimal functionality
                kwargs["simplified_mode"] = True
            elif degradation_level >= 1:  # Limited functionality
                kwargs["reduced_features"] = True
            
            result = await primary_func(*args, **kwargs) if asyncio.iscoroutinefunction(primary_func) else primary_func(*args, **kwargs)
            return result
        
        except Exception as e:
            logger.warning(f"Primary service {service_name} failed: {e}. Using fallback.")
            return await self._execute_fallback(service_name, *args, **kwargs)
    
    async def _execute_fallback(self, service_name: str, *args, **kwargs) -> Any:
        """Execute fallback handler."""
        fallback = self.fallback_handlers.get(service_name)
        if not fallback:
            raise ServiceUnavailableError(f"No fallback available for service: {service_name}")
        
        try:
            result = await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Fallback for {service_name} also failed: {e}")
            raise ServiceUnavailableError(f"Both primary and fallback failed for {service_name}")


class ResilienceOrchestrator:
    """Orchestrates all resilience patterns for comprehensive protection."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.bulkheads: Dict[str, BulkheadPattern] = {}
        self.retry_policies: Dict[str, AdaptiveRetry] = {}
        self.degradation = GracefulDegradation()
        
    def get_or_create_circuit_breaker(self, service_name: str, **kwargs) -> AdaptiveCircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = AdaptiveCircuitBreaker(**kwargs)
        return self.circuit_breakers[service_name]
    
    def get_or_create_bulkhead(self, service_name: str, **kwargs) -> BulkheadPattern:
        """Get or create bulkhead for service."""
        if service_name not in self.bulkheads:
            self.bulkheads[service_name] = BulkheadPattern(**kwargs)
        return self.bulkheads[service_name]
    
    def get_or_create_retry_policy(self, service_name: str, **kwargs) -> AdaptiveRetry:
        """Get or create retry policy for service."""
        if service_name not in self.retry_policies:
            self.retry_policies[service_name] = AdaptiveRetry(**kwargs)
        return self.retry_policies[service_name]
    
    async def execute_with_full_protection(
        self,
        service_name: str,
        func: Callable[..., T],
        *args,
        circuit_breaker_config: Optional[Dict] = None,
        bulkhead_config: Optional[Dict] = None,
        retry_config: Optional[Dict] = None,
        fallback_func: Optional[Callable] = None,
        **kwargs
    ) -> T:
        """Execute function with full resilience protection."""
        
        # Setup resilience components
        circuit_breaker = self.get_or_create_circuit_breaker(
            service_name, **(circuit_breaker_config or {})
        )
        bulkhead = self.get_or_create_bulkhead(
            service_name, **(bulkhead_config or {})
        )
        retry_policy = self.get_or_create_retry_policy(
            service_name, **(retry_config or {})
        )
        
        if fallback_func:
            self.degradation.register_fallback(service_name, fallback_func)
        
        # Wrapper function that applies all resilience patterns
        async def protected_execution():
            async def circuit_breaker_wrapper():
                async def bulkhead_wrapper():
                    return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                return await bulkhead.execute(bulkhead_wrapper)
            
            return await circuit_breaker.call(circuit_breaker_wrapper)
        
        # Execute with retry and graceful degradation
        try:
            return await retry_policy.execute(protected_execution)
        except Exception as e:
            logger.warning(f"All resilience patterns exhausted for {service_name}: {e}")
            if fallback_func:
                return await self.degradation.execute_with_fallback(
                    service_name, func, *args, **kwargs
                )
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all resilience components."""
        status = {
            "circuit_breakers": {},
            "bulkheads": {},
            "retry_policies": {},
            "overall_health": "healthy"
        }
        
        # Circuit breaker status
        open_breakers = 0
        for name, cb in self.circuit_breakers.items():
            cb_status = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            status["circuit_breakers"][name] = cb_status
            if cb.state == "OPEN":
                open_breakers += 1
        
        # Bulkhead status
        high_utilization = 0
        for name, bh in self.bulkheads.items():
            metrics = bh.get_metrics()
            status["bulkheads"][name] = metrics
            if metrics["pool_utilization"] > 0.8:
                high_utilization += 1
        
        # Overall health assessment
        if open_breakers > 0 or high_utilization > len(self.bulkheads) * 0.5:
            status["overall_health"] = "degraded"
        
        return status


# Exception classes for resilience patterns
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class BulkheadTimeoutError(Exception):
    """Raised when bulkhead timeout is exceeded."""
    pass


class ServiceUnavailableError(Exception):
    """Raised when service is unavailable."""
    pass


# Global resilience orchestrator instance
resilience_orchestrator = ResilienceOrchestrator()