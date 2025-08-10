"""Rate limiting utilities for model card generator."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from functools import wraps

from .exceptions import ResourceError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    window_size_seconds: int = 60
    max_queue_size: int = 100
    

class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # maximum tokens
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.monotonic()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for tokens to become available."""
        start_time = time.monotonic()
        
        while True:
            if await self.acquire(tokens):
                return True
            
            if timeout and (time.monotonic() - start_time) > timeout:
                return False
            
            # Calculate wait time for next token
            wait_time = tokens / self.rate
            await asyncio.sleep(min(wait_time, 0.1))  # Check at least every 100ms


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests
            request_times = self.requests[identifier]
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # Check if under limit
            if len(request_times) < self.max_requests:
                request_times.append(now)
                return True
            else:
                return False
    
    async def time_until_allowed(self, identifier: str) -> float:
        """Get time in seconds until next request is allowed."""
        async with self._lock:
            request_times = self.requests[identifier]
            if len(request_times) < self.max_requests:
                return 0.0
            
            # Time until oldest request falls out of window
            oldest_request = request_times[0]
            window_end = oldest_request + self.window_seconds
            return max(0.0, window_end - time.time())


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load."""
    
    def __init__(self, base_config: RateLimitConfig):
        self.base_config = base_config
        self.current_rate = base_config.requests_per_second
        self.bucket = TokenBucket(self.current_rate, base_config.burst_size)
        self.window_limiter = SlidingWindowRateLimiter(
            int(base_config.requests_per_second * base_config.window_size_seconds),
            base_config.window_size_seconds
        )
        self.success_rate = 1.0
        self.recent_errors: deque = deque(maxlen=100)
        self.last_adjustment = time.monotonic()
        
    async def acquire(self, identifier: str = "default") -> bool:
        """Acquire permission for a request."""
        # Check sliding window first
        if not await self.window_limiter.is_allowed(identifier):
            return False
        
        # Then check token bucket
        return await self.bucket.acquire()
    
    async def wait_for_permission(self, identifier: str = "default", timeout: Optional[float] = None) -> bool:
        """Wait for permission to make a request."""
        start_time = time.monotonic()
        
        while True:
            if await self.acquire(identifier):
                return True
            
            if timeout and (time.monotonic() - start_time) > timeout:
                return False
            
            # Wait for either bucket or window
            bucket_wait = 1.0 / self.current_rate
            window_wait = await self.window_limiter.time_until_allowed(identifier)
            
            wait_time = min(bucket_wait, window_wait, 1.0)
            await asyncio.sleep(wait_time)
    
    def record_success(self) -> None:
        """Record a successful request."""
        now = time.monotonic()
        self.recent_errors.append((now, False))
        self._adjust_rate_if_needed()
    
    def record_error(self, error_type: str = "general") -> None:
        """Record a failed request."""
        now = time.monotonic()
        self.recent_errors.append((now, True, error_type))
        self._adjust_rate_if_needed()
    
    def _adjust_rate_if_needed(self) -> None:
        """Adjust rate based on recent performance."""
        now = time.monotonic()
        
        # Only adjust every 10 seconds
        if now - self.last_adjustment < 10.0:
            return
        
        # Calculate error rate in last minute
        minute_ago = now - 60
        recent_events = [(t, is_error) for t, is_error in self.recent_errors if t > minute_ago]
        
        if len(recent_events) < 10:  # Not enough data
            return
        
        error_count = sum(1 for _, is_error in recent_events if is_error)
        error_rate = error_count / len(recent_events)
        
        # Adjust rate based on error rate
        if error_rate > 0.1:  # > 10% error rate
            # Decrease rate by 20%
            new_rate = self.current_rate * 0.8
            logger.warning(f"High error rate ({error_rate:.1%}), reducing rate to {new_rate:.1f}")
        elif error_rate < 0.05:  # < 5% error rate
            # Gradually increase rate (up to base rate)
            new_rate = min(self.base_config.requests_per_second, self.current_rate * 1.1)
            if new_rate > self.current_rate:
                logger.info(f"Low error rate ({error_rate:.1%}), increasing rate to {new_rate:.1f}")
        else:
            return  # No adjustment needed
        
        self.current_rate = new_rate
        self.bucket = TokenBucket(new_rate, self.base_config.burst_size)
        self.last_adjustment = now


class RateLimitedExecutor:
    """Executor that applies rate limiting to function calls."""
    
    def __init__(self, rate_limiter: AdaptiveRateLimiter, max_concurrent: int = 10):
        self.rate_limiter = rate_limiter
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.total_requests = 0
        self.total_errors = 0
    
    async def execute(self, func: Callable[..., Awaitable[Any]], *args, identifier: str = "default", timeout: Optional[float] = None, **kwargs) -> Any:
        """Execute function with rate limiting."""
        # Wait for rate limit permission
        if not await self.rate_limiter.wait_for_permission(identifier, timeout):
            raise ResourceError("rate_limit", f"Rate limit timeout after {timeout}s")
        
        async with self.semaphore:
            self.active_requests += 1
            self.total_requests += 1
            
            try:
                start_time = time.monotonic()
                result = await func(*args, **kwargs)
                duration = time.monotonic() - start_time
                
                self.rate_limiter.record_success()
                logger.debug(f"Rate-limited execution succeeded", 
                           function=func.__name__, duration_ms=duration*1000)
                return result
                
            except Exception as e:
                self.total_errors += 1
                error_type = type(e).__name__
                self.rate_limiter.record_error(error_type)
                
                logger.error(f"Rate-limited execution failed", 
                           function=func.__name__, error=str(e))
                raise
            finally:
                self.active_requests -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(1, self.total_requests),
            "current_rate_limit": self.rate_limiter.current_rate
        }


def rate_limited(config: Optional[RateLimitConfig] = None, identifier_func: Optional[Callable] = None):
    """Decorator for rate-limited async functions."""
    config = config or RateLimitConfig()
    rate_limiter = AdaptiveRateLimiter(config)
    executor = RateLimitedExecutor(rate_limiter)
    
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Determine identifier for rate limiting
            identifier = "default"
            if identifier_func:
                try:
                    identifier = identifier_func(*args, **kwargs)
                except Exception:
                    pass  # Fall back to default
            
            return await executor.execute(func, *args, identifier=identifier, **kwargs)
        
        # Attach stats method to wrapper
        wrapper.get_rate_limit_stats = executor.get_stats
        return wrapper
    
    return decorator


# Global rate limiters for common operations
api_rate_limiter = AdaptiveRateLimiter(RateLimitConfig(
    requests_per_second=5.0,
    burst_size=10,
    window_size_seconds=60
))

file_operation_rate_limiter = AdaptiveRateLimiter(RateLimitConfig(
    requests_per_second=20.0,
    burst_size=50,
    window_size_seconds=30
))

validation_rate_limiter = AdaptiveRateLimiter(RateLimitConfig(
    requests_per_second=2.0,
    burst_size=5,
    window_size_seconds=60
))