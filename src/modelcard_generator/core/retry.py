"""Retry mechanisms with exponential backoff and jitter."""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Union, List, Awaitable, Tuple
from functools import wraps

from .exceptions import ModelCardError, ResourceError, ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1  # +/- 10%
    
    # Retry conditions
    retry_exceptions: Tuple[Type[Exception], ...] = (ResourceError, ConnectionError, TimeoutError)
    stop_exceptions: Tuple[Type[Exception], ...] = (ValidationError, ValueError, TypeError)


class RetryStatistics:
    """Statistics for retry operations."""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_retry_attempts = 0
        self.total_delay_time = 0.0
    
    def record_attempt(self, attempt_number: int, success: bool, delay_time: float = 0.0) -> None:
        """Record a retry attempt."""
        self.total_attempts += 1
        
        if success:
            self.successful_operations += 1
        else:
            if attempt_number > 1:
                self.total_retry_attempts += 1
            else:
                self.failed_operations += 1
        
        self.total_delay_time += delay_time
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "total_attempts": self.total_attempts,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "total_retry_attempts": self.total_retry_attempts,
            "total_delay_time": self.total_delay_time,
            "success_rate": self.successful_operations / max(1, self.total_attempts),
            "average_delay_per_retry": self.total_delay_time / max(1, self.total_retry_attempts)
        }


class ExponentialBackoff:
    """Exponential backoff with jitter implementation."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if attempt <= 1:
            return 0.0
        
        # Calculate exponential delay
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** (attempt - 2)),
            self.config.max_delay
        )
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.0, delay)


class RetryManager:
    """Manages retry operations with comprehensive error handling."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.backoff = ExponentialBackoff(self.config)
        self.stats = RetryStatistics()
    
    async def execute_with_retry(self, 
                               func: Callable[..., Awaitable[Any]], 
                               *args, 
                               operation_name: Optional[str] = None,
                               **kwargs) -> Any:
        """Execute function with retry logic."""
        operation_name = operation_name or func.__name__
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                start_time = time.time()
                
                logger.debug(f"Attempting {operation_name} (attempt {attempt}/{self.config.max_attempts})")
                
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                self.stats.record_attempt(attempt, success=True)
                
                if attempt > 1:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt} after {duration:.2f}s")
                else:
                    logger.debug(f"Operation {operation_name} succeeded on first attempt")
                
                return result
                
            except Exception as e:
                last_exception = e
                duration = time.time() - start_time
                
                # Check if we should stop retrying
                if self._should_stop_retry(e, attempt):
                    logger.error(f"Operation {operation_name} failed permanently", 
                               attempt=attempt, error=str(e))
                    self.stats.record_attempt(attempt, success=False)
                    raise e
                
                # Calculate delay for next attempt
                delay = self.backoff.calculate_delay(attempt + 1)
                
                logger.warning(f"Operation {operation_name} failed on attempt {attempt}", 
                             error=str(e), retry_delay=delay, next_attempt=attempt + 1)
                
                # Wait before retry (except on last attempt)
                if attempt < self.config.max_attempts and delay > 0:
                    await asyncio.sleep(delay)
                    self.stats.record_attempt(attempt, success=False, delay_time=delay)
                else:
                    self.stats.record_attempt(attempt, success=False)
        
        # All attempts exhausted
        logger.error(f"Operation {operation_name} failed after {self.config.max_attempts} attempts")
        raise last_exception
    
    def _should_stop_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should stop retrying based on exception type."""
        # Stop on explicitly non-retryable exceptions
        if isinstance(exception, self.config.stop_exceptions):
            return True
        
        # Continue on explicitly retryable exceptions
        if isinstance(exception, self.config.retry_exceptions):
            return False
        
        # For other exceptions, be conservative and don't retry
        return True
    
    def get_statistics(self) -> dict:
        """Get retry statistics."""
        return self.stats.get_stats()


def create_retry_decorator(config: Optional[RetryConfig] = None, 
                          operation_name: Optional[str] = None):
    """Create a retry decorator with specified configuration."""
    retry_manager = RetryManager(config)
    
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            name = operation_name or func.__name__
            return await retry_manager.execute_with_retry(func, *args, operation_name=name, **kwargs)
        
        # Attach utility methods
        wrapper.get_retry_stats = retry_manager.get_statistics
        wrapper.retry_manager = retry_manager
        
        return wrapper
    
    return decorator


# Common retry configurations
API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    retry_exceptions=(ResourceError, ConnectionError, TimeoutError, OSError)
)

FILE_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=0.5,
    max_delay=5.0,
    retry_exceptions=(OSError, IOError, PermissionError)
)

VALIDATION_RETRY_CONFIG = RetryConfig(
    max_attempts=1,  # Usually don't retry validation
    base_delay=0.0,
    max_delay=0.0,
    retry_exceptions=()
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    exponential_base=2.0,
    retry_exceptions=(ConnectionError, TimeoutError)
)


# Convenience decorators
api_retry = create_retry_decorator(API_RETRY_CONFIG)
file_retry = create_retry_decorator(FILE_RETRY_CONFIG)
validation_retry = create_retry_decorator(VALIDATION_RETRY_CONFIG)
database_retry = create_retry_decorator(DATABASE_RETRY_CONFIG)


class BatchRetryManager:
    """Manages retry for batch operations with partial failure handling."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.retry_manager = RetryManager(config)
    
    async def execute_batch_with_retry(self, 
                                     operations: List[Callable[..., Awaitable[Any]]], 
                                     max_concurrent: int = 5,
                                     fail_fast: bool = False) -> List[Union[Any, Exception]]:
        """Execute batch operations with retry and partial failure handling."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single(op_index: int, operation: Callable) -> Tuple[int, Union[Any, Exception]]:
            async with semaphore:
                try:
                    result = await self.retry_manager.execute_with_retry(
                        operation, 
                        operation_name=f"batch_op_{op_index}"
                    )
                    return op_index, result
                except Exception as e:
                    logger.error(f"Batch operation {op_index} failed permanently", error=str(e))
                    if fail_fast:
                        raise
                    return op_index, e
        
        # Execute all operations concurrently
        tasks = [execute_single(i, op) for i, op in enumerate(operations)]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=not fail_fast)
        except Exception as e:
            if fail_fast:
                logger.error("Batch operation failed fast due to error", error=str(e))
                raise
        
        # Sort results by original index
        sorted_results = [None] * len(operations)
        for index, result in results:
            sorted_results[index] = result
        
        # Log batch summary
        successful = sum(1 for r in sorted_results if not isinstance(r, Exception))
        failed = len(operations) - successful
        
        logger.info(f"Batch operation completed", 
                   total_operations=len(operations), 
                   successful=successful, 
                   failed=failed)
        
        return sorted_results


# Global batch retry manager
batch_retry_manager = BatchRetryManager()