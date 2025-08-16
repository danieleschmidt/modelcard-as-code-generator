"""Advanced performance optimization and scaling capabilities."""

import asyncio
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .advanced_monitoring import metrics_collector, monitor_performance
from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class IntelligentResourcePool:
    """Dynamic resource pool with intelligent scaling."""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = None,
        scale_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        worker_timeout: float = 300.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.scale_threshold = scale_threshold
        self.scale_down_threshold = scale_down_threshold
        self.worker_timeout = worker_timeout
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.min_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.min_workers)
        
        self.active_threads = 0
        self.active_processes = 0
        self.queue_size = 0
        self.last_scale_time = time.time()
        
        # Performance tracking
        self.task_completion_times: List[float] = []
        self.peak_usage = {"threads": 0, "processes": 0}
        
    async def execute_io_bound(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute I/O bound task with intelligent threading."""
        self.queue_size += 1
        self.active_threads += 1
        
        try:
            # Check if we need to scale up
            await self._maybe_scale_threads()
            
            start_time = time.time()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
            
            completion_time = time.time() - start_time
            self.task_completion_times.append(completion_time)
            
            # Keep only recent completion times
            if len(self.task_completion_times) > 100:
                self.task_completion_times = self.task_completion_times[-100:]
            
            metrics_collector.record_histogram("thread_task_duration", completion_time * 1000)
            return result
            
        finally:
            self.active_threads -= 1
            self.queue_size -= 1
    
    async def execute_cpu_bound(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute CPU bound task with intelligent process pool."""
        self.queue_size += 1
        self.active_processes += 1
        
        try:
            # Check if we need to scale up
            await self._maybe_scale_processes()
            
            start_time = time.time()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
            
            completion_time = time.time() - start_time
            self.task_completion_times.append(completion_time)
            
            metrics_collector.record_histogram("process_task_duration", completion_time * 1000)
            return result
            
        finally:
            self.active_processes -= 1
            self.queue_size -= 1
    
    async def execute_batch(
        self,
        tasks: List[tuple],
        executor_type: str = "thread",
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Execute batch of tasks with optimal concurrency."""
        if not tasks:
            return []
        
        max_concurrent = max_concurrent or min(len(tasks), self.max_workers)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task_func, *task_args, **task_kwargs):
            async with semaphore:
                if executor_type == "thread":
                    return await self.execute_io_bound(task_func, *task_args, **task_kwargs)
                else:
                    return await self.execute_cpu_bound(task_func, *task_args, **task_kwargs)
        
        # Execute all tasks concurrently
        start_time = time.time()
        task_coroutines = []
        
        for task in tasks:
            if isinstance(task, tuple) and len(task) >= 1:
                func = task[0]
                args = task[1:] if len(task) > 1 else ()
                task_coroutines.append(execute_with_semaphore(func, *args))
        
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        batch_duration = time.time() - start_time
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        metrics_collector.record_histogram("batch_duration", batch_duration * 1000)
        metrics_collector.increment_counter("batch_tasks_successful", len(successful_results))
        metrics_collector.increment_counter("batch_tasks_failed", len(failed_results))
        
        logger.info(f"Batch execution completed: {len(successful_results)} successful, {len(failed_results)} failed in {batch_duration:.2f}s")
        
        return results
    
    async def _maybe_scale_threads(self) -> None:
        """Intelligently scale thread pool based on load."""
        current_utilization = self.active_threads / self.thread_pool._max_workers
        
        if current_utilization > self.scale_threshold and self.thread_pool._max_workers < self.max_workers:
            new_max = min(self.thread_pool._max_workers * 2, self.max_workers)
            await self._scale_thread_pool(new_max)
            logger.info(f"Scaled thread pool up to {new_max} workers")
        
        elif current_utilization < self.scale_down_threshold and self.thread_pool._max_workers > self.min_workers:
            new_max = max(self.thread_pool._max_workers // 2, self.min_workers)
            await self._scale_thread_pool(new_max)
            logger.info(f"Scaled thread pool down to {new_max} workers")
    
    async def _maybe_scale_processes(self) -> None:
        """Intelligently scale process pool based on load."""
        current_utilization = self.active_processes / self.process_pool._max_workers
        
        if current_utilization > self.scale_threshold and self.process_pool._max_workers < self.max_workers:
            new_max = min(self.process_pool._max_workers * 2, self.max_workers)
            await self._scale_process_pool(new_max)
            logger.info(f"Scaled process pool up to {new_max} workers")
    
    async def _scale_thread_pool(self, new_max_workers: int) -> None:
        """Scale thread pool to new size."""
        old_pool = self.thread_pool
        self.thread_pool = ThreadPoolExecutor(max_workers=new_max_workers)
        # Note: In production, we'd gracefully shutdown the old pool
        
    async def _scale_process_pool(self, new_max_workers: int) -> None:
        """Scale process pool to new size."""
        old_pool = self.process_pool
        self.process_pool = ProcessPoolExecutor(max_workers=new_max_workers)
        # Note: In production, we'd gracefully shutdown the old pool
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource pool metrics."""
        avg_completion_time = sum(self.task_completion_times) / len(self.task_completion_times) if self.task_completion_times else 0
        
        return {
            "thread_pool": {
                "max_workers": self.thread_pool._max_workers,
                "active_workers": self.active_threads,
                "utilization": self.active_threads / self.thread_pool._max_workers
            },
            "process_pool": {
                "max_workers": self.process_pool._max_workers,
                "active_workers": self.active_processes,
                "utilization": self.active_processes / self.process_pool._max_workers
            },
            "queue_size": self.queue_size,
            "avg_completion_time_ms": avg_completion_time * 1000,
            "peak_usage": self.peak_usage
        }


class AdaptiveCache:
    """Multi-layer adaptive cache with intelligent eviction."""
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l3_size: int = 10000,
        default_ttl: int = 3600
    ):
        self.l1_cache = {}  # In-memory, fastest
        self.l2_cache = {}  # In-memory, larger
        self.l3_cache = {}  # Persistent, largest
        
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        self.default_ttl = default_ttl
        
        # Cache statistics
        self.stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0,
            "evictions": 0, "expired": 0
        }
        
        # Access patterns for intelligent caching
        self.access_patterns = {}
        self.prediction_model = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent layer promotion."""
        current_time = time.time()
        
        # L1 Cache check
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if current_time < entry["expires_at"]:
                self.stats["l1_hits"] += 1
                self._update_access_pattern(key)
                return entry["value"]
            else:
                del self.l1_cache[key]
                self.stats["expired"] += 1
        
        # L2 Cache check
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if current_time < entry["expires_at"]:
                self.stats["l2_hits"] += 1
                # Promote to L1 if frequently accessed
                if self._should_promote_to_l1(key):
                    self._set_l1(key, entry["value"], entry["expires_at"] - current_time)
                self._update_access_pattern(key)
                return entry["value"]
            else:
                del self.l2_cache[key]
                self.stats["expired"] += 1
        
        # L3 Cache check
        if key in self.l3_cache:
            entry = self.l3_cache[key]
            if current_time < entry["expires_at"]:
                self.stats["l3_hits"] += 1
                # Promote to L2 if frequently accessed
                if self._should_promote_to_l2(key):
                    self._set_l2(key, entry["value"], entry["expires_at"] - current_time)
                self._update_access_pattern(key)
                return entry["value"]
            else:
                del self.l3_cache[key]
                self.stats["expired"] += 1
        
        # Cache miss
        self.stats["l1_misses"] += 1
        self.stats["l2_misses"] += 1
        self.stats["l3_misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with intelligent layer placement."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        # Determine optimal cache layer based on value characteristics
        value_size = self._estimate_size(value)
        access_frequency = self.access_patterns.get(key, {}).get("frequency", 0)
        
        if access_frequency > 10 or value_size < 1024:  # Frequently accessed or small
            self._set_l1(key, value, ttl)
        elif access_frequency > 3 or value_size < 10240:  # Moderately accessed or medium
            self._set_l2(key, value, ttl)
        else:  # Large or infrequently accessed
            self._set_l3(key, value, ttl)
        
        self._update_access_pattern(key)
    
    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries matching pattern."""
        if pattern is None:
            # Clear all caches
            count = len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l3_cache.clear()
            return count
        
        # Pattern-based invalidation
        count = 0
        for cache in [self.l1_cache, self.l2_cache, self.l3_cache]:
            keys_to_delete = [k for k in cache.keys() if pattern in k]
            for key in keys_to_delete:
                del cache[key]
                count += 1
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = sum(self.stats.values())
        if total_requests == 0:
            return self.stats
        
        l1_hit_rate = self.stats["l1_hits"] / (self.stats["l1_hits"] + self.stats["l1_misses"])
        l2_hit_rate = self.stats["l2_hits"] / (self.stats["l2_hits"] + self.stats["l2_misses"])
        l3_hit_rate = self.stats["l3_hits"] / (self.stats["l3_hits"] + self.stats["l3_misses"])
        
        overall_hit_rate = (self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]) / total_requests
        
        return {
            **self.stats,
            "hit_rates": {
                "l1": l1_hit_rate,
                "l2": l2_hit_rate,
                "l3": l3_hit_rate,
                "overall": overall_hit_rate
            },
            "cache_sizes": {
                "l1": len(self.l1_cache),
                "l2": len(self.l2_cache),
                "l3": len(self.l3_cache)
            },
            "cache_utilization": {
                "l1": len(self.l1_cache) / self.l1_size,
                "l2": len(self.l2_cache) / self.l2_size,
                "l3": len(self.l3_cache) / self.l3_size
            }
        }
    
    def _set_l1(self, key: str, value: Any, ttl: int) -> None:
        """Set value in L1 cache with LRU eviction."""
        if len(self.l1_cache) >= self.l1_size:
            self._evict_lru(self.l1_cache)
        
        self.l1_cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "accessed_at": time.time()
        }
    
    def _set_l2(self, key: str, value: Any, ttl: int) -> None:
        """Set value in L2 cache with LRU eviction."""
        if len(self.l2_cache) >= self.l2_size:
            self._evict_lru(self.l2_cache)
        
        self.l2_cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "accessed_at": time.time()
        }
    
    def _set_l3(self, key: str, value: Any, ttl: int) -> None:
        """Set value in L3 cache with LRU eviction."""
        if len(self.l3_cache) >= self.l3_size:
            self._evict_lru(self.l3_cache)
        
        self.l3_cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "accessed_at": time.time()
        }
    
    def _evict_lru(self, cache: Dict[str, Any]) -> None:
        """Evict least recently used item from cache."""
        if not cache:
            return
        
        oldest_key = min(cache.keys(), key=lambda k: cache[k]["accessed_at"])
        del cache[oldest_key]
        self.stats["evictions"] += 1
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if key should be promoted to L1 cache."""
        pattern = self.access_patterns.get(key, {})
        return pattern.get("frequency", 0) > 5
    
    def _should_promote_to_l2(self, key: str) -> bool:
        """Determine if key should be promoted to L2 cache."""
        pattern = self.access_patterns.get(key, {})
        return pattern.get("frequency", 0) > 2
    
    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern for intelligent caching decisions."""
        if key not in self.access_patterns:
            self.access_patterns[key] = {"frequency": 0, "last_access": time.time()}
        
        self.access_patterns[key]["frequency"] += 1
        self.access_patterns[key]["last_access"] = time.time()
        
        # Clean old patterns periodically
        if len(self.access_patterns) > 10000:
            self._cleanup_access_patterns()
    
    def _cleanup_access_patterns(self) -> None:
        """Clean up old access patterns to prevent memory bloat."""
        cutoff_time = time.time() - 86400  # 24 hours
        keys_to_remove = [
            k for k, v in self.access_patterns.items()
            if v["last_access"] < cutoff_time
        ]
        for key in keys_to_remove:
            del self.access_patterns[key]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        import sys
        try:
            return sys.getsizeof(value)
        except:
            return 1024  # Default estimate


class LoadBalancer:
    """Intelligent load balancing for distributed processing."""
    
    def __init__(self, workers: List[str]):
        self.workers = workers
        self.worker_stats = {worker: {"requests": 0, "errors": 0, "response_time": 0.0, "healthy": True} for worker in workers}
        self.current_index = 0
        
    def get_next_worker(self, strategy: str = "least_connections") -> str:
        """Get next worker based on load balancing strategy."""
        healthy_workers = [w for w in self.workers if self.worker_stats[w]["healthy"]]
        
        if not healthy_workers:
            raise Exception("No healthy workers available")
        
        if strategy == "round_robin":
            worker = healthy_workers[self.current_index % len(healthy_workers)]
            self.current_index += 1
            return worker
        
        elif strategy == "least_connections":
            return min(healthy_workers, key=lambda w: self.worker_stats[w]["requests"])
        
        elif strategy == "least_response_time":
            return min(healthy_workers, key=lambda w: self.worker_stats[w]["response_time"])
        
        elif strategy == "weighted_round_robin":
            # Weight based on inverse of error rate and response time
            weights = {}
            for worker in healthy_workers:
                stats = self.worker_stats[worker]
                error_rate = stats["errors"] / max(stats["requests"], 1)
                weight = 1 / (1 + error_rate + stats["response_time"] / 1000)  # Convert ms to s
                weights[worker] = weight
            
            # Weighted selection
            total_weight = sum(weights.values())
            if total_weight == 0:
                return healthy_workers[0]
            
            import random
            rand_val = random.uniform(0, total_weight)
            cumulative = 0
            for worker, weight in weights.items():
                cumulative += weight
                if rand_val <= cumulative:
                    return worker
            
            return healthy_workers[-1]  # Fallback
        
        else:
            return healthy_workers[0]  # Default to first healthy worker
    
    def record_request(self, worker: str, response_time_ms: float, success: bool) -> None:
        """Record request statistics for load balancing decisions."""
        stats = self.worker_stats[worker]
        stats["requests"] += 1
        if not success:
            stats["errors"] += 1
        
        # Exponential moving average for response time
        alpha = 0.1
        stats["response_time"] = alpha * response_time_ms + (1 - alpha) * stats["response_time"]
        
        # Health check based on error rate and response time
        error_rate = stats["errors"] / stats["requests"]
        stats["healthy"] = error_rate < 0.1 and stats["response_time"] < 5000  # 5 second threshold


# Global instances
resource_pool = IntelligentResourcePool()
adaptive_cache = AdaptiveCache()


# Performance optimization decorators
def optimize_for_io(func: Callable) -> Callable:
    """Decorator to optimize I/O bound functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await resource_pool.execute_io_bound(func, *args, **kwargs)
    return wrapper


def optimize_for_cpu(func: Callable) -> Callable:
    """Decorator to optimize CPU bound functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await resource_pool.execute_cpu_bound(func, *args, **kwargs)
    return wrapper


def cache_result(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = adaptive_cache.get(cache_key)
            if cached_result is not None:
                metrics_collector.increment_counter("cache_hits")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            adaptive_cache.set(cache_key, result, ttl)
            metrics_collector.increment_counter("cache_misses")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = adaptive_cache.get(cache_key)
            if cached_result is not None:
                metrics_collector.increment_counter("cache_hits")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            adaptive_cache.set(cache_key, result, ttl)
            metrics_collector.increment_counter("cache_misses")
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@monitor_performance("batch_optimization")
async def optimize_batch_processing(
    tasks: List[tuple],
    batch_size: int = 50,
    max_concurrent: int = 10,
    executor_type: str = "thread"
) -> List[Any]:
    """Optimize batch processing with intelligent chunking and concurrency."""
    if not tasks:
        return []
    
    # Split tasks into optimal batch sizes
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    
    # Process batches concurrently
    batch_results = []
    for batch in batches:
        batch_result = await resource_pool.execute_batch(batch, executor_type, max_concurrent)
        batch_results.extend(batch_result)
    
    return batch_results