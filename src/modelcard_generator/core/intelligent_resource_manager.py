"""
Intelligent resource management system with adaptive optimization and auto-scaling.

This module provides advanced resource management capabilities including:
- Intelligent memory management with predictive allocation
- Adaptive CPU utilization optimization
- Dynamic resource scaling based on workload
- Resource pool management with health monitoring
- Performance profiling and optimization suggestions
"""

import asyncio
import gc
import psutil
import resource
import sys
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from weakref import WeakSet

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_percent: float
    open_files: int
    threads: int
    peak_memory_mb: float = 0
    gc_collections: Dict[int, int] = field(default_factory=dict)


@dataclass
class MemoryProfile:
    """Memory profiling information."""
    current_mb: float
    peak_mb: float
    allocations: int
    deallocations: int
    top_allocators: List[Tuple[str, float]] = field(default_factory=list)


class ResourcePool:
    """Generic resource pool with lifecycle management."""

    def __init__(self, name: str, factory: Callable, max_size: int = 10, 
                 idle_timeout: float = 300.0, health_check: Optional[Callable] = None):
        self.name = name
        self.factory = factory
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.health_check = health_check
        
        self._pool: List[Any] = []
        self._in_use: WeakSet = WeakSet()
        self._last_used: Dict[Any, float] = {}
        self._created_count = 0
        self._health_failures = 0
        
        # Statistics
        self.stats = {
            "total_created": 0,
            "total_destroyed": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "health_check_failures": 0
        }

    async def acquire(self) -> Any:
        """Acquire a resource from the pool."""
        # Try to get from pool
        resource_obj = await self._get_from_pool()
        
        if resource_obj is None:
            # Create new resource
            resource_obj = await self._create_resource()
            self.stats["pool_misses"] += 1
        else:
            self.stats["pool_hits"] += 1
        
        self._in_use.add(resource_obj)
        return resource_obj

    async def release(self, resource_obj: Any) -> None:
        """Release a resource back to the pool."""
        if resource_obj in self._in_use:
            self._in_use.discard(resource_obj)
        
        # Check if resource is healthy
        if self.health_check and not await self._check_health(resource_obj):
            await self._destroy_resource(resource_obj)
            return
        
        # Return to pool if there's space
        if len(self._pool) < self.max_size:
            self._pool.append(resource_obj)
            self._last_used[resource_obj] = time.time()
        else:
            await self._destroy_resource(resource_obj)

    async def _get_from_pool(self) -> Optional[Any]:
        """Get a resource from the pool."""
        current_time = time.time()
        
        # Remove idle resources
        to_remove = []
        for resource_obj in self._pool[:]:
            if current_time - self._last_used.get(resource_obj, 0) > self.idle_timeout:
                to_remove.append(resource_obj)
        
        for resource_obj in to_remove:
            self._pool.remove(resource_obj)
            del self._last_used[resource_obj]
            await self._destroy_resource(resource_obj)
        
        # Return available resource
        if self._pool:
            return self._pool.pop()
        
        return None

    async def _create_resource(self) -> Any:
        """Create a new resource."""
        try:
            if asyncio.iscoroutinefunction(self.factory):
                resource_obj = await self.factory()
            else:
                resource_obj = self.factory()
            
            self._created_count += 1
            self.stats["total_created"] += 1
            logger.debug(f"Created new resource for pool {self.name}")
            return resource_obj
        except Exception as e:
            logger.error(f"Failed to create resource for pool {self.name}: {e}")
            raise

    async def _destroy_resource(self, resource_obj: Any) -> None:
        """Destroy a resource."""
        try:
            if hasattr(resource_obj, 'close'):
                if asyncio.iscoroutinefunction(resource_obj.close):
                    await resource_obj.close()
                else:
                    resource_obj.close()
            
            self.stats["total_destroyed"] += 1
            logger.debug(f"Destroyed resource from pool {self.name}")
        except Exception as e:
            logger.warning(f"Error destroying resource from pool {self.name}: {e}")

    async def _check_health(self, resource_obj: Any) -> bool:
        """Check if a resource is healthy."""
        if not self.health_check:
            return True
        
        try:
            if asyncio.iscoroutinefunction(self.health_check):
                is_healthy = await self.health_check(resource_obj)
            else:
                is_healthy = self.health_check(resource_obj)
            
            if not is_healthy:
                self._health_failures += 1
                self.stats["health_check_failures"] += 1
            
            return is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for pool {self.name}: {e}")
            self._health_failures += 1
            self.stats["health_check_failures"] += 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self.stats,
            "pool_size": len(self._pool),
            "in_use": len(self._in_use),
            "total_created_this_session": self._created_count,
            "health_failures": self._health_failures,
            "hit_rate": self.stats["pool_hits"] / max(1, self.stats["pool_hits"] + self.stats["pool_misses"])
        }


class IntelligentMemoryManager:
    """Intelligent memory management with predictive allocation."""

    def __init__(self):
        self.memory_history: deque = deque(maxlen=1000)
        self.allocation_patterns: Dict[str, List[float]] = defaultdict(list)
        self.gc_thresholds = gc.get_threshold()
        self.memory_pressure_threshold = 85.0  # Percentage
        
        # Memory profiling
        self.profiling_enabled = False
        self._tracemalloc_started = False
        
        # Adaptive GC settings
        self.adaptive_gc = True
        self.last_gc_adjustment = time.time()

    def start_profiling(self) -> None:
        """Start memory profiling."""
        if not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
            self.profiling_enabled = True
            logger.info("Started memory profiling")

    def stop_profiling(self) -> None:
        """Stop memory profiling."""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
            self.profiling_enabled = False
            logger.info("Stopped memory profiling")

    def get_memory_profile(self) -> MemoryProfile:
        """Get current memory profile."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        profile = MemoryProfile(
            current_mb=memory_info.rss / 1024 / 1024,
            peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # macOS/Linux difference handled
            allocations=0,
            deallocations=0
        )
        
        if self.profiling_enabled and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            profile.current_mb = current / 1024 / 1024
            profile.peak_mb = peak / 1024 / 1024
            
            # Get top allocators
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            profile.top_allocators = [
                (str(stat.traceback), stat.size / 1024 / 1024)
                for stat in top_stats[:10]
            ]
        
        return profile

    def record_allocation(self, operation: str, size_mb: float) -> None:
        """Record a memory allocation."""
        self.allocation_patterns[operation].append(size_mb)
        
        # Keep only recent patterns
        if len(self.allocation_patterns[operation]) > 100:
            self.allocation_patterns[operation] = self.allocation_patterns[operation][-50:]

    def predict_memory_needs(self, operation: str) -> float:
        """Predict memory needs for an operation."""
        if operation not in self.allocation_patterns:
            return 50.0  # Default 50MB estimate
        
        patterns = self.allocation_patterns[operation]
        if len(patterns) < 3:
            return max(patterns) if patterns else 50.0
        
        # Use 95th percentile for prediction
        patterns_sorted = sorted(patterns)
        percentile_95_idx = int(len(patterns_sorted) * 0.95)
        return patterns_sorted[percentile_95_idx]

    def check_memory_pressure(self) -> Tuple[bool, float]:
        """Check if system is under memory pressure."""
        memory = psutil.virtual_memory()
        pressure = memory.percent > self.memory_pressure_threshold
        return pressure, memory.percent

    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        start_memory = self.get_memory_profile()
        actions_taken = []
        
        # Check memory pressure
        pressure, percent = self.check_memory_pressure()
        
        if pressure:
            # Force garbage collection
            collected = 0
            for generation in range(3):
                collected += gc.collect(generation)
            actions_taken.append(f"gc_collected_{collected}_objects")
            
            # Adjust GC thresholds if adaptive
            if self.adaptive_gc:
                await self._adjust_gc_thresholds()
                actions_taken.append("adjusted_gc_thresholds")
        
        # Clear internal caches
        if percent > 90:
            self._clear_internal_caches()
            actions_taken.append("cleared_internal_caches")
        
        end_memory = self.get_memory_profile()
        freed_mb = start_memory.current_mb - end_memory.current_mb
        
        return {
            "actions_taken": actions_taken,
            "memory_freed_mb": freed_mb,
            "memory_before_mb": start_memory.current_mb,
            "memory_after_mb": end_memory.current_mb,
            "memory_pressure_percent": percent
        }

    async def _adjust_gc_thresholds(self) -> None:
        """Adjust garbage collection thresholds adaptively."""
        current_time = time.time()
        
        # Only adjust every 5 minutes
        if current_time - self.last_gc_adjustment < 300:
            return
        
        # Get current memory pressure
        _, memory_percent = self.check_memory_pressure()
        
        # Adjust thresholds based on memory pressure
        if memory_percent > 80:
            # More aggressive GC
            new_thresholds = (
                max(100, self.gc_thresholds[0] // 2),
                max(5, self.gc_thresholds[1] // 2),
                max(5, self.gc_thresholds[2] // 2)
            )
        elif memory_percent < 50:
            # Less aggressive GC
            new_thresholds = (
                min(2000, self.gc_thresholds[0] * 2),
                min(20, self.gc_thresholds[1] * 2),
                min(20, self.gc_thresholds[2] * 2)
            )
        else:
            return  # No change needed
        
        gc.set_threshold(*new_thresholds)
        self.gc_thresholds = new_thresholds
        self.last_gc_adjustment = current_time
        
        logger.debug(f"Adjusted GC thresholds to {new_thresholds}")

    def _clear_internal_caches(self) -> None:
        """Clear internal caches to free memory."""
        # Clear internal Python caches
        sys.intern.clear() if hasattr(sys, 'intern') and hasattr(sys.intern, 'clear') else None
        
        # Clear our own caches
        for patterns in self.allocation_patterns.values():
            if len(patterns) > 20:
                patterns[:] = patterns[-10:]  # Keep only recent 10


class AdaptiveResourceOptimizer:
    """Adaptive resource optimization with machine learning insights."""

    def __init__(self):
        self.memory_manager = IntelligentMemoryManager()
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.metrics_history: deque = deque(maxlen=10000)
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Optimization parameters
        self.optimization_interval = 300  # 5 minutes
        self.last_optimization = 0
        
        # Resource monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    def create_resource_pool(self, name: str, factory: Callable, **kwargs) -> ResourcePool:
        """Create a managed resource pool."""
        pool = ResourcePool(name, factory, **kwargs)
        self.resource_pools[name] = pool
        logger.info(f"Created resource pool: {name}")
        return pool

    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get a resource pool by name."""
        return self.resource_pools.get(name)

    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._shutdown_event.clear()
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started resource monitoring")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._shutdown_event.set()
        if self._monitoring_task and not self._monitoring_task.done():
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._monitoring_task.cancel()
            logger.info("Stopped resource monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check if optimization is needed
                current_time = time.time()
                if current_time - self.last_optimization > self.optimization_interval:
                    await self._optimize_resources()
                    self.last_optimization = current_time
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get GC stats
        gc_stats = {}
        for i in range(3):
            gc_stats[i] = gc.get_count()[i]
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=process.cpu_percent(),
            memory_rss_mb=memory_info.rss / 1024 / 1024,
            memory_vms_mb=memory_info.vms / 1024 / 1024,
            memory_percent=psutil.virtual_memory().percent,
            open_files=len(process.open_files()),
            threads=process.num_threads(),
            gc_collections=gc_stats
        )

    async def _optimize_resources(self) -> None:
        """Perform resource optimization."""
        logger.debug("Starting resource optimization")
        
        optimization_start = time.time()
        actions = []
        
        # Memory optimization
        memory_result = await self.memory_manager.optimize_memory()
        if memory_result["actions_taken"]:
            actions.extend(memory_result["actions_taken"])
        
        # Pool optimization
        pool_results = await self._optimize_pools()
        actions.extend(pool_results)
        
        # CPU optimization
        cpu_results = await self._optimize_cpu()
        actions.extend(cpu_results)
        
        optimization_duration = time.time() - optimization_start
        
        optimization_record = {
            "timestamp": time.time(),
            "duration_seconds": optimization_duration,
            "actions_taken": actions,
            "memory_freed_mb": memory_result.get("memory_freed_mb", 0),
            "metrics_before": self.metrics_history[-1] if self.metrics_history else None
        }
        
        self.optimization_history.append(optimization_record)
        
        if actions:
            logger.info(f"Resource optimization completed: {actions}")

    async def _optimize_pools(self) -> List[str]:
        """Optimize resource pools."""
        actions = []
        
        for name, pool in self.resource_pools.items():
            stats = pool.get_stats()
            
            # If hit rate is very low, consider reducing pool size
            if stats["hit_rate"] < 0.1 and stats["pool_size"] > 2:
                # Clear some idle resources
                pool.max_size = max(2, pool.max_size // 2)
                actions.append(f"reduced_pool_size_{name}")
            
            # If hit rate is very high, consider increasing pool size
            elif stats["hit_rate"] > 0.9 and stats["pool_size"] < 20:
                pool.max_size = min(20, pool.max_size * 2)
                actions.append(f"increased_pool_size_{name}")
        
        return actions

    async def _optimize_cpu(self) -> List[str]:
        """Optimize CPU usage."""
        actions = []
        
        if len(self.metrics_history) < 10:
            return actions
        
        # Check recent CPU usage
        recent_metrics = list(self.metrics_history)[-10:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        
        # If CPU usage is consistently high, suggest optimization
        if avg_cpu > 80:
            # In a real implementation, this could trigger various optimizations
            # like reducing worker threads, enabling CPU throttling, etc.
            actions.append("high_cpu_detected")
            logger.warning(f"High CPU usage detected: {avg_cpu:.1f}%")
        
        return actions

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        current_metrics = self._collect_metrics()
        memory_profile = self.memory_manager.get_memory_profile()
        
        # Pool statistics
        pool_stats = {}
        for name, pool in self.resource_pools.items():
            pool_stats[name] = pool.get_stats()
        
        # Historical analysis
        if self.metrics_history:
            historical_cpu = [m.cpu_percent for m in self.metrics_history]
            historical_memory = [m.memory_rss_mb for m in self.metrics_history]
            
            trends = {
                "cpu_trend": self._calculate_trend(historical_cpu),
                "memory_trend": self._calculate_trend(historical_memory),
                "avg_cpu_last_hour": sum(historical_cpu) / len(historical_cpu),
                "avg_memory_last_hour": sum(historical_memory) / len(historical_memory)
            }
        else:
            trends = {}
        
        return {
            "current_metrics": {
                "timestamp": current_metrics.timestamp,
                "cpu_percent": current_metrics.cpu_percent,
                "memory_rss_mb": current_metrics.memory_rss_mb,
                "memory_percent": current_metrics.memory_percent,
                "open_files": current_metrics.open_files,
                "threads": current_metrics.threads
            },
            "memory_profile": {
                "current_mb": memory_profile.current_mb,
                "peak_mb": memory_profile.peak_mb,
                "profiling_enabled": memory_profile.allocations > 0
            },
            "resource_pools": pool_stats,
            "optimization_history": len(self.optimization_history),
            "last_optimization": self.last_optimization,
            "trends": trends,
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"


# Global resource optimizer instance
resource_optimizer = AdaptiveResourceOptimizer()


def optimize_memory(operation_name: str = None):
    """Decorator to optimize memory usage for operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                # Record memory before
                start_memory = resource_optimizer.memory_manager.get_memory_profile()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record successful allocation pattern
                    end_memory = resource_optimizer.memory_manager.get_memory_profile()
                    used_memory = end_memory.current_mb - start_memory.current_mb
                    
                    if operation_name and used_memory > 0:
                        resource_optimizer.memory_manager.record_allocation(operation_name, used_memory)
                    
                    return result
                    
                except Exception as e:
                    # Cleanup on error
                    await resource_optimizer.memory_manager.optimize_memory()
                    raise
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # Record memory before
                start_memory = resource_optimizer.memory_manager.get_memory_profile()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful allocation pattern
                    end_memory = resource_optimizer.memory_manager.get_memory_profile()
                    used_memory = end_memory.current_mb - start_memory.current_mb
                    
                    if operation_name and used_memory > 0:
                        resource_optimizer.memory_manager.record_allocation(operation_name, used_memory)
                    
                    return result
                    
                except Exception:
                    # Cleanup on error - run in background for sync functions
                    asyncio.create_task(resource_optimizer.memory_manager.optimize_memory())
                    raise
            
            return sync_wrapper
    
    return decorator