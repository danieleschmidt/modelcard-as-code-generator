"""Performance monitoring and profiling utilities."""

import asyncio
import gc
import psutil
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    gc_objects: int


class PerformanceTracker:
    """Tracks performance metrics for operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
        self._process = psutil.Process()
        
        # Enable memory tracing if available
        try:
            tracemalloc.start()
            self._memory_tracing = True
        except Exception:
            self._memory_tracing = False
            logger.warning("Memory tracing not available")
    
    def start_operation(self, operation_name: str, operation_id: Optional[str] = None) -> str:
        """Start tracking an operation."""
        op_id = operation_id or f"{operation_name}_{time.time()}"
        self.active_operations[op_id] = time.time()
        return op_id
    
    def end_operation(self, operation_id: str, operation_name: str, 
                     success: bool = True, error: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """End tracking an operation and record metrics."""
        end_time = time.time()
        start_time = self.active_operations.pop(operation_id, end_time)
        duration_ms = (end_time - start_time) * 1000
        
        # Get current system metrics
        try:
            memory_info = self._process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            cpu_percent = self._process.cpu_percent()
        except Exception:
            memory_usage_mb = 0.0
            cpu_percent = 0.0
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.operation_stats[operation_name].append(metrics)
        
        # Keep only recent metrics per operation
        if len(self.operation_stats[operation_name]) > 100:
            self.operation_stats[operation_name] = self.operation_stats[operation_name][-100:]
        
        return metrics
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        metrics_list = self.operation_stats.get(operation_name, [])
        
        if not metrics_list:
            return {}
        
        successful = [m for m in metrics_list if m.success]
        failed = [m for m in metrics_list if not m.success]
        
        durations = [m.duration_ms for m in successful]
        memory_usage = [m.memory_usage_mb for m in successful]
        
        stats = {
            "operation_name": operation_name,
            "total_calls": len(metrics_list),
            "successful_calls": len(successful),
            "failed_calls": len(failed),
            "success_rate": len(successful) / len(metrics_list) if metrics_list else 0.0,
            "duration_stats": {
                "count": len(durations),
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
                "avg_ms": sum(durations) / len(durations) if durations else 0,
                "median_ms": sorted(durations)[len(durations)//2] if durations else 0
            },
            "memory_stats": {
                "count": len(memory_usage),
                "min_mb": min(memory_usage) if memory_usage else 0,
                "max_mb": max(memory_usage) if memory_usage else 0,
                "avg_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0
            }
        }
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all tracked operations."""
        return {
            operation_name: self.get_operation_stats(operation_name)
            for operation_name in self.operation_stats.keys()
        }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network
            network = psutil.net_io_counters()
            
            # Process info
            process_count = len(psutil.pids())
            thread_count = self._process.num_threads()
            
            # Garbage collection info
            gc_objects = len(gc.get_objects())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_io_read_mb=(disk_io.read_bytes if disk_io else 0) / (1024 * 1024),
                disk_io_write_mb=(disk_io.write_bytes if disk_io else 0) / (1024 * 1024),
                network_bytes_sent=network.bytes_sent if network else 0,
                network_bytes_recv=network.bytes_recv if network else 0,
                process_count=process_count,
                thread_count=thread_count,
                gc_objects=gc_objects
            )
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                thread_count=0,
                gc_objects=0
            )
    
    def get_memory_profile(self) -> Optional[Dict[str, Any]]:
        """Get memory profiling information."""
        if not self._memory_tracing:
            return None
        
        try:
            current, peak = tracemalloc.get_traced_memory()
            top_stats = tracemalloc.take_snapshot().statistics('lineno')
            
            return {
                "current_mb": current / (1024 * 1024),
                "peak_mb": peak / (1024 * 1024),
                "top_allocations": [
                    {
                        "filename": stat.traceback.format()[0],
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count
                    }
                    for stat in top_stats[:5]
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get memory profile: {e}")
            return None
    
    @contextmanager
    def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking an operation."""
        op_id = self.start_operation(operation_name)
        error = None
        
        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.end_operation(op_id, operation_name, success=(error is None), 
                             error=error, metadata=metadata)
    
    @asynccontextmanager
    async def track_async_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Async context manager for tracking an operation."""
        op_id = self.start_operation(operation_name)
        error = None
        
        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.end_operation(op_id, operation_name, success=(error is None), 
                             error=error, metadata=metadata)


class PerformanceAnalyzer:
    """Analyzes performance data and provides insights."""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect potential performance issues."""
        issues = []
        
        # Check for slow operations
        for operation_name, stats in self.tracker.get_all_stats().items():
            if stats.get('duration_stats', {}).get('avg_ms', 0) > 5000:  # > 5 seconds
                issues.append({
                    "type": "slow_operation",
                    "operation": operation_name,
                    "avg_duration_ms": stats['duration_stats']['avg_ms'],
                    "severity": "warning"
                })
            
            # Check for high failure rate
            if stats.get('success_rate', 1.0) < 0.9:  # < 90% success rate
                issues.append({
                    "type": "high_failure_rate",
                    "operation": operation_name,
                    "success_rate": stats['success_rate'],
                    "failed_calls": stats['failed_calls'],
                    "severity": "error"
                })
        
        # Check system metrics
        system_metrics = self.tracker.get_system_metrics()
        
        if system_metrics.memory_percent > 90:
            issues.append({
                "type": "high_memory_usage",
                "memory_percent": system_metrics.memory_percent,
                "severity": "warning"
            })
        
        if system_metrics.cpu_percent > 90:
            issues.append({
                "type": "high_cpu_usage",
                "cpu_percent": system_metrics.cpu_percent,
                "severity": "warning"
            })
        
        return issues
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        system_metrics = self.tracker.get_system_metrics()
        operation_stats = self.tracker.get_all_stats()
        issues = self.detect_performance_issues()
        memory_profile = self.tracker.get_memory_profile()
        
        # Calculate overall stats
        total_operations = sum(stats.get('total_calls', 0) for stats in operation_stats.values())
        total_failures = sum(stats.get('failed_calls', 0) for stats in operation_stats.values())
        overall_success_rate = (total_operations - total_failures) / max(1, total_operations)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "memory_used_mb": system_metrics.memory_used_mb,
                "disk_usage_percent": system_metrics.disk_usage_percent,
                "process_count": system_metrics.process_count,
                "thread_count": system_metrics.thread_count,
                "gc_objects": system_metrics.gc_objects
            },
            "operation_summary": {
                "total_operations": total_operations,
                "total_failures": total_failures,
                "overall_success_rate": overall_success_rate,
                "tracked_operation_types": len(operation_stats)
            },
            "operation_stats": operation_stats,
            "performance_issues": issues,
            "memory_profile": memory_profile
        }


# Global performance tracker
performance_tracker = PerformanceTracker()
performance_analyzer = PerformanceAnalyzer(performance_tracker)


def performance_monitor(operation_name: Optional[str] = None, 
                       include_metadata: bool = False):
    """Decorator for performance monitoring."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                metadata = {}
                if include_metadata:
                    metadata = {
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "function_module": func.__module__
                    }
                
                async with performance_tracker.track_async_operation(op_name, metadata):
                    return await func(*args, **kwargs)
            
            async_wrapper.get_performance_stats = lambda: performance_tracker.get_operation_stats(op_name)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                metadata = {}
                if include_metadata:
                    metadata = {
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "function_module": func.__module__
                    }
                
                with performance_tracker.track_operation(op_name, metadata):
                    return func(*args, **kwargs)
            
            sync_wrapper.get_performance_stats = lambda: performance_tracker.get_operation_stats(op_name)
            return sync_wrapper
    
    return decorator


class PerformanceBenchmark:
    """Benchmarking utilities for performance testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[float] = []
    
    @contextmanager
    def measure(self):
        """Context manager for measuring execution time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.results.append(duration)
    
    def run_benchmark(self, func: Callable, iterations: int = 10, *args, **kwargs) -> Dict[str, float]:
        """Run benchmark on a function."""
        self.results.clear()
        
        for _ in range(iterations):
            with self.measure():
                func(*args, **kwargs)
        
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, float]:
        """Get benchmark statistics."""
        if not self.results:
            return {}
        
        sorted_results = sorted(self.results)
        n = len(sorted_results)
        
        return {
            "count": n,
            "min_seconds": min(sorted_results),
            "max_seconds": max(sorted_results),
            "mean_seconds": sum(sorted_results) / n,
            "median_seconds": sorted_results[n // 2],
            "p95_seconds": sorted_results[int(n * 0.95)],
            "p99_seconds": sorted_results[int(n * 0.99)],
            "std_dev": (sum((x - sum(sorted_results)/n)**2 for x in sorted_results) / n)**0.5
        }