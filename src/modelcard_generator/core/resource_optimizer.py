"""Resource optimization and adaptive scaling utilities."""

import asyncio
import gc
import os
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import psutil

from .logging_config import get_logger
from .performance_monitor import performance_tracker

logger = get_logger(__name__)


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_threads: int
    gc_objects: int
    load_average: Optional[Tuple[float, float, float]] = None


@dataclass
class OptimizationStrategy:
    """Configuration for resource optimization strategies."""
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_io_optimization: bool = True
    enable_gc_tuning: bool = True

    # Thresholds
    memory_high_threshold: float = 80.0  # %
    memory_critical_threshold: float = 95.0  # %
    cpu_high_threshold: float = 80.0  # %
    cpu_critical_threshold: float = 95.0  # %

    # Adaptive scaling
    enable_adaptive_scaling: bool = True
    min_workers: int = 2
    max_workers: int = 20
    scale_up_threshold: float = 75.0
    scale_down_threshold: float = 25.0
    scale_cooldown_seconds: int = 60


class ResourceMonitor:
    """Monitors system resources and provides optimization recommendations."""

    def __init__(self, monitoring_interval: int = 30, history_size: int = 100):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self._process = psutil.Process()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self._lock = Lock()

        # Baseline metrics
        self._baseline_metrics: Optional[ResourceMetrics] = None

    async def start_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Resource monitoring started (interval: {self.monitoring_interval}s)")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Resource monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                metrics = await self.collect_metrics()

                with self._lock:
                    self.metrics_history.append(metrics)

                    # Set baseline if not set
                    if self._baseline_metrics is None:
                        self._baseline_metrics = metrics

                # Log significant changes
                self._log_resource_changes(metrics)

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        try:
            # CPU metrics
            cpu_percent = self._process.cpu_percent(interval=0.1)

            # Memory metrics
            memory_info = self._process.memory_info()
            system_memory = psutil.virtual_memory()

            # I/O metrics
            io_counters = psutil.disk_io_counters()
            network_counters = psutil.net_io_counters()

            # Thread count
            active_threads = self._process.num_threads()

            # Garbage collection
            gc_objects = len(gc.get_objects())

            # Load average (Unix-like systems)
            load_avg = None
            if hasattr(os, "getloadavg"):
                load_avg = os.getloadavg()

            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=system_memory.percent,
                memory_used_mb=memory_info.rss / (1024 * 1024),
                memory_available_mb=system_memory.available / (1024 * 1024),
                disk_io_read_mb=(io_counters.read_bytes if io_counters else 0) / (1024 * 1024),
                disk_io_write_mb=(io_counters.write_bytes if io_counters else 0) / (1024 * 1024),
                network_bytes_sent=network_counters.bytes_sent if network_counters else 0,
                network_bytes_recv=network_counters.bytes_recv if network_counters else 0,
                active_threads=active_threads,
                gc_objects=gc_objects,
                load_average=load_avg
            )

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            raise

    def _log_resource_changes(self, current_metrics: ResourceMetrics) -> None:
        """Log significant resource changes."""
        if not self._baseline_metrics:
            return

        baseline = self._baseline_metrics

        # Check for significant changes
        memory_change = current_metrics.memory_used_mb - baseline.memory_used_mb
        cpu_spike = current_metrics.cpu_percent > 90.0
        memory_spike = current_metrics.memory_percent > 90.0

        if abs(memory_change) > 100:  # 100MB change
            logger.info(f"Memory usage change: {memory_change:+.1f}MB")

        if cpu_spike:
            logger.warning(f"High CPU usage: {current_metrics.cpu_percent:.1f}%")

        if memory_spike:
            logger.warning(f"High memory usage: {current_metrics.memory_percent:.1f}%")

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, minutes: int = 10) -> List[ResourceMetrics]:
        """Get metrics history for the last N minutes."""
        cutoff = time.time() - (minutes * 60)

        with self._lock:
            return [m for m in self.metrics_history if m.timestamp > cutoff]

    def get_resource_trends(self) -> Dict[str, Any]:
        """Analyze resource usage trends."""
        with self._lock:
            if len(self.metrics_history) < 2:
                return {}

            metrics = list(self.metrics_history)

        # Calculate trends over last 10 data points
        recent_metrics = metrics[-10:] if len(metrics) >= 10 else metrics

        if len(recent_metrics) < 2:
            return {}

        # Calculate slopes (trend direction)
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        gc_values = [m.gc_objects for m in recent_metrics]

        def calculate_trend(values: List[float]) -> str:
            if len(values) < 2:
                return "stable"

            # Simple linear trend
            n = len(values)
            sum_x = sum(range(n))
            sum_y = sum(values)
            sum_xy = sum(i * y for i, y in enumerate(values))
            sum_x2 = sum(i * i for i in range(n))

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"

        return {
            "cpu_trend": calculate_trend(cpu_values),
            "memory_trend": calculate_trend(memory_values),
            "gc_objects_trend": calculate_trend([float(v) for v in gc_values]),
            "avg_cpu": sum(cpu_values) / len(cpu_values),
            "avg_memory": sum(memory_values) / len(memory_values),
            "memory_growth_mb": memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        }


class ResourceOptimizer:
    """Optimizes resource usage based on monitoring data."""

    def __init__(self, monitor: ResourceMonitor, strategy: OptimizationStrategy):
        self.monitor = monitor
        self.strategy = strategy
        self._last_optimization = 0.0
        self._optimization_history: deque = deque(maxlen=100)
        self._optimization_cooldown = 30.0  # seconds

    async def optimize_resources(self, force: bool = False) -> Dict[str, Any]:
        """Perform resource optimization."""
        current_time = time.time()

        # Check cooldown
        if not force and (current_time - self._last_optimization) < self._optimization_cooldown:
            return {"status": "skipped", "reason": "cooldown"}

        metrics = self.monitor.get_current_metrics()
        if not metrics:
            return {"status": "skipped", "reason": "no_metrics"}

        optimizations = []

        # Memory optimization
        if self.strategy.enable_memory_optimization:
            memory_opts = await self._optimize_memory(metrics)
            optimizations.extend(memory_opts)

        # CPU optimization
        if self.strategy.enable_cpu_optimization:
            cpu_opts = await self._optimize_cpu(metrics)
            optimizations.extend(cpu_opts)

        # Garbage collection optimization
        if self.strategy.enable_gc_tuning:
            gc_opts = await self._optimize_garbage_collection(metrics)
            optimizations.extend(gc_opts)

        self._last_optimization = current_time
        self._optimization_history.append({
            "timestamp": current_time,
            "optimizations": optimizations,
            "metrics": metrics
        })

        if optimizations:
            logger.info(f"Applied {len(optimizations)} resource optimizations")

        return {
            "status": "completed",
            "optimizations": optimizations,
            "metrics": {
                "memory_percent": metrics.memory_percent,
                "cpu_percent": metrics.cpu_percent,
                "gc_objects": metrics.gc_objects
            }
        }

    async def _optimize_memory(self, metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Optimize memory usage."""
        optimizations = []

        # Force garbage collection if memory is high
        if metrics.memory_percent > self.strategy.memory_high_threshold:
            gc_collected = gc.collect()
            optimizations.append({
                "type": "memory_gc",
                "action": "forced_garbage_collection",
                "objects_collected": gc_collected
            })
            logger.info(f"Forced GC collected {gc_collected} objects")

        # Clear caches if memory is critical
        if metrics.memory_percent > self.strategy.memory_critical_threshold:
            # Clear performance tracker cache
            performance_tracker.metrics_history.clear()
            optimizations.append({
                "type": "memory_cache",
                "action": "cleared_performance_cache"
            })
            logger.warning("Cleared caches due to critical memory usage")

        return optimizations

    async def _optimize_cpu(self, metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Optimize CPU usage."""
        optimizations = []

        # Reduce thread pool size if CPU is high
        if metrics.cpu_percent > self.strategy.cpu_high_threshold:
            # This would need integration with actual thread pools
            optimizations.append({
                "type": "cpu_threads",
                "action": "recommend_reduce_workers",
                "current_threads": metrics.active_threads
            })
            logger.warning(f"High CPU usage ({metrics.cpu_percent:.1f}%), consider reducing workers")

        return optimizations

    async def _optimize_garbage_collection(self, metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Optimize garbage collection settings."""
        optimizations = []

        # Tune GC based on object count
        if metrics.gc_objects > 100000:  # 100k objects
            # Get GC stats
            gc_stats = gc.get_stats()

            # Adjust GC thresholds for better performance
            current_thresholds = gc.get_threshold()

            # Increase thresholds to reduce GC frequency
            new_thresholds = (
                min(current_thresholds[0] * 2, 2000),
                min(current_thresholds[1] * 2, 20000),
                min(current_thresholds[2] * 2, 200000)
            )

            gc.set_threshold(*new_thresholds)

            optimizations.append({
                "type": "gc_tuning",
                "action": "adjusted_thresholds",
                "old_thresholds": current_thresholds,
                "new_thresholds": new_thresholds,
                "gc_stats": gc_stats
            })

            logger.info(f"Adjusted GC thresholds: {current_thresholds} -> {new_thresholds}")

        return optimizations

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization actions."""
        return list(self._optimization_history)


class AdaptiveScaler:
    """Adaptively scales resources based on load."""

    def __init__(self, monitor: ResourceMonitor, strategy: OptimizationStrategy):
        self.monitor = monitor
        self.strategy = strategy
        self._current_workers = strategy.min_workers
        self._last_scale_time = 0.0
        self._scaling_history: deque = deque(maxlen=50)

    async def should_scale_up(self) -> bool:
        """Determine if resources should be scaled up."""
        trends = self.monitor.get_resource_trends()
        current_metrics = self.monitor.get_current_metrics()

        if not current_metrics or not trends:
            return False

        # Scale up conditions
        high_cpu = current_metrics.cpu_percent > self.strategy.scale_up_threshold
        high_memory = current_metrics.memory_percent > self.strategy.scale_up_threshold
        increasing_trend = trends.get("cpu_trend") == "increasing"

        # Check cooldown
        cooldown_elapsed = (time.time() - self._last_scale_time) > self.strategy.scale_cooldown_seconds

        return (high_cpu or high_memory or increasing_trend) and cooldown_elapsed and self._current_workers < self.strategy.max_workers

    async def should_scale_down(self) -> bool:
        """Determine if resources should be scaled down."""
        trends = self.monitor.get_resource_trends()
        current_metrics = self.monitor.get_current_metrics()

        if not current_metrics or not trends:
            return False

        # Scale down conditions
        low_cpu = current_metrics.cpu_percent < self.strategy.scale_down_threshold
        low_memory = current_metrics.memory_percent < self.strategy.scale_down_threshold
        decreasing_trend = trends.get("cpu_trend") == "decreasing"

        # Check cooldown
        cooldown_elapsed = (time.time() - self._last_scale_time) > self.strategy.scale_cooldown_seconds

        return (low_cpu and low_memory) or decreasing_trend and cooldown_elapsed and self._current_workers > self.strategy.min_workers

    async def scale_resources(self) -> Optional[Dict[str, Any]]:
        """Scale resources up or down as needed."""
        if await self.should_scale_up():
            return await self._scale_up()
        elif await self.should_scale_down():
            return await self._scale_down()

        return None

    async def _scale_up(self) -> Dict[str, Any]:
        """Scale resources up."""
        old_workers = self._current_workers
        self._current_workers = min(self._current_workers + 1, self.strategy.max_workers)
        self._last_scale_time = time.time()

        scale_info = {
            "action": "scale_up",
            "old_workers": old_workers,
            "new_workers": self._current_workers,
            "timestamp": self._last_scale_time
        }

        self._scaling_history.append(scale_info)
        logger.info(f"Scaled up workers: {old_workers} -> {self._current_workers}")

        return scale_info

    async def _scale_down(self) -> Dict[str, Any]:
        """Scale resources down."""
        old_workers = self._current_workers
        self._current_workers = max(self._current_workers - 1, self.strategy.min_workers)
        self._last_scale_time = time.time()

        scale_info = {
            "action": "scale_down",
            "old_workers": old_workers,
            "new_workers": self._current_workers,
            "timestamp": self._last_scale_time
        }

        self._scaling_history.append(scale_info)
        logger.info(f"Scaled down workers: {old_workers} -> {self._current_workers}")

        return scale_info

    def get_current_scale(self) -> int:
        """Get current worker count."""
        return self._current_workers

    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get scaling history."""
        return list(self._scaling_history)


class ResourceManager:
    """Central resource management system."""

    def __init__(self, strategy: Optional[OptimizationStrategy] = None):
        self.strategy = strategy or OptimizationStrategy()
        self.monitor = ResourceMonitor()
        self.optimizer = ResourceOptimizer(self.monitor, self.strategy)
        self.scaler = AdaptiveScaler(self.monitor, self.strategy)

        self._management_task: Optional[asyncio.Task] = None
        self._is_managing = False

    async def start(self) -> None:
        """Start resource management."""
        if self._is_managing:
            return

        await self.monitor.start_monitoring()

        self._is_managing = True
        self._management_task = asyncio.create_task(self._management_loop())

        logger.info("Resource management started")

    async def stop(self) -> None:
        """Stop resource management."""
        if not self._is_managing:
            return

        self._is_managing = False

        if self._management_task:
            self._management_task.cancel()
            try:
                await self._management_task
            except asyncio.CancelledError:
                pass

        await self.monitor.stop_monitoring()

        logger.info("Resource management stopped")

    async def _management_loop(self) -> None:
        """Main management loop."""
        while self._is_managing:
            try:
                # Optimize resources
                if self.strategy.enable_adaptive_scaling:
                    scale_result = await self.scaler.scale_resources()
                    if scale_result:
                        logger.info(f"Resource scaling: {scale_result}")

                # Optimize resources every 2 minutes
                await self.optimizer.optimize_resources()

                # Sleep for 30 seconds before next iteration
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in management loop: {e}")
                await asyncio.sleep(30)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive resource management status."""
        current_metrics = self.monitor.get_current_metrics()
        trends = self.monitor.get_resource_trends()
        optimization_history = self.optimizer.get_optimization_history()
        scaling_history = self.scaler.get_scaling_history()

        return {
            "is_managing": self._is_managing,
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "trends": trends,
            "current_workers": self.scaler.get_current_scale(),
            "recent_optimizations": len([o for o in optimization_history if time.time() - o["timestamp"] < 300]),  # last 5 min
            "recent_scaling_actions": len([s for s in scaling_history if time.time() - s["timestamp"] < 300]),  # last 5 min
            "strategy": self.strategy.__dict__
        }


# Global resource manager instance
resource_manager: Optional[ResourceManager] = None


async def get_resource_manager() -> ResourceManager:
    """Get or create global resource manager."""
    global resource_manager

    if resource_manager is None:
        resource_manager = ResourceManager()
        await resource_manager.start()

    return resource_manager


async def shutdown_resource_manager() -> None:
    """Shutdown global resource manager."""
    global resource_manager

    if resource_manager:
        await resource_manager.stop()
        resource_manager = None
