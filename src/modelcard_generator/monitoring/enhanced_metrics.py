"""Enhanced metrics collection and monitoring for model card generation."""

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil

from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricSnapshot:
    """A point-in-time snapshot of system metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    open_files: int


@dataclass
class OperationMetrics:
    """Metrics for a specific operation."""
    operation_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))
    error_types: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_calls / max(1, self.total_calls)

    @property
    def average_duration(self) -> float:
        """Calculate average duration."""
        return self.total_duration / max(1, self.successful_calls)

    @property
    def p95_duration(self) -> float:
        """Calculate 95th percentile duration."""
        if not self.recent_durations:
            return 0.0
        sorted_durations = sorted(self.recent_durations)
        index = int(len(sorted_durations) * 0.95)
        return sorted_durations[index] if index < len(sorted_durations) else sorted_durations[-1]


class SystemMonitor:
    """Advanced system resource monitoring."""

    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.snapshots: deque[MetricSnapshot] = deque(maxlen=1440)  # 24 hours at 1min intervals
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Baseline measurements
        self.baseline_snapshot: Optional[MetricSnapshot] = None

    async def start(self) -> None:
        """Start system monitoring."""
        if self.running:
            return

        self.running = True
        self.baseline_snapshot = await self._collect_snapshot()
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")

    async def stop(self) -> None:
        """Stop system monitoring."""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                snapshot = await self._collect_snapshot()
                self.snapshots.append(snapshot)

                # Check for anomalies
                await self._check_anomalies(snapshot)

                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)

    async def _collect_snapshot(self) -> MetricSnapshot:
        """Collect current system metrics."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._collect_metrics_sync)

    def _collect_metrics_sync(self) -> MetricSnapshot:
        """Synchronous metrics collection."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            io_counters = process.io_counters()

            return MetricSnapshot(
                timestamp=datetime.now(),
                cpu_percent=process.cpu_percent(),
                memory_percent=process.memory_percent(),
                memory_used_mb=memory_info.rss / 1024 / 1024,
                disk_io_read_mb=io_counters.read_bytes / 1024 / 1024,
                disk_io_write_mb=io_counters.write_bytes / 1024 / 1024,
                network_sent_mb=0,  # Process-level network stats not available in psutil
                network_recv_mb=0,
                active_threads=process.num_threads(),
                open_files=process.num_fds() if hasattr(process, "num_fds") else 0
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return MetricSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                active_threads=0,
                open_files=0
            )

    async def _check_anomalies(self, snapshot: MetricSnapshot) -> None:
        """Check for system anomalies."""
        if not self.baseline_snapshot or len(self.snapshots) < 10:
            return

        # Check for significant changes
        cpu_increase = snapshot.cpu_percent - self.baseline_snapshot.cpu_percent
        memory_increase = snapshot.memory_used_mb - self.baseline_snapshot.memory_used_mb

        if cpu_increase > 50:  # 50% CPU increase
            logger.warning(f"High CPU usage detected: {snapshot.cpu_percent:.1f}% (+{cpu_increase:.1f}%)")

        if memory_increase > 500:  # 500MB memory increase
            logger.warning(f"High memory usage detected: {snapshot.memory_used_mb:.1f}MB (+{memory_increase:.1f}MB)")

    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get current system metrics."""
        return self.snapshots[-1] if self.snapshots else None

    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get system metrics summary for the specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {}

        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_snapshots),
            "avg_cpu_percent": sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots),
            "max_cpu_percent": max(s.cpu_percent for s in recent_snapshots),
            "avg_memory_mb": sum(s.memory_used_mb for s in recent_snapshots) / len(recent_snapshots),
            "max_memory_mb": max(s.memory_used_mb for s in recent_snapshots),
            "total_disk_io_mb": sum(s.disk_io_read_mb + s.disk_io_write_mb for s in recent_snapshots),
            "avg_threads": sum(s.active_threads for s in recent_snapshots) / len(recent_snapshots),
            "max_open_files": max(s.open_files for s in recent_snapshots)
        }


class PerformanceTracker:
    """Enhanced performance tracking with advanced analytics."""

    def __init__(self):
        self.operations: Dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self.custom_metrics: Dict[str, List[float]] = defaultdict(list)
        self.tags: Dict[str, Dict[str, Any]] = defaultdict(dict)

    @asynccontextmanager
    async def track_operation(self, operation_name: str, tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracking operation performance."""
        start_time = time.perf_counter()
        operation = self.operations[operation_name]
        operation.operation_name = operation_name
        operation.total_calls += 1

        if tags:
            self.tags[operation_name].update(tags)

        try:
            yield operation

            # Success
            duration = time.perf_counter() - start_time
            operation.successful_calls += 1
            operation.total_duration += duration
            operation.min_duration = min(operation.min_duration, duration)
            operation.max_duration = max(operation.max_duration, duration)
            operation.recent_durations.append(duration)

        except Exception as e:
            # Failure
            operation.failed_calls += 1
            error_type = type(e).__name__
            operation.error_types[error_type] = operation.error_types.get(error_type, 0) + 1
            raise

    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a custom metric value."""
        self.custom_metrics[metric_name].append(value)

        # Keep only recent values
        if len(self.custom_metrics[metric_name]) > 1000:
            self.custom_metrics[metric_name] = self.custom_metrics[metric_name][-1000:]

        if tags:
            tag_key = f"{metric_name}_tags"
            if tag_key not in self.tags:
                self.tags[tag_key] = {}
            self.tags[tag_key].update(tags)

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an operation."""
        if operation_name not in self.operations:
            return {}

        op = self.operations[operation_name]
        return {
            "name": operation_name,
            "total_calls": op.total_calls,
            "successful_calls": op.successful_calls,
            "failed_calls": op.failed_calls,
            "success_rate": op.success_rate,
            "avg_duration_ms": op.average_duration * 1000,
            "min_duration_ms": op.min_duration * 1000 if op.min_duration != float("inf") else 0,
            "max_duration_ms": op.max_duration * 1000,
            "p95_duration_ms": op.p95_duration * 1000,
            "error_types": dict(op.error_types),
            "tags": self.tags.get(operation_name, {})
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked operations."""
        return {name: self.get_operation_stats(name) for name in self.operations.keys()}

    def get_custom_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a custom metric."""
        if metric_name not in self.custom_metrics:
            return {}

        values = self.custom_metrics[metric_name]
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": sorted_values[len(sorted_values) // 2],
            "p95": sorted_values[int(len(sorted_values) * 0.95)],
            "p99": sorted_values[int(len(sorted_values) * 0.99)],
            "tags": self.tags.get(f"{metric_name}_tags", {})
        }


class AlertManager:
    """Advanced alerting system with configurable thresholds."""

    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: deque[Alert] = deque(maxlen=1000)
        self.notification_handlers: List[Callable[[Alert], None]] = []

    def add_alert_rule(self, rule: "AlertRule") -> None:
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def add_notification_handler(self, handler: Callable[["Alert"], None]) -> None:
        """Add a notification handler."""
        self.notification_handlers.append(handler)

    async def evaluate_alerts(self, metrics: Dict[str, Any]) -> None:
        """Evaluate all alert rules against current metrics."""
        current_time = datetime.now()

        for rule in self.alert_rules:
            try:
                should_fire = rule.evaluate(metrics)
                existing_alert = self._find_active_alert(rule.name)

                if should_fire and not existing_alert:
                    # Fire new alert
                    alert = Alert(
                        name=rule.name,
                        message=rule.get_message(metrics),
                        severity=rule.severity,
                        timestamp=current_time,
                        rule=rule
                    )
                    self.active_alerts.append(alert)
                    self.alert_history.append(alert)

                    # Send notifications
                    for handler in self.notification_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Failed to send alert notification: {e}")

                elif not should_fire and existing_alert:
                    # Resolve alert
                    self.active_alerts.remove(existing_alert)
                    existing_alert.resolved_at = current_time
                    logger.info(f"Alert resolved: {existing_alert.name}")

            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")

    def _find_active_alert(self, alert_name: str) -> Optional["Alert"]:
        """Find an active alert by name."""
        return next((alert for alert in self.active_alerts if alert.name == alert_name), None)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        return {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.alert_rules),
            "recent_alerts": len([a for a in self.alert_history
                                if (datetime.now() - a.timestamp).total_seconds() < 3600]),
            "alert_rate": len(self.alert_history) / max(1, len(self.alert_rules)) if self.alert_rules else 0
        }


@dataclass
class Alert:
    """Represents a fired alert."""
    name: str
    message: str
    severity: str
    timestamp: datetime
    rule: "AlertRule"
    resolved_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.resolved_at is None


@dataclass
class AlertRule:
    """Defines conditions for firing an alert."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    message_template: str
    severity: str = "warning"
    cooldown_seconds: int = 300

    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate if this rule should fire."""
        try:
            return self.condition(metrics)
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")
            return False

    def get_message(self, metrics: Dict[str, Any]) -> str:
        """Generate alert message from template."""
        try:
            return self.message_template.format(**metrics)
        except Exception:
            return f"Alert: {self.name}"


# Global instances
system_monitor = SystemMonitor()
performance_tracker = PerformanceTracker()
alert_manager = AlertManager()

# Default alert rules
default_alert_rules = [
    AlertRule(
        name="high_error_rate",
        condition=lambda m: m.get("error_rate", 0) > 0.1,
        message_template="High error rate detected: {error_rate:.2%}",
        severity="warning"
    ),
    AlertRule(
        name="low_success_rate",
        condition=lambda m: m.get("success_rate", 1.0) < 0.9,
        message_template="Low success rate: {success_rate:.2%}",
        severity="critical"
    ),
    AlertRule(
        name="high_response_time",
        condition=lambda m: m.get("avg_response_time", 0) > 10.0,
        message_template="High response time: {avg_response_time:.2f}s",
        severity="warning"
    ),
    AlertRule(
        name="high_memory_usage",
        condition=lambda m: m.get("memory_percent", 0) > 90,
        message_template="High memory usage: {memory_percent:.1f}%",
        severity="critical"
    )
]

# Register default alert rules
for rule in default_alert_rules:
    alert_manager.add_alert_rule(rule)


async def start_monitoring() -> None:
    """Start all monitoring systems."""
    await system_monitor.start()
    logger.info("Enhanced monitoring started")


async def stop_monitoring() -> None:
    """Stop all monitoring systems."""
    await system_monitor.stop()
    logger.info("Enhanced monitoring stopped")


def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data."""
    return {
        "system_metrics": system_monitor.get_metrics_summary(),
        "performance_stats": performance_tracker.get_all_stats(),
        "alert_summary": alert_manager.get_alert_summary(),
        "active_alerts": [
            {
                "name": alert.name,
                "message": alert.message,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "duration_seconds": (datetime.now() - alert.timestamp).total_seconds()
            }
            for alert in alert_manager.active_alerts
        ]
    }
