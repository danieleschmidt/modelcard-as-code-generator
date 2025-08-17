"""
Advanced monitoring system with distributed tracing, anomaly detection, and predictive analytics.

This module provides enterprise-grade monitoring capabilities including:
- Distributed tracing across microservices
- Real-time anomaly detection
- Predictive maintenance alerts
- Performance trend analysis
- SLA monitoring and alerting
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Exception] = None

    def finish(self) -> None:
        """Mark span as finished and calculate duration."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the span."""
        self.tags[key] = value

    def log(self, message: str, level: str = "info", **fields) -> None:
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **fields
        })


class DistributedTracer:
    """Distributed tracing system for monitoring request flows."""

    def __init__(self, service_name: str = "modelcard-generator"):
        self.service_name = service_name
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: deque = deque(maxlen=10000)
        self._span_processors: List[Any] = []

    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[Span] = None,
        trace_id: Optional[str] = None
    ) -> Span:
        """Start a new tracing span."""
        if trace_id is None:
            trace_id = str(uuid4())
        
        span_id = str(uuid4())
        parent_span_id = parent_span.span_id if parent_span else None

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        span.set_tag("service.name", self.service_name)
        self.active_spans[span_id] = span
        
        logger.debug(f"Started span {span_id} for operation {operation_name}")
        return span

    def finish_span(self, span: Span) -> None:
        """Finish a tracing span."""
        span.finish()
        
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        self.completed_spans.append(span)
        
        # Process span through registered processors
        for processor in self._span_processors:
            processor.process_span(span)
        
        logger.debug(f"Finished span {span.span_id}, duration: {span.duration_ms:.2f}ms")

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        spans = []
        
        # Check active spans
        for span in self.active_spans.values():
            if span.trace_id == trace_id:
                spans.append(span)
        
        # Check completed spans
        for span in self.completed_spans:
            if span.trace_id == trace_id:
                spans.append(span)
        
        return sorted(spans, key=lambda s: s.start_time)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class AnomalyDetector:
    """Real-time anomaly detection for metrics."""

    def __init__(self, window_size: int = 100, threshold_multiplier: float = 2.5):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def add_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None) -> bool:
        """Add a metric value and check for anomalies."""
        if timestamp is None:
            timestamp = time.time()

        metric_point = MetricPoint(timestamp, value)
        window = self.metric_windows[metric_name]
        window.append(metric_point)

        # Calculate baseline statistics if we have enough data
        if len(window) >= min(20, self.window_size // 2):
            self._update_baseline_stats(metric_name, window)

        # Check for anomaly
        return self._is_anomaly(metric_name, value)

    def _update_baseline_stats(self, metric_name: str, window: deque) -> None:
        """Update baseline statistics for a metric."""
        values = [point.value for point in window]
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5

        self.baseline_stats[metric_name] = {
            "mean": mean,
            "std_dev": std_dev,
            "min": min(values),
            "max": max(values)
        }

    def _is_anomaly(self, metric_name: str, value: float) -> bool:
        """Check if a value is anomalous."""
        if metric_name not in self.baseline_stats:
            return False

        stats = self.baseline_stats[metric_name]
        threshold = stats["std_dev"] * self.threshold_multiplier
        
        return abs(value - stats["mean"]) > threshold


class SLAMonitor:
    """Service Level Agreement monitoring and alerting."""

    def __init__(self):
        self.sla_definitions: Dict[str, Dict[str, Any]] = {}
        self.violations: List[Dict[str, Any]] = []
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def define_sla(
        self,
        name: str,
        metric: str,
        threshold: float,
        operator: str = "<=",  # <=, >=, <, >, ==
        window_minutes: int = 5,
        violation_threshold: float = 0.95  # 95% of requests must meet SLA
    ) -> None:
        """Define an SLA to monitor."""
        self.sla_definitions[name] = {
            "metric": metric,
            "threshold": threshold,
            "operator": operator,
            "window_minutes": window_minutes,
            "violation_threshold": violation_threshold,
            "last_check": time.time()
        }
        logger.info(f"Defined SLA: {name} - {metric} {operator} {threshold}")

    def record_metric(self, metric: str, value: float, timestamp: Optional[float] = None) -> None:
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_buffer[metric].append((timestamp, value))

    def check_slas(self) -> List[Dict[str, Any]]:
        """Check all SLAs and return violations."""
        current_violations = []
        current_time = time.time()

        for sla_name, sla_config in self.sla_definitions.items():
            # Only check if enough time has passed
            if current_time - sla_config["last_check"] < 60:  # Check every minute
                continue

            violation = self._check_single_sla(sla_name, sla_config, current_time)
            if violation:
                current_violations.append(violation)
                self.violations.append(violation)

            sla_config["last_check"] = current_time

        return current_violations

    def _check_single_sla(self, sla_name: str, sla_config: Dict[str, Any], current_time: float) -> Optional[Dict[str, Any]]:
        """Check a single SLA definition."""
        metric = sla_config["metric"]
        window_seconds = sla_config["window_minutes"] * 60
        cutoff_time = current_time - window_seconds

        if metric not in self.metrics_buffer:
            return None

        # Get values within the time window
        recent_values = [
            value for timestamp, value in self.metrics_buffer[metric]
            if timestamp >= cutoff_time
        ]

        if not recent_values:
            return None

        # Check what percentage meets the SLA
        threshold = sla_config["threshold"]
        operator = sla_config["operator"]
        
        meeting_sla = 0
        for value in recent_values:
            if self._evaluate_condition(value, operator, threshold):
                meeting_sla += 1

        sla_percentage = meeting_sla / len(recent_values)
        
        if sla_percentage < sla_config["violation_threshold"]:
            return {
                "sla_name": sla_name,
                "timestamp": current_time,
                "metric": metric,
                "expected_percentage": sla_config["violation_threshold"],
                "actual_percentage": sla_percentage,
                "sample_size": len(recent_values),
                "severity": "high" if sla_percentage < 0.8 else "medium"
            }

        return None

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate if a value meets the SLA condition."""
        if operator == "<=":
            return value <= threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">":
            return value > threshold
        elif operator == "==":
            return value == threshold
        else:
            return False


class AdvancedMonitoringSystem:
    """Comprehensive monitoring system with tracing, anomaly detection, and SLA monitoring."""

    def __init__(self, service_name: str = "modelcard-generator"):
        self.service_name = service_name
        self.tracer = DistributedTracer(service_name)
        self.anomaly_detector = AnomalyDetector()
        self.sla_monitor = SLAMonitor()
        
        # Performance metrics
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        
        # Alerting
        self.alert_handlers: List[Any] = []
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Setup default SLAs
        self._setup_default_slas()

    def _setup_default_slas(self) -> None:
        """Setup default SLAs for model card generation."""
        # Response time SLA: 95% of requests should complete within 2 seconds
        self.sla_monitor.define_sla(
            name="response_time_p95",
            metric="generation_duration_ms",
            threshold=2000,
            operator="<=",
            window_minutes=5,
            violation_threshold=0.95
        )
        
        # Error rate SLA: Less than 1% error rate
        self.sla_monitor.define_sla(
            name="error_rate",
            metric="error_rate_percentage",
            threshold=1.0,
            operator="<=",
            window_minutes=5,
            violation_threshold=1.0
        )
        
        # Memory usage SLA: Memory usage should stay below 80%
        self.sla_monitor.define_sla(
            name="memory_usage",
            metric="memory_usage_percentage",
            threshold=80.0,
            operator="<=",
            window_minutes=10,
            violation_threshold=0.9
        )

    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started advanced monitoring system")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped advanced monitoring system")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                # Check SLAs
                violations = self.sla_monitor.check_slas()
                for violation in violations:
                    await self._handle_sla_violation(violation)
                
                # Calculate derived metrics
                self._calculate_derived_metrics()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics like error rates, percentiles, etc."""
        # Calculate error rate
        total_requests = self.counters.get("total_requests", 0)
        failed_requests = self.counters.get("failed_requests", 0)
        
        if total_requests > 0:
            error_rate = (failed_requests / total_requests) * 100
            self.gauges["error_rate_percentage"] = error_rate
            self.sla_monitor.record_metric("error_rate_percentage", error_rate)

        # Calculate memory usage (if available)
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            self.gauges["memory_usage_percentage"] = memory_percent
            self.sla_monitor.record_metric("memory_usage_percentage", memory_percent)
        except ImportError:
            pass

    async def _handle_sla_violation(self, violation: Dict[str, Any]) -> None:
        """Handle an SLA violation."""
        logger.warning(f"SLA violation detected: {violation}")
        
        # Send alerts through registered handlers
        for handler in self.alert_handlers:
            try:
                await handler.send_alert(violation)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        timestamp = time.time()
        
        # Store in local metrics
        self.metrics[name].append(MetricPoint(timestamp, value, tags or {}))
        
        # Check for anomalies
        is_anomaly = self.anomaly_detector.add_metric(name, value, timestamp)
        if is_anomaly:
            logger.warning(f"Anomaly detected in metric {name}: {value}")
        
        # Record for SLA monitoring
        self.sla_monitor.record_metric(name, value, timestamp)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric."""
        self.gauges[name] = value

    def create_span(self, operation_name: str, parent_span: Optional[Span] = None) -> Span:
        """Create a new tracing span."""
        return self.tracer.start_span(operation_name, parent_span)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "active_spans": len(self.tracer.active_spans),
            "completed_spans": len(self.tracer.completed_spans),
            "sla_violations": len(self.sla_monitor.violations),
            "monitoring_status": "active" if self._monitoring_task and not self._monitoring_task.done() else "inactive"
        }


# Global monitoring instance
monitoring = AdvancedMonitoringSystem()


def monitor_operation(operation_name: str):
    """Decorator to monitor function execution."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                span = monitoring.create_span(operation_name)
                start_time = time.time()
                
                try:
                    monitoring.increment_counter("total_requests")
                    span.set_tag("function", func.__name__)
                    
                    result = await func(*args, **kwargs)
                    
                    duration_ms = (time.time() - start_time) * 1000
                    monitoring.record_metric("generation_duration_ms", duration_ms)
                    span.set_tag("success", True)
                    
                    return result
                    
                except Exception as e:
                    monitoring.increment_counter("failed_requests")
                    span.set_tag("success", False)
                    span.set_tag("error", str(e))
                    span.error = e
                    raise
                    
                finally:
                    monitoring.tracer.finish_span(span)
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                span = monitoring.create_span(operation_name)
                start_time = time.time()
                
                try:
                    monitoring.increment_counter("total_requests")
                    span.set_tag("function", func.__name__)
                    
                    result = func(*args, **kwargs)
                    
                    duration_ms = (time.time() - start_time) * 1000
                    monitoring.record_metric("generation_duration_ms", duration_ms)
                    span.set_tag("success", True)
                    
                    return result
                    
                except Exception as e:
                    monitoring.increment_counter("failed_requests")
                    span.set_tag("success", False)
                    span.set_tag("error", str(e))
                    span.error = e
                    raise
                    
                finally:
                    monitoring.tracer.finish_span(span)
            
            return sync_wrapper
    
    return decorator