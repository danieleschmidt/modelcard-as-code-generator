"""
Metrics collection and monitoring for Model Card Generator.

Provides comprehensive metrics collection including:
- Application performance metrics
- Business metrics (card generation, validation, etc.)
- System resource metrics
- Custom metrics support
- Prometheus-compatible exports
"""

import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import psutil


@dataclass
class MetricValue:
    """Individual metric value with timestamp."""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric values."""
    name: str
    metric_type: str  # counter, gauge, histogram, summary
    help_text: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Thread-safe counter metric."""
    
    def __init__(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or {}
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, amount: Union[int, float] = 1) -> None:
        """Increment the counter."""
        with self._lock:
            self._value += amount
    
    def get(self) -> float:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Gauge:
    """Thread-safe gauge metric."""
    
    def __init__(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or {}
        self._value = 0
        self._lock = threading.Lock()
    
    def set(self, value: Union[int, float]) -> None:
        """Set the gauge value."""
        with self._lock:
            self._value = value
    
    def inc(self, amount: Union[int, float] = 1) -> None:
        """Increment the gauge."""
        with self._lock:
            self._value += amount
    
    def dec(self, amount: Union[int, float] = 1) -> None:
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount
    
    def get(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value


class Histogram:
    """Thread-safe histogram metric."""
    
    def __init__(self, name: str, help_text: str, buckets: Optional[List[float]] = None, labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or {}
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')]
        
        self._count = 0
        self._sum = 0
        self._bucket_counts = {bucket: 0 for bucket in self.buckets}
        self._lock = threading.Lock()
    
    def observe(self, value: Union[int, float]) -> None:
        """Observe a value."""
        with self._lock:
            self._count += 1
            self._sum += value
            
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
    
    def get_count(self) -> int:
        """Get total number of observations."""
        with self._lock:
            return self._count
    
    def get_sum(self) -> float:
        """Get sum of all observed values."""
        with self._lock:
            return self._sum
    
    def get_buckets(self) -> Dict[float, int]:
        """Get bucket counts."""
        with self._lock:
            return self._bucket_counts.copy()


class Summary:
    """Thread-safe summary metric with quantiles."""
    
    def __init__(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None, max_age_seconds: int = 600):
        self.name = name
        self.help_text = help_text
        self.labels = labels or {}
        self.max_age_seconds = max_age_seconds
        
        self._values = deque()
        self._count = 0
        self._sum = 0
        self._lock = threading.Lock()
    
    def observe(self, value: Union[int, float]) -> None:
        """Observe a value."""
        now = time.time()
        
        with self._lock:
            # Remove old values
            while self._values and self._values[0][1] < now - self.max_age_seconds:
                old_value, _ = self._values.popleft()
                self._sum -= old_value
                self._count -= 1
            
            # Add new value
            self._values.append((value, now))
            self._sum += value
            self._count += 1
    
    def get_count(self) -> int:
        """Get count of observations."""
        with self._lock:
            return self._count
    
    def get_sum(self) -> float:
        """Get sum of observations."""
        with self._lock:
            return self._sum
    
    def get_quantile(self, quantile: float) -> float:
        """Get quantile value."""
        with self._lock:
            if not self._values:
                return 0.0
            
            sorted_values = sorted([v[0] for v in self._values])
            index = int(quantile * len(sorted_values))
            
            if index >= len(sorted_values):
                index = len(sorted_values) - 1
            
            return sorted_values[index]


class MetricsCollector:
    """Main metrics collector."""
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._summaries: Dict[str, Summary] = {}
        self._lock = threading.Lock()
        
        # Register default metrics
        self._register_default_metrics()
    
    def _register_default_metrics(self) -> None:
        """Register default application metrics."""
        # Application metrics
        self.counter("mcg_cards_generated_total", "Total number of model cards generated")
        self.counter("mcg_validations_total", "Total number of validations performed")
        self.counter("mcg_errors_total", "Total number of errors", labels={"error_type": ""})
        
        # Performance metrics
        self.histogram("mcg_generation_duration_seconds", "Time spent generating model cards")
        self.histogram("mcg_validation_duration_seconds", "Time spent validating model cards")
        self.histogram("mcg_template_render_duration_seconds", "Time spent rendering templates")
        
        # Resource metrics
        self.gauge("mcg_memory_usage_bytes", "Memory usage in bytes")
        self.gauge("mcg_cpu_usage_percent", "CPU usage percentage")
        self.gauge("mcg_disk_usage_bytes", "Disk usage in bytes")
        
        # Business metrics
        self.counter("mcg_formats_generated_total", "Total cards generated by format", labels={"format": ""})
        self.gauge("mcg_cache_size_bytes", "Cache size in bytes")
        self.gauge("mcg_active_operations", "Number of active operations")
    
    def counter(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None) -> Counter:
        """Get or create a counter metric."""
        key = self._metric_key(name, labels)
        
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(name, help_text, labels)
            return self._counters[key]
    
    def gauge(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Get or create a gauge metric."""
        key = self._metric_key(name, labels)
        
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(name, help_text, labels)
            return self._gauges[key]
    
    def histogram(self, name: str, help_text: str, buckets: Optional[List[float]] = None, labels: Optional[Dict[str, str]] = None) -> Histogram:
        """Get or create a histogram metric."""
        key = self._metric_key(name, labels)
        
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(name, help_text, buckets, labels)
            return self._histograms[key]
    
    def summary(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None, max_age_seconds: int = 600) -> Summary:
        """Get or create a summary metric."""
        key = self._metric_key(name, labels)
        
        with self._lock:
            if key not in self._summaries:
                self._summaries[key] = Summary(name, help_text, labels, max_age_seconds)
            return self._summaries[key]
    
    def _metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_pairs = sorted(labels.items())
        label_str = ",".join(f"{k}={v}" for k, v in label_pairs)
        return f"{name}{{{label_str}}}"
    
    @contextmanager
    def timer(self, histogram_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            hist = self.histogram(histogram_name, f"Duration of {histogram_name}", labels=labels)
            hist.observe(duration)
    
    def time_function(self, histogram_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator for timing function calls."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(histogram_name, labels):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def record_card_generation(self, format_name: str, duration: float, success: bool = True) -> None:
        """Record model card generation metrics."""
        # Increment generation counter
        self.counter("mcg_cards_generated_total").inc()
        
        # Record format-specific counter
        format_counter = self.counter("mcg_formats_generated_total", "Cards generated by format", {"format": format_name})
        format_counter.inc()
        
        # Record duration
        duration_hist = self.histogram("mcg_generation_duration_seconds")
        duration_hist.observe(duration)
        
        # Record errors if not successful
        if not success:
            error_counter = self.counter("mcg_errors_total", "Total errors", {"error_type": "generation"})
            error_counter.inc()
    
    def record_validation(self, validation_type: str, duration: float, success: bool = True) -> None:
        """Record validation metrics."""
        # Increment validation counter
        self.counter("mcg_validations_total").inc()
        
        # Record duration
        duration_hist = self.histogram("mcg_validation_duration_seconds", "Validation duration", {"type": validation_type})
        duration_hist.observe(duration)
        
        # Record errors if not successful
        if not success:
            error_counter = self.counter("mcg_errors_total", "Total errors", {"error_type": "validation"})
            error_counter.inc()
    
    def update_resource_metrics(self) -> None:
        """Update system resource metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.gauge("mcg_memory_usage_bytes").set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.gauge("mcg_cpu_usage_percent").set(cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.gauge("mcg_disk_usage_bytes").set(disk.used)
            
        except Exception:
            # Ignore errors in resource collection
            pass
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a structured format."""
        metrics = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "summaries": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        with self._lock:
            # Counters
            for key, counter in self._counters.items():
                metrics["counters"][key] = {
                    "value": counter.get(),
                    "name": counter.name,
                    "help": counter.help_text,
                    "labels": counter.labels
                }
            
            # Gauges
            for key, gauge in self._gauges.items():
                metrics["gauges"][key] = {
                    "value": gauge.get(),
                    "name": gauge.name,
                    "help": gauge.help_text,
                    "labels": gauge.labels
                }
            
            # Histograms
            for key, histogram in self._histograms.items():
                metrics["histograms"][key] = {
                    "count": histogram.get_count(),
                    "sum": histogram.get_sum(),
                    "buckets": histogram.get_buckets(),
                    "name": histogram.name,
                    "help": histogram.help_text,
                    "labels": histogram.labels
                }
            
            # Summaries
            for key, summary in self._summaries.items():
                metrics["summaries"][key] = {
                    "count": summary.get_count(),
                    "sum": summary.get_sum(),
                    "quantiles": {
                        "0.5": summary.get_quantile(0.5),
                        "0.9": summary.get_quantile(0.9),
                        "0.95": summary.get_quantile(0.95),
                        "0.99": summary.get_quantile(0.99)
                    },
                    "name": summary.name,
                    "help": summary.help_text,
                    "labels": summary.labels
                }
        
        return metrics
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        with self._lock:
            # Counters
            for key, counter in self._counters.items():
                lines.append(f"# HELP {counter.name} {counter.help_text}")
                lines.append(f"# TYPE {counter.name} counter")
                
                labels_str = ""
                if counter.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in counter.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{counter.name}{labels_str} {counter.get()}")
                lines.append("")
            
            # Gauges
            for key, gauge in self._gauges.items():
                lines.append(f"# HELP {gauge.name} {gauge.help_text}")
                lines.append(f"# TYPE {gauge.name} gauge")
                
                labels_str = ""
                if gauge.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in gauge.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{gauge.name}{labels_str} {gauge.get()}")
                lines.append("")
            
            # Histograms
            for key, histogram in self._histograms.items():
                lines.append(f"# HELP {histogram.name} {histogram.help_text}")
                lines.append(f"# TYPE {histogram.name} histogram")
                
                base_labels = histogram.labels.copy() if histogram.labels else {}
                
                # Bucket counts
                for bucket, count in histogram.get_buckets().items():
                    bucket_labels = base_labels.copy()
                    bucket_labels["le"] = str(bucket if bucket != float('inf') else '+Inf')
                    
                    label_pairs = [f'{k}="{v}"' for k, v in bucket_labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                    
                    lines.append(f"{histogram.name}_bucket{labels_str} {count}")
                
                # Count and sum
                base_labels_str = ""
                if base_labels:
                    label_pairs = [f'{k}="{v}"' for k, v in base_labels.items()]
                    base_labels_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{histogram.name}_count{base_labels_str} {histogram.get_count()}")
                lines.append(f"{histogram.name}_sum{base_labels_str} {histogram.get_sum()}")
                lines.append("")
        
        return "\n".join(lines)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def create_metrics_app():
    """Create a simple WSGI app for serving metrics."""
    def metrics_app(environ, start_response):
        """WSGI app for Prometheus metrics endpoint."""
        if environ['PATH_INFO'] == '/metrics':
            collector = get_metrics_collector()
            
            # Update resource metrics before serving
            collector.update_resource_metrics()
            
            metrics_text = collector.export_prometheus()
            
            status = '200 OK'
            headers = [
                ('Content-Type', 'text/plain; version=0.0.4; charset=utf-8'),
                ('Content-Length', str(len(metrics_text)))
            ]
            
            start_response(status, headers)
            return [metrics_text.encode('utf-8')]
        
        elif environ['PATH_INFO'] == '/metrics/json':
            collector = get_metrics_collector()
            
            # Update resource metrics before serving
            collector.update_resource_metrics()
            
            import json
            metrics_json = json.dumps(collector.get_all_metrics(), indent=2)
            
            status = '200 OK'
            headers = [
                ('Content-Type', 'application/json'),
                ('Content-Length', str(len(metrics_json)))
            ]
            
            start_response(status, headers)
            return [metrics_json.encode('utf-8')]
        
        else:
            status = '404 Not Found'
            headers = [('Content-Type', 'text/plain')]
            start_response(status, headers)
            return [b'Not Found']
    
    return metrics_app


# Convenience decorators
def track_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            with collector.timer(metric_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            counter = collector.counter(metric_name, f"Number of calls to {func.__name__}", labels)
            counter.inc()
            return func(*args, **kwargs)
        return wrapper
    return decorator