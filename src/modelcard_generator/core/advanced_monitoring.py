"""Advanced monitoring and observability for production model card generation."""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .logging_config import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Advanced metrics collection with statistical analysis."""
    
    def __init__(self, window_size: int = 1000, retention_hours: int = 24):
        self.window_size = window_size
        self.retention_hours = retention_hours
        
        # Metric storage
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.timeseries: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Statistical caches
        self.stats_cache: Dict[str, Dict[str, float]] = {}
        self.cache_timestamp: Dict[str, datetime] = {}
        
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        key = self._build_key(name, tags)
        self.counters[key] += value
        logger.debug(f"Counter {key}: {self.counters[key]}")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        key = self._build_key(name, tags)
        self.gauges[key] = value
        logger.debug(f"Gauge {key}: {value}")
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        key = self._build_key(name, tags)
        self.histograms[key].append(value)
        logger.debug(f"Histogram {key}: {value}")
    
    def record_timeseries(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a time series value."""
        key = self._build_key(name, tags)
        timestamp = datetime.now()
        self.timeseries[key].append((timestamp, value))
        logger.debug(f"Timeseries {key}: {value} at {timestamp}")
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get statistical summary of histogram data."""
        key = self._build_key(name, tags)
        
        # Check cache
        if key in self.stats_cache:
            cache_time = self.cache_timestamp.get(key, datetime.min)
            if datetime.now() - cache_time < timedelta(seconds=30):
                return self.stats_cache[key]
        
        # Calculate statistics
        values = list(self.histograms[key])
        if not values:
            return {"count": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        
        values.sort()
        count = len(values)
        
        stats = {
            "count": count,
            "mean": sum(values) / count,
            "median": values[count // 2],
            "p95": values[int(count * 0.95)] if count > 0 else 0,
            "p99": values[int(count * 0.99)] if count > 0 else 0,
            "min": min(values),
            "max": max(values)
        }
        
        # Cache results
        self.stats_cache[key] = stats
        self.cache_timestamp[key] = datetime.now()
        
        return stats
    
    def get_rate(self, counter_name: str, window_seconds: int = 60, tags: Optional[Dict[str, str]] = None) -> float:
        """Calculate rate per second for a counter."""
        key = self._build_key(counter_name, tags)
        current_count = self.counters.get(key, 0)
        
        # For simplification, assume constant rate over window
        # In production, this would track actual timestamps
        return current_count / window_seconds
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histogram_stats": {key: self.get_histogram_stats(key.split(":")[0], 
                                                             self._parse_tags(key)) 
                               for key in self.histograms.keys()},
            "timestamp": datetime.now().isoformat()
        }
    
    def _build_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Build metric key with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"
    
    def _parse_tags(self, key: str) -> Optional[Dict[str, str]]:
        """Parse tags from metric key."""
        if ":" not in key:
            return None
        tag_str = key.split(":", 1)[1]
        tags = {}
        for tag in tag_str.split(","):
            if "=" in tag:
                k, v = tag.split("=", 1)
                tags[k] = v
        return tags


class PerformanceMonitor:
    """Monitor performance metrics and detect anomalies."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def set_threshold(self, metric_name: str, threshold_type: str, value: float) -> None:
        """Set performance threshold for monitoring."""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        self.thresholds[metric_name][threshold_type] = value
        logger.info(f"Set threshold {metric_name}.{threshold_type} = {value}")
    
    def establish_baseline(self, metric_name: str, duration_minutes: int = 60) -> None:
        """Establish performance baseline for anomaly detection."""
        stats = self.metrics.get_histogram_stats(metric_name)
        self.baseline_stats[metric_name] = {
            "mean": stats["mean"],
            "std": self._calculate_std(metric_name),
            "p95": stats["p95"],
            "established_at": datetime.now().timestamp()
        }
        logger.info(f"Established baseline for {metric_name}: {self.baseline_stats[metric_name]}")
    
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """Check all metrics against defined thresholds."""
        violations = []
        
        for metric_name, thresholds in self.thresholds.items():
            stats = self.metrics.get_histogram_stats(metric_name)
            
            for threshold_type, threshold_value in thresholds.items():
                current_value = stats.get(threshold_type, 0)
                
                if self._is_threshold_violated(threshold_type, current_value, threshold_value):
                    violation = {
                        "metric": metric_name,
                        "threshold_type": threshold_type,
                        "current_value": current_value,
                        "threshold_value": threshold_value,
                        "severity": self._get_severity(threshold_type, current_value, threshold_value),
                        "timestamp": datetime.now().isoformat()
                    }
                    violations.append(violation)
                    logger.warning(f"Threshold violation: {violation}")
        
        return violations
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in metrics."""
        anomalies = []
        
        for metric_name, baseline in self.baseline_stats.items():
            stats = self.metrics.get_histogram_stats(metric_name)
            current_mean = stats["mean"]
            
            # Z-score based anomaly detection
            z_score = abs(current_mean - baseline["mean"]) / max(baseline["std"], 0.001)
            
            if z_score > 2.5:  # 2.5 standard deviations
                anomaly = {
                    "metric": metric_name,
                    "type": "statistical_anomaly",
                    "z_score": z_score,
                    "current_value": current_mean,
                    "baseline_mean": baseline["mean"],
                    "baseline_std": baseline["std"],
                    "severity": "high" if z_score > 3.0 else "medium",
                    "timestamp": datetime.now().isoformat()
                }
                anomalies.append(anomaly)
                logger.warning(f"Anomaly detected: {anomaly}")
        
        return anomalies
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "summary": {
                "total_metrics": len(self.metrics.histograms),
                "active_thresholds": len(self.thresholds),
                "baseline_metrics": len(self.baseline_stats),
                "recent_alerts": len([a for a in self.alerts if self._is_recent(a, minutes=60)])
            },
            "threshold_violations": self.check_thresholds(),
            "anomalies": self.detect_anomalies(),
            "key_metrics": self._get_key_metrics(),
            "health_score": self._calculate_health_score(),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_std(self, metric_name: str) -> float:
        """Calculate standard deviation for metric."""
        values = list(self.metrics.histograms[metric_name])
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _is_threshold_violated(self, threshold_type: str, current: float, threshold: float) -> bool:
        """Check if threshold is violated."""
        if threshold_type in ["max", "p95", "p99"]:
            return current > threshold
        elif threshold_type in ["min"]:
            return current < threshold
        return False
    
    def _get_severity(self, threshold_type: str, current: float, threshold: float) -> str:
        """Determine severity of threshold violation."""
        if threshold_type in ["max", "p95", "p99"]:
            ratio = current / threshold
            if ratio > 2.0:
                return "critical"
            elif ratio > 1.5:
                return "high"
            else:
                return "medium"
        return "medium"
    
    def _is_recent(self, alert: Dict[str, Any], minutes: int) -> bool:
        """Check if alert is recent."""
        alert_time = datetime.fromisoformat(alert["timestamp"].replace("Z", "+00:00"))
        return datetime.now() - alert_time < timedelta(minutes=minutes)
    
    def _get_key_metrics(self) -> Dict[str, Any]:
        """Get summary of key performance metrics."""
        key_metrics = {}
        
        important_metrics = ["response_time", "throughput", "error_rate", "cpu_usage", "memory_usage"]
        
        for metric in important_metrics:
            if metric in self.metrics.histograms:
                key_metrics[metric] = self.metrics.get_histogram_stats(metric)
        
        return key_metrics
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Deduct points for threshold violations
        violations = self.check_thresholds()
        for violation in violations:
            if violation["severity"] == "critical":
                score -= 20
            elif violation["severity"] == "high":
                score -= 10
            else:
                score -= 5
        
        # Deduct points for anomalies
        anomalies = self.detect_anomalies()
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                score -= 15
            else:
                score -= 8
        
        return max(0.0, min(100.0, score))


class AlertManager:
    """Manage alerts and notifications with intelligent routing."""
    
    def __init__(self):
        self.alert_channels: Dict[str, List[Callable]] = defaultdict(list)
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}
        self.escalation_policies: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
    def register_channel(self, severity: str, handler: Callable) -> None:
        """Register alert channel for specific severity."""
        self.alert_channels[severity].append(handler)
        logger.info(f"Registered alert channel for {severity} severity")
    
    def set_suppression_rule(self, alert_type: str, window_minutes: int, max_alerts: int) -> None:
        """Set alert suppression rule to prevent spam."""
        self.suppression_rules[alert_type] = {
            "window_minutes": window_minutes,
            "max_alerts": max_alerts,
            "alerts_sent": [],
            "suppressed_count": 0
        }
        logger.info(f"Set suppression rule for {alert_type}: {max_alerts} alerts per {window_minutes} minutes")
    
    def set_escalation_policy(self, alert_type: str, escalation_minutes: int, escalation_severity: str) -> None:
        """Set escalation policy for unresolved alerts."""
        self.escalation_policies[alert_type] = {
            "escalation_minutes": escalation_minutes,
            "escalation_severity": escalation_severity
        }
        logger.info(f"Set escalation policy for {alert_type}: escalate to {escalation_severity} after {escalation_minutes} minutes")
    
    async def send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send alert with intelligent routing and suppression."""
        
        # Check suppression rules
        if not self._should_send_alert(alert_type):
            logger.debug(f"Alert {alert_type} suppressed due to rate limiting")
            return False
        
        alert_id = f"{alert_type}_{int(time.time())}"
        alert = {
            "id": alert_id,
            "type": alert_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Store active alert
        self.active_alerts[alert_id] = alert
        
        # Send to appropriate channels
        channels = self.alert_channels.get(severity, []) + self.alert_channels.get("all", [])
        
        success = True
        for channel in channels:
            try:
                await channel(alert) if asyncio.iscoroutinefunction(channel) else channel(alert)
                logger.info(f"Alert {alert_id} sent via {channel.__name__}")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.__name__}: {e}")
                success = False
        
        # Update suppression tracking
        self._track_alert_sent(alert_type)
        
        # Schedule escalation if configured
        if alert_type in self.escalation_policies:
            asyncio.create_task(self._schedule_escalation(alert_id, alert_type))
        
        return success
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["status"] = "resolved"
            self.active_alerts[alert_id]["resolved_at"] = datetime.now().isoformat()
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() if alert["status"] == "active"]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and health metrics."""
        active_alerts = self.get_active_alerts()
        
        stats = {
            "active_alerts": len(active_alerts),
            "total_alerts": len(self.active_alerts),
            "alerts_by_severity": defaultdict(int),
            "alerts_by_type": defaultdict(int),
            "suppressed_alerts": sum(rule.get("suppressed_count", 0) for rule in self.suppression_rules.values()),
            "escalated_alerts": len([a for a in self.active_alerts.values() if a.get("escalated", False)])
        }
        
        for alert in self.active_alerts.values():
            stats["alerts_by_severity"][alert["severity"]] += 1
            stats["alerts_by_type"][alert["type"]] += 1
        
        return dict(stats)
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent based on suppression rules."""
        if alert_type not in self.suppression_rules:
            return True
        
        rule = self.suppression_rules[alert_type]
        window_start = datetime.now() - timedelta(minutes=rule["window_minutes"])
        
        # Count recent alerts
        recent_alerts = [
            timestamp for timestamp in rule["alerts_sent"]
            if timestamp > window_start
        ]
        
        if len(recent_alerts) >= rule["max_alerts"]:
            rule["suppressed_count"] += 1
            return False
        
        return True
    
    def _track_alert_sent(self, alert_type: str) -> None:
        """Track that an alert was sent for suppression tracking."""
        if alert_type in self.suppression_rules:
            rule = self.suppression_rules[alert_type]
            rule["alerts_sent"].append(datetime.now())
            
            # Clean old entries
            window_start = datetime.now() - timedelta(minutes=rule["window_minutes"])
            rule["alerts_sent"] = [t for t in rule["alerts_sent"] if t > window_start]
    
    async def _schedule_escalation(self, alert_id: str, alert_type: str) -> None:
        """Schedule alert escalation if not resolved."""
        policy = self.escalation_policies[alert_type]
        escalation_delay = policy["escalation_minutes"] * 60
        
        await asyncio.sleep(escalation_delay)
        
        # Check if alert is still active
        if alert_id in self.active_alerts and self.active_alerts[alert_id]["status"] == "active":
            alert = self.active_alerts[alert_id]
            alert["escalated"] = True
            
            escalated_message = f"ESCALATED: {alert['message']} (unresolved for {policy['escalation_minutes']} minutes)"
            
            await self.send_alert(
                f"escalated_{alert_type}",
                policy["escalation_severity"],
                escalated_message,
                alert["details"]
            )


# Global monitoring instances
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor(metrics_collector)
alert_manager = AlertManager()


# Convenient monitoring decorators
def monitor_performance(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_histogram(f"{metric_name}_duration", duration * 1000, tags)  # ms
                metrics_collector.increment_counter(f"{metric_name}_calls", 1, tags)
                return result
            except Exception as e:
                metrics_collector.increment_counter(f"{metric_name}_errors", 1, tags)
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_histogram(f"{metric_name}_duration", duration * 1000, tags)  # ms
                metrics_collector.increment_counter(f"{metric_name}_calls", 1, tags)
                return result
            except Exception as e:
                metrics_collector.increment_counter(f"{metric_name}_errors", 1, tags)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator