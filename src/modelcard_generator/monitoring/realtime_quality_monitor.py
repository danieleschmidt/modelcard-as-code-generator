"""
Real-time Quality Monitoring System
Continuous monitoring of model card generation quality with adaptive alerting.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque, defaultdict
from enum import Enum
import statistics
import numpy as np

from ..core.models import ModelCard
from ..core.progressive_quality_gates import QualityGateResult, ProgressiveQualityGateEngine
from ..core.autonomous_learning_system import AutonomousLearningSystem, AdaptiveRecommendation


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"


class MetricTrend(Enum):
    """Metric trend directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class QualityMetricSnapshot:
    """Snapshot of quality metrics at a point in time."""
    timestamp: datetime
    overall_quality: float
    gate_scores: Dict[str, float]
    processing_time: float
    error_count: int
    user_satisfaction: Optional[float] = None
    generation_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAlert:
    """Quality alert with context and recommendations."""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    trend: MetricTrend
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)
    auto_remediation: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)


@dataclass
class QualityDashboard:
    """Real-time quality dashboard data."""
    current_metrics: Dict[str, float]
    trends: Dict[str, MetricTrend]
    alerts: List[QualityAlert]
    performance_summary: Dict[str, Any]
    recommendations: List[AdaptiveRecommendation]
    system_health: str
    uptime: float
    last_updated: datetime = field(default_factory=datetime.now)


class RealTimeQualityMonitor:
    """
    Advanced real-time quality monitoring system with adaptive alerting,
    trend analysis, and automatic remediation capabilities.
    """
    
    def __init__(
        self,
        monitor_dir: Optional[Path] = None,
        quality_gates: Optional[ProgressiveQualityGateEngine] = None,
        learning_system: Optional[AutonomousLearningSystem] = None
    ):
        self.monitor_dir = monitor_dir or Path.home() / ".modelcard_generator" / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.quality_gates = quality_gates or ProgressiveQualityGateEngine()
        self.learning_system = learning_system or AutonomousLearningSystem()
        
        # Monitoring state
        self.is_monitoring = False
        self.start_time = datetime.now()
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Data storage
        self.metric_history: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=100)
        self.performance_buffer: deque = deque(maxlen=50)
        
        # Alert subscribers
        self.alert_subscribers: List[Callable[[QualityAlert], None]] = []
        
        # Adaptive thresholds
        self.adaptive_thresholds: Dict[str, float] = {
            "overall_quality": 0.8,
            "processing_time": 2.0,
            "error_rate": 0.05,
            "user_satisfaction": 0.7
        }
        
        # Trend analysis
        self.trend_analyzers: Dict[str, 'TrendAnalyzer'] = {}
        self._initialize_trend_analyzers()
        
        # Auto-remediation
        self.remediation_strategies: Dict[str, Callable] = {}
        self._initialize_remediation_strategies()
        
        # Load historical data
        self._load_monitoring_state()
    
    def start_monitoring(self) -> None:
        """Start real-time quality monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.start_time = datetime.now()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Real-time quality monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time quality monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self._save_monitoring_state()
        self.logger.info("Real-time quality monitoring stopped")
    
    async def record_generation(
        self,
        model_card: ModelCard,
        quality_results: List[QualityGateResult],
        generation_time: float,
        context: Dict[str, Any] = None,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a model card generation for quality monitoring."""
        
        context = context or {}
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(quality_results)
        
        # Count errors
        error_count = sum(1 for r in quality_results if not r.passed)
        
        # Extract user satisfaction if provided
        user_satisfaction = None
        if user_feedback:
            user_satisfaction = user_feedback.get("satisfaction_score")
        
        # Create quality snapshot
        snapshot = QualityMetricSnapshot(
            timestamp=datetime.now(),
            overall_quality=overall_quality,
            gate_scores={r.gate_name: r.score for r in quality_results},
            processing_time=generation_time,
            error_count=error_count,
            user_satisfaction=user_satisfaction,
            generation_context=context
        )
        
        # Store in history
        self.metric_history.append(snapshot)
        self.performance_buffer.append(generation_time)
        
        # Analyze trends and generate alerts
        await self._analyze_quality_trends(snapshot)
        
        # Learn from this generation
        if self.learning_system:
            recommendations = await self.learning_system.learn_from_generation(
                model_card=model_card,
                generation_context=context,
                quality_results=quality_results,
                user_feedback=user_feedback
            )
        
        self.logger.debug(f"Recorded generation with quality {overall_quality:.3f}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        
        while self.is_monitoring:
            try:
                # Periodic analysis every 30 seconds
                self._periodic_analysis()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Save state every 5 minutes
                if int(time.time()) % 300 == 0:
                    self._save_monitoring_state()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _periodic_analysis(self) -> None:
        """Perform periodic analysis of quality metrics."""
        
        if len(self.metric_history) < 10:
            return  # Not enough data
        
        # Analyze recent performance
        recent_snapshots = list(self.metric_history)[-10:]
        
        # Check for quality degradation
        recent_qualities = [s.overall_quality for s in recent_snapshots]
        quality_trend = self._calculate_trend(recent_qualities)
        
        if quality_trend == MetricTrend.DEGRADING:
            avg_quality = statistics.mean(recent_qualities)
            if avg_quality < self.adaptive_thresholds["overall_quality"]:
                alert = self._create_alert(
                    level=AlertLevel.WARNING,
                    title="Quality Degradation Detected",
                    description=f"Average quality dropped to {avg_quality:.3f}",
                    metric_name="overall_quality",
                    current_value=avg_quality,
                    threshold_value=self.adaptive_thresholds["overall_quality"]
                )
                self._emit_alert(alert)
        
        # Check processing time
        if self.performance_buffer:
            avg_processing_time = statistics.mean(self.performance_buffer)
            if avg_processing_time > self.adaptive_thresholds["processing_time"]:
                alert = self._create_alert(
                    level=AlertLevel.INFO,
                    title="Performance Degradation",
                    description=f"Average processing time: {avg_processing_time:.2f}s",
                    metric_name="processing_time",
                    current_value=avg_processing_time,
                    threshold_value=self.adaptive_thresholds["processing_time"]
                )
                alert.recommendations = [
                    "Consider scaling computational resources",
                    "Review caching effectiveness",
                    "Analyze generation complexity"
                ]
                self._emit_alert(alert)
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds()
    
    async def _analyze_quality_trends(self, snapshot: QualityMetricSnapshot) -> None:
        """Analyze quality trends and generate alerts if needed."""
        
        if len(self.metric_history) < 5:
            return  # Need more data for trend analysis
        
        # Analyze each gate's performance
        for gate_name, score in snapshot.gate_scores.items():
            if gate_name not in self.trend_analyzers:
                continue
                
            analyzer = self.trend_analyzers[gate_name]
            trend = analyzer.analyze_trend(score)
            
            # Generate alerts for significant changes
            if trend == MetricTrend.DEGRADING and score < 0.7:
                alert = self._create_alert(
                    level=AlertLevel.WARNING,
                    title=f"{gate_name.title()} Quality Declining",
                    description=f"{gate_name} score dropped to {score:.3f}",
                    metric_name=gate_name,
                    current_value=score,
                    threshold_value=0.7,
                    trend=trend
                )
                alert.recommendations = await self._get_improvement_recommendations(gate_name, score)
                self._emit_alert(alert)
            
            elif trend == MetricTrend.VOLATILE:
                alert = self._create_alert(
                    level=AlertLevel.INFO,
                    title=f"{gate_name.title()} Performance Volatile",
                    description=f"{gate_name} showing unstable performance",
                    metric_name=gate_name,
                    current_value=score,
                    threshold_value=0.8,
                    trend=trend
                )
                self._emit_alert(alert)
    
    def _calculate_overall_quality(self, quality_results: List[QualityGateResult]) -> float:
        """Calculate overall quality score."""
        if not quality_results:
            return 0.0
        
        # Weight by importance
        gate_weights = {
            "completeness": 2.0,
            "accuracy": 2.5,
            "security": 3.0,
            "performance": 1.5,
            "compliance": 2.0,
            "maintainability": 1.0
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in quality_results:
            weight = gate_weights.get(result.gate_name, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_trend(self, values: List[float]) -> MetricTrend:
        """Calculate trend from a series of values."""
        if len(values) < 3:
            return MetricTrend.STABLE
        
        # Calculate linear trend
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        # Calculate volatility
        volatility = np.std(values) / (np.mean(values) + 1e-6)
        
        # Determine trend
        if volatility > 0.2:  # High volatility
            return MetricTrend.VOLATILE
        elif slope > 0.01:  # Improving
            return MetricTrend.IMPROVING
        elif slope < -0.01:  # Degrading
            return MetricTrend.DEGRADING
        else:
            return MetricTrend.STABLE
    
    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        description: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        trend: MetricTrend = MetricTrend.STABLE
    ) -> QualityAlert:
        """Create a quality alert."""
        
        alert_id = f"{metric_name}_{int(time.time())}"
        
        alert = QualityAlert(
            alert_id=alert_id,
            level=level,
            title=title,
            description=description,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            trend=trend
        )
        
        # Add auto-remediation if available
        if metric_name in self.remediation_strategies:
            alert.auto_remediation = f"Auto-remediation available for {metric_name}"
        
        return alert
    
    def _emit_alert(self, alert: QualityAlert) -> None:
        """Emit an alert to all subscribers."""
        
        # Store in history
        self.alert_history.append(alert)
        
        # Log the alert
        level_map = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.CRITICAL: self.logger.error,
            AlertLevel.URGENT: self.logger.critical
        }
        
        log_func = level_map.get(alert.level, self.logger.info)
        log_func(f"Quality Alert: {alert.title} - {alert.description}")
        
        # Notify subscribers
        for subscriber in self.alert_subscribers:
            try:
                subscriber(alert)
            except Exception as e:
                self.logger.error(f"Error notifying alert subscriber: {e}")
        
        # Trigger auto-remediation if available and appropriate
        if (alert.level in [AlertLevel.CRITICAL, AlertLevel.URGENT] and 
            alert.metric_name in self.remediation_strategies):
            self._trigger_auto_remediation(alert)
    
    def _trigger_auto_remediation(self, alert: QualityAlert) -> None:
        """Trigger automatic remediation for an alert."""
        
        try:
            strategy = self.remediation_strategies[alert.metric_name]
            self.executor.submit(strategy, alert)
            self.logger.info(f"Triggered auto-remediation for {alert.metric_name}")
        except Exception as e:
            self.logger.error(f"Auto-remediation failed for {alert.metric_name}: {e}")
    
    async def _get_improvement_recommendations(
        self, 
        gate_name: str, 
        current_score: float
    ) -> List[str]:
        """Get improvement recommendations for a specific quality gate."""
        
        recommendations = []
        
        gate_specific_recommendations = {
            "completeness": [
                "Add missing required fields",
                "Include comprehensive ethical considerations",
                "Add detailed model limitations",
                "Include quantitative analysis"
            ],
            "accuracy": [
                "Validate numerical consistency",
                "Cross-check data relationships",
                "Verify temporal consistency",
                "Review calculation methods"
            ],
            "security": [
                "Scan for sensitive information",
                "Review data privacy compliance",
                "Check for exposed credentials",
                "Validate access controls"
            ],
            "performance": [
                "Optimize generation algorithms",
                "Review caching strategies",
                "Reduce computational complexity",
                "Scale infrastructure resources"
            ],
            "compliance": [
                "Add regulatory compliance statements",
                "Include bias risk assessments",
                "Document environmental impact",
                "Add usage recommendations"
            ],
            "maintainability": [
                "Add version information",
                "Include update timestamps",
                "Provide contact information",
                "Document maintenance procedures"
            ]
        }
        
        base_recommendations = gate_specific_recommendations.get(gate_name, [])
        
        # Add score-specific recommendations
        if current_score < 0.5:
            recommendations.append("Critical improvement needed - consider full review")
        elif current_score < 0.7:
            recommendations.append("Moderate improvement needed")
        
        return base_recommendations + recommendations
    
    def _initialize_trend_analyzers(self) -> None:
        """Initialize trend analyzers for different metrics."""
        
        metric_names = ["completeness", "accuracy", "security", "performance", 
                       "compliance", "maintainability", "overall_quality"]
        
        for metric_name in metric_names:
            self.trend_analyzers[metric_name] = TrendAnalyzer(
                metric_name=metric_name,
                window_size=10
            )
    
    def _initialize_remediation_strategies(self) -> None:
        """Initialize automatic remediation strategies."""
        
        self.remediation_strategies = {
            "performance": self._remediate_performance,
            "security": self._remediate_security,
            "completeness": self._remediate_completeness
        }
    
    def _remediate_performance(self, alert: QualityAlert) -> None:
        """Auto-remediate performance issues."""
        self.logger.info(f"Auto-remediating performance issue: {alert.description}")
        # Implementation would include cache optimization, resource scaling, etc.
    
    def _remediate_security(self, alert: QualityAlert) -> None:
        """Auto-remediate security issues."""
        self.logger.info(f"Auto-remediating security issue: {alert.description}")
        # Implementation would include data redaction, access control updates, etc.
    
    def _remediate_completeness(self, alert: QualityAlert) -> None:
        """Auto-remediate completeness issues."""
        self.logger.info(f"Auto-remediating completeness issue: {alert.description}")
        # Implementation would include template updates, field validation, etc.
    
    def _update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on historical performance."""
        
        if len(self.metric_history) < 50:
            return  # Need more data
        
        recent_snapshots = list(self.metric_history)[-50:]
        
        # Update overall quality threshold
        recent_qualities = [s.overall_quality for s in recent_snapshots]
        q75 = np.percentile(recent_qualities, 75)
        self.adaptive_thresholds["overall_quality"] = max(0.6, min(0.9, q75 * 0.9))
        
        # Update processing time threshold
        recent_times = [s.processing_time for s in recent_snapshots]
        q90 = np.percentile(recent_times, 90)
        self.adaptive_thresholds["processing_time"] = max(1.0, q90 * 1.2)
        
        # Update error rate threshold
        recent_errors = [s.error_count for s in recent_snapshots]
        max_errors = max(recent_errors) if recent_errors else 0
        self.adaptive_thresholds["error_rate"] = min(0.1, max(0.01, max_errors * 0.5))
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data to prevent memory issues."""
        
        # Clean old alerts (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.alert_history = deque(
            [alert for alert in self.alert_history if alert.timestamp > cutoff_time],
            maxlen=100
        )
    
    def get_dashboard_data(self) -> QualityDashboard:
        """Get current dashboard data."""
        
        current_metrics = {}
        trends = {}
        
        if self.metric_history:
            latest = self.metric_history[-1]
            current_metrics = {
                "overall_quality": latest.overall_quality,
                "processing_time": latest.processing_time,
                "error_count": latest.error_count,
                **latest.gate_scores
            }
            
            # Calculate trends
            for metric_name, analyzer in self.trend_analyzers.items():
                if metric_name in current_metrics:
                    trends[metric_name] = analyzer.get_current_trend()
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.alert_history 
                        if alert.timestamp > datetime.now() - timedelta(hours=1)]
        
        # Performance summary
        performance_summary = self._generate_performance_summary()
        
        # System health
        system_health = self._calculate_system_health()
        
        # Uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Get recommendations from learning system
        recommendations = []
        if self.learning_system:
            learning_summary = self.learning_system.get_learning_summary()
            # Convert optimization strategies to recommendations (simplified)
            for strategy in learning_summary.get("optimization_strategies", [])[:3]:
                rec = AdaptiveRecommendation(
                    recommendation_id=f"strategy_{int(time.time())}",
                    description=strategy.get("description", ""),
                    category="optimization",
                    confidence=strategy.get("weight", 0.5),
                    expected_improvement=0.05,
                    implementation_cost="medium",
                    learned_from_pattern="system_analysis"
                )
                recommendations.append(rec)
        
        return QualityDashboard(
            current_metrics=current_metrics,
            trends=trends,
            alerts=recent_alerts,
            performance_summary=performance_summary,
            recommendations=recommendations,
            system_health=system_health,
            uptime=uptime
        )
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        
        if not self.metric_history:
            return {}
        
        recent_snapshots = list(self.metric_history)[-50:]
        
        qualities = [s.overall_quality for s in recent_snapshots]
        processing_times = [s.processing_time for s in recent_snapshots]
        error_counts = [s.error_count for s in recent_snapshots]
        
        return {
            "avg_quality": statistics.mean(qualities),
            "min_quality": min(qualities),
            "max_quality": max(qualities),
            "avg_processing_time": statistics.mean(processing_times),
            "total_generations": len(recent_snapshots),
            "total_errors": sum(error_counts),
            "success_rate": sum(1 for q in qualities if q > 0.8) / len(qualities)
        }
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health."""
        
        if not self.metric_history:
            return "unknown"
        
        recent_snapshots = list(self.metric_history)[-10:]
        avg_quality = statistics.mean(s.overall_quality for s in recent_snapshots)
        
        critical_alerts = sum(1 for alert in self.alert_history 
                            if alert.level in [AlertLevel.CRITICAL, AlertLevel.URGENT]
                            and alert.timestamp > datetime.now() - timedelta(hours=1))
        
        if critical_alerts > 0:
            return "critical"
        elif avg_quality > 0.85:
            return "excellent"
        elif avg_quality > 0.75:
            return "good"
        elif avg_quality > 0.65:
            return "fair"
        else:
            return "poor"
    
    def subscribe_to_alerts(self, callback: Callable[[QualityAlert], None]) -> None:
        """Subscribe to quality alerts."""
        self.alert_subscribers.append(callback)
    
    def unsubscribe_from_alerts(self, callback: Callable[[QualityAlert], None]) -> None:
        """Unsubscribe from quality alerts."""
        if callback in self.alert_subscribers:
            self.alert_subscribers.remove(callback)
    
    def _save_monitoring_state(self) -> None:
        """Save monitoring state to disk."""
        
        state_file = self.monitor_dir / "monitoring_state.json"
        
        state = {
            "adaptive_thresholds": self.adaptive_thresholds,
            "alert_count": len(self.alert_history),
            "metric_count": len(self.metric_history),
            "uptime": (datetime.now() - self.start_time).total_seconds(),
            "last_saved": datetime.now().isoformat()
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save monitoring state: {e}")
    
    def _load_monitoring_state(self) -> None:
        """Load monitoring state from disk."""
        
        state_file = self.monitor_dir / "monitoring_state.json"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.adaptive_thresholds = state.get("adaptive_thresholds", self.adaptive_thresholds)
            
            self.logger.info("Loaded monitoring state")
            
        except Exception as e:
            self.logger.error(f"Failed to load monitoring state: {e}")


class TrendAnalyzer:
    """Analyzer for detecting trends in metric values."""
    
    def __init__(self, metric_name: str, window_size: int = 10):
        self.metric_name = metric_name
        self.window_size = window_size
        self.values: deque = deque(maxlen=window_size)
        self.current_trend = MetricTrend.STABLE
    
    def analyze_trend(self, new_value: float) -> MetricTrend:
        """Analyze trend with new value."""
        
        self.values.append(new_value)
        
        if len(self.values) < 3:
            return MetricTrend.STABLE
        
        # Calculate trend
        values_list = list(self.values)
        trend = self._calculate_trend(values_list)
        self.current_trend = trend
        
        return trend
    
    def _calculate_trend(self, values: List[float]) -> MetricTrend:
        """Calculate trend from values."""
        if len(values) < 3:
            return MetricTrend.STABLE
        
        # Linear regression
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        # Volatility
        volatility = np.std(values) / (np.mean(values) + 1e-6)
        
        if volatility > 0.15:
            return MetricTrend.VOLATILE
        elif slope > 0.02:
            return MetricTrend.IMPROVING
        elif slope < -0.02:
            return MetricTrend.DEGRADING
        else:
            return MetricTrend.STABLE
    
    def get_current_trend(self) -> MetricTrend:
        """Get current trend."""
        return self.current_trend


# Factory function
def create_realtime_quality_monitor(
    monitor_dir: Optional[Path] = None,
    quality_gates: Optional[ProgressiveQualityGateEngine] = None,
    learning_system: Optional[AutonomousLearningSystem] = None
) -> RealTimeQualityMonitor:
    """Create a real-time quality monitor with default configuration."""
    return RealTimeQualityMonitor(
        monitor_dir=monitor_dir,
        quality_gates=quality_gates,
        learning_system=learning_system
    )