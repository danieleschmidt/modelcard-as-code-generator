"""
Drift detection for model cards.

Detects changes in model performance, data characteristics,
and other aspects that may indicate model drift requiring
model card updates.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime

from ..core.model_card import ModelCard, MetricValue


@dataclass
class DriftChange:
    """Represents a detected change/drift."""
    metric_name: str
    old_value: Union[float, str, None]
    new_value: Union[float, str, None]
    delta: Optional[float]
    delta_percent: Optional[float]
    threshold_exceeded: bool
    significance: str  # low, medium, high, critical


@dataclass
class DriftReport:
    """Report of detected drift."""
    has_drift: bool
    timestamp: str
    changes: List[DriftChange]
    summary: str
    recommendation: str
    severity: str  # low, medium, high, critical


class DriftDetector:
    """Detects drift in model cards and performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default thresholds for different metrics
        self.default_thresholds = {
            'accuracy': 0.02,  # 2% change
            'precision': 0.02,
            'recall': 0.02,
            'f1': 0.02,
            'auc': 0.02,
            'loss': 0.1,  # 10% change
            'mse': 0.1,
            'mae': 0.1,
            'rmse': 0.1,
            'bleu': 0.02,
            'rouge': 0.02,
            'perplexity': 0.1,
            'inference_time': 0.2,  # 20% change
            'memory_usage': 0.15,  # 15% change
        }
    
    def check_drift(
        self,
        current_card: ModelCard,
        new_data: Union[Dict[str, Any], ModelCard, str, Path],
        thresholds: Optional[Dict[str, float]] = None
    ) -> DriftReport:
        """
        Check for drift between current model card and new data.
        
        Args:
            current_card: Current model card
            new_data: New evaluation data, model card, or file path
            thresholds: Custom thresholds for drift detection
            
        Returns:
            DriftReport with detected changes
        """
        try:
            # Parse new data
            new_metrics = self._extract_metrics_from_source(new_data)
            
            # Use provided thresholds or defaults
            thresholds = thresholds or self.default_thresholds
            
            # Detect changes
            changes = self._detect_metric_changes(current_card.metrics, new_metrics, thresholds)
            
            # Assess overall drift
            has_drift = any(change.threshold_exceeded for change in changes)
            severity = self._assess_severity(changes)
            
            # Generate report
            report = DriftReport(
                has_drift=has_drift,
                timestamp=datetime.now().isoformat(),
                changes=changes,
                summary=self._generate_summary(changes),
                recommendation=self._generate_recommendation(changes, severity),
                severity=severity
            )
            
            self.logger.info(f"Drift detection completed. Has drift: {has_drift}, Severity: {severity}")
            return report
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return DriftReport(
                has_drift=False,
                timestamp=datetime.now().isoformat(),
                changes=[],
                summary=f"Drift detection failed: {str(e)}",
                recommendation="Manual review required due to detection failure.",
                severity="unknown"
            )
    
    def _extract_metrics_from_source(self, source: Union[Dict, ModelCard, str, Path]) -> List[MetricValue]:
        """Extract metrics from various source types."""
        if isinstance(source, ModelCard):
            return source.metrics
        
        elif isinstance(source, dict):
            # Assume it's evaluation data or model card dict
            if 'metrics' in source:
                # Model card format
                if isinstance(source['metrics'], list):
                    return [
                        MetricValue(
                            name=m.get('name', ''),
                            value=m.get('value'),
                            unit=m.get('unit'),
                            description=m.get('description'),
                            dataset=m.get('dataset')
                        )
                        for m in source['metrics']
                    ]
                else:
                    # Dictionary of metrics
                    return [
                        MetricValue(name=name, value=value)
                        for name, value in source['metrics'].items()
                    ]
            else:
                # Assume top-level metrics
                return [
                    MetricValue(name=name, value=value)
                    for name, value in source.items()
                    if isinstance(value, (int, float))
                ]
        
        elif isinstance(source, (str, Path)):
            # Load from file
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    data = json.load(f)
                    return self._extract_metrics_from_source(data)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def _detect_metric_changes(
        self,
        current_metrics: List[MetricValue],
        new_metrics: List[MetricValue],
        thresholds: Dict[str, float]
    ) -> List[DriftChange]:
        """Detect changes in metrics."""
        changes = []
        
        # Create lookup for current metrics
        current_lookup = {metric.name: metric for metric in current_metrics}
        new_lookup = {metric.name: metric for metric in new_metrics}
        
        # Check for changes in existing metrics
        for metric_name, current_metric in current_lookup.items():
            if metric_name in new_lookup:
                new_metric = new_lookup[metric_name]
                change = self._compare_metrics(current_metric, new_metric, thresholds)
                if change:
                    changes.append(change)
        
        # Check for new metrics
        for metric_name, new_metric in new_lookup.items():
            if metric_name not in current_lookup:
                changes.append(DriftChange(
                    metric_name=metric_name,
                    old_value=None,
                    new_value=new_metric.value,
                    delta=None,
                    delta_percent=None,
                    threshold_exceeded=False,  # New metrics don't exceed thresholds by default
                    significance="low"
                ))
        
        # Check for removed metrics
        for metric_name in current_lookup:
            if metric_name not in new_lookup:
                changes.append(DriftChange(
                    metric_name=metric_name,
                    old_value=current_lookup[metric_name].value,
                    new_value=None,
                    delta=None,
                    delta_percent=None,
                    threshold_exceeded=True,  # Removed metrics are significant
                    significance="high"
                ))
        
        return changes
    
    def _compare_metrics(
        self,
        current_metric: MetricValue,
        new_metric: MetricValue,
        thresholds: Dict[str, float]
    ) -> Optional[DriftChange]:
        """Compare two metrics and detect significant changes."""
        if not isinstance(current_metric.value, (int, float)) or not isinstance(new_metric.value, (int, float)):
            # Can't compare non-numeric values for drift
            if current_metric.value != new_metric.value:
                return DriftChange(
                    metric_name=current_metric.name,
                    old_value=current_metric.value,
                    new_value=new_metric.value,
                    delta=None,
                    delta_percent=None,
                    threshold_exceeded=True,
                    significance="medium"
                )
            return None
        
        # Calculate changes
        delta = new_metric.value - current_metric.value
        delta_percent = (delta / current_metric.value * 100) if current_metric.value != 0 else float('inf')
        
        # Get threshold for this metric
        threshold = self._get_threshold(current_metric.name, thresholds)
        
        # Check if threshold exceeded
        abs_delta_percent = abs(delta_percent)
        threshold_exceeded = abs_delta_percent > (threshold * 100)
        
        # Determine significance
        significance = self._determine_significance(abs_delta_percent, threshold * 100)
        
        # Only report if there's a meaningful change
        if abs(delta) > 1e-6:  # Avoid floating point noise
            return DriftChange(
                metric_name=current_metric.name,
                old_value=current_metric.value,
                new_value=new_metric.value,
                delta=delta,
                delta_percent=delta_percent,
                threshold_exceeded=threshold_exceeded,
                significance=significance
            )
        
        return None
    
    def _get_threshold(self, metric_name: str, thresholds: Dict[str, float]) -> float:
        """Get threshold for a specific metric."""
        # Check exact match first
        if metric_name in thresholds:
            return thresholds[metric_name]
        
        # Check for partial matches
        metric_lower = metric_name.lower()
        for threshold_name, threshold_value in thresholds.items():
            if threshold_name.lower() in metric_lower or metric_lower in threshold_name.lower():
                return threshold_value
        
        # Default threshold
        return 0.05  # 5% default
    
    def _determine_significance(self, abs_delta_percent: float, threshold_percent: float) -> str:
        """Determine significance level of the change."""
        if abs_delta_percent < threshold_percent * 0.5:
            return "low"
        elif abs_delta_percent < threshold_percent:
            return "medium"
        elif abs_delta_percent < threshold_percent * 2:
            return "high"
        else:
            return "critical"
    
    def _assess_severity(self, changes: List[DriftChange]) -> str:
        """Assess overall severity of detected changes."""
        if not changes:
            return "none"
        
        # Count changes by significance
        significance_counts = {}
        for change in changes:
            significance_counts[change.significance] = significance_counts.get(change.significance, 0) + 1
        
        # Determine overall severity
        if significance_counts.get("critical", 0) > 0:
            return "critical"
        elif significance_counts.get("high", 0) > 2:
            return "critical"
        elif significance_counts.get("high", 0) > 0:
            return "high"
        elif significance_counts.get("medium", 0) > 3:
            return "high"
        elif significance_counts.get("medium", 0) > 0:
            return "medium"
        else:
            return "low"
    
    def _generate_summary(self, changes: List[DriftChange]) -> str:
        """Generate summary of detected changes."""
        if not changes:
            return "No significant changes detected."
        
        significant_changes = [c for c in changes if c.threshold_exceeded]
        
        if not significant_changes:
            return f"Detected {len(changes)} minor changes, none exceeding thresholds."
        
        summary_parts = []
        
        # Group by type
        degraded = [c for c in significant_changes if c.delta and c.delta < 0 and 'accuracy' in c.metric_name.lower()]
        improved = [c for c in significant_changes if c.delta and c.delta > 0 and 'accuracy' in c.metric_name.lower()]
        new_metrics = [c for c in changes if c.old_value is None]
        removed_metrics = [c for c in changes if c.new_value is None]
        
        if degraded:
            summary_parts.append(f"{len(degraded)} metrics degraded")
        if improved:
            summary_parts.append(f"{len(improved)} metrics improved")
        if new_metrics:
            summary_parts.append(f"{len(new_metrics)} new metrics")
        if removed_metrics:
            summary_parts.append(f"{len(removed_metrics)} metrics removed")
        
        if not summary_parts:
            summary_parts.append(f"{len(significant_changes)} metrics changed significantly")
        
        return f"Detected {len(significant_changes)} significant changes: " + ", ".join(summary_parts) + "."
    
    def _generate_recommendation(self, changes: List[DriftChange], severity: str) -> str:
        """Generate recommendation based on detected changes."""
        if severity == "critical":
            return (
                "Critical changes detected. Immediate review required. "
                "Consider model retraining, validation, or rollback."
            )
        elif severity == "high":
            return (
                "Significant changes detected. Review model performance and "
                "consider updating model card. Monitor closely."
            )
        elif severity == "medium":
            return (
                "Moderate changes detected. Update model card if changes are "
                "confirmed. Continue monitoring."
            )
        elif severity == "low":
            return (
                "Minor changes detected. Consider updating model card "
                "documentation during next regular review."
            )
        else:
            return "No significant changes detected. Continue regular monitoring."
    
    def check_data_drift(
        self,
        current_card: ModelCard,
        new_data_stats: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None
    ) -> DriftReport:
        """Check for data drift based on dataset statistics."""
        # This would compare dataset statistics, distributions, etc.
        # For now, we'll implement a basic version
        
        changes = []
        
        # Check dataset changes
        current_datasets = [ds.name for ds in current_card.training_data + current_card.evaluation_data]
        new_datasets = new_data_stats.get('datasets', [])
        
        if set(current_datasets) != set(new_datasets):
            changes.append(DriftChange(
                metric_name="datasets",
                old_value=str(current_datasets),
                new_value=str(new_datasets),
                delta=None,
                delta_percent=None,
                threshold_exceeded=True,
                significance="high"
            ))
        
        has_drift = len(changes) > 0
        
        return DriftReport(
            has_drift=has_drift,
            timestamp=datetime.now().isoformat(),
            changes=changes,
            summary=f"Data drift analysis: {len(changes)} changes detected" if changes else "No data drift detected",
            recommendation="Review data sources and preprocessing" if has_drift else "Continue monitoring",
            severity="high" if has_drift else "none"
        )
    
    def suggest_updates(self, drift_report: DriftReport) -> List[str]:
        """Suggest specific model card updates based on drift report."""
        suggestions = []
        
        if not drift_report.has_drift:
            return ["No updates needed based on drift analysis"]
        
        for change in drift_report.changes:
            if change.threshold_exceeded:
                if change.new_value is None:
                    suggestions.append(f"Remove or archive metric: {change.metric_name}")
                elif change.old_value is None:
                    suggestions.append(f"Add new metric to model card: {change.metric_name} = {change.new_value}")
                else:
                    suggestions.append(f"Update {change.metric_name}: {change.old_value} → {change.new_value}")
        
        # General suggestions based on severity
        if drift_report.severity in ["critical", "high"]:
            suggestions.append("Update model card version and add drift explanation")
            suggestions.append("Review and update limitations and caveats section")
            suggestions.append("Consider adding performance degradation warnings")
        
        return suggestions