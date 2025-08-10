"""Model card drift detection functionality."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .models import ModelCard, DriftReport, DriftMetricChange


logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect drift in model cards by comparing metrics and content."""
    
    def __init__(self, default_thresholds: Optional[Dict[str, float]] = None):
        """Initialize drift detector with default thresholds."""
        self.default_thresholds = default_thresholds or {
            "accuracy": 0.02,      # 2% change
            "precision": 0.02,
            "recall": 0.02,
            "f1": 0.02,
            "f1_score": 0.02,
            "f1_macro": 0.02,
            "f1_micro": 0.02,
            "auc": 0.02,
            "roc_auc": 0.02,
            "loss": 0.05,          # 5% change for loss metrics
            "val_loss": 0.05,
            "inference_time": 0.1,  # 10% change for timing
            "inference_time_ms": 50  # 50ms change
        }
    
    def check(
        self,
        card: ModelCard,
        new_eval_results: Optional[Union[str, Path, Dict[str, Any]]] = None,
        new_card: Optional[ModelCard] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> DriftReport:
        """
        Check for drift between current card and new evaluation results or card.
        
        Args:
            card: Current model card
            new_eval_results: New evaluation results (file path or dict)
            new_card: New model card to compare against
            thresholds: Custom thresholds for drift detection
            
        Returns:
            DriftReport with detected changes
        """
        effective_thresholds = {**self.default_thresholds, **(thresholds or {})}
        changes = []
        
        if new_card:
            # Compare two model cards
            changes.extend(self._compare_cards(card, new_card, effective_thresholds))
        elif new_eval_results:
            # Compare card against new evaluation results
            changes.extend(self._compare_with_eval_results(card, new_eval_results, effective_thresholds))
        else:
            raise ValueError("Either new_eval_results or new_card must be provided")
        
        has_drift = any(change.is_significant for change in changes)
        
        logger.info(f"Drift detection completed: {len(changes)} changes found, {len([c for c in changes if c.is_significant])} significant")
        
        return DriftReport(
            has_drift=has_drift,
            changes=changes,
            timestamp=datetime.now()
        )
    
    def suggest_updates(self, drift_report: DriftReport) -> List[Dict[str, Any]]:
        """
        Suggest updates based on detected drift.
        
        Args:
            drift_report: Drift report from check() method
            
        Returns:
            List of suggested updates
        """
        suggestions = []
        
        for change in drift_report.significant_changes:
            suggestion = {
                "metric": change.metric_name,
                "action": "update_metric",
                "old_value": change.old_value,
                "new_value": change.new_value,
                "reason": f"Metric changed by {change.delta:+.3f} (threshold: {change.threshold})",
                "priority": self._get_update_priority(change)
            }
            suggestions.append(suggestion)
        
        # Add general suggestions based on drift patterns
        if len(drift_report.significant_changes) > 3:
            suggestions.append({
                "action": "full_regeneration",
                "reason": "Multiple significant changes detected",
                "priority": "high"
            })
        
        performance_changes = [c for c in drift_report.significant_changes 
                             if c.metric_name in ['accuracy', 'f1', 'precision', 'recall']]
        if performance_changes:
            avg_change = sum(c.delta for c in performance_changes) / len(performance_changes)
            if avg_change < -0.05:  # Performance degraded by >5%
                suggestions.append({
                    "action": "add_limitation",
                    "reason": "Performance degradation detected",
                    "priority": "high",
                    "content": "Recent evaluation shows potential performance degradation"
                })
        
        return suggestions
    
    def _compare_cards(self, old_card: ModelCard, new_card: ModelCard, thresholds: Dict[str, float]) -> List[DriftMetricChange]:
        """Compare metrics between two model cards."""
        changes = []
        
        # Create lookup for old metrics
        old_metrics = {metric.name: metric.value for metric in old_card.evaluation_results}
        
        # Compare each new metric with old values
        for new_metric in new_card.evaluation_results:
            metric_name = new_metric.name
            new_value = new_metric.value
            
            if metric_name in old_metrics:
                old_value = old_metrics[metric_name]
                delta = new_value - old_value
                threshold = thresholds.get(metric_name, 0.05)  # Default 5% threshold
                
                # For percentage metrics, use relative threshold
                if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'f1_score', 'auc', 'roc_auc']:
                    relative_threshold = threshold
                    is_significant = abs(delta) > relative_threshold
                else:
                    # For other metrics, use absolute threshold
                    is_significant = abs(delta) > threshold
                
                change = DriftMetricChange(
                    metric_name=metric_name,
                    old_value=old_value,
                    new_value=new_value,
                    delta=delta,
                    threshold=threshold,
                    is_significant=is_significant
                )
                changes.append(change)
                
                if is_significant:
                    logger.warning(f"Significant change in {metric_name}: {old_value:.3f} → {new_value:.3f} (Δ{delta:+.3f})")
        
        # Check for new metrics
        new_metric_names = {metric.name for metric in new_card.evaluation_results}
        old_metric_names = set(old_metrics.keys())
        
        added_metrics = new_metric_names - old_metric_names
        removed_metrics = old_metric_names - new_metric_names
        
        if added_metrics:
            logger.info(f"New metrics added: {', '.join(added_metrics)}")
        if removed_metrics:
            logger.warning(f"Metrics removed: {', '.join(removed_metrics)}")
        
        return changes
    
    def _compare_with_eval_results(self, card: ModelCard, eval_results: Union[str, Path, Dict[str, Any]], thresholds: Dict[str, float]) -> List[DriftMetricChange]:
        """Compare card metrics with new evaluation results."""
        # Parse evaluation results
        if isinstance(eval_results, (str, Path)):
            eval_data = self._parse_eval_file(eval_results)
        else:
            eval_data = eval_results
        
        changes = []
        
        # Create lookup for old metrics
        old_metrics = {metric.name: metric.value for metric in card.evaluation_results}
        
        # Compare with new evaluation data
        for metric_name, new_value in eval_data.items():
            if not isinstance(new_value, (int, float)):
                continue
                
            if metric_name in old_metrics:
                old_value = old_metrics[metric_name]
                delta = float(new_value) - old_value
                threshold = thresholds.get(metric_name, 0.05)
                
                # Determine significance based on metric type
                if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'f1_score', 'auc', 'roc_auc']:
                    is_significant = abs(delta) > threshold
                elif metric_name in ['loss', 'val_loss']:
                    # For loss metrics, consider relative change
                    relative_change = abs(delta) / old_value if old_value != 0 else abs(delta)
                    is_significant = relative_change > threshold
                else:
                    is_significant = abs(delta) > threshold
                
                change = DriftMetricChange(
                    metric_name=metric_name,
                    old_value=old_value,
                    new_value=float(new_value),
                    delta=delta,
                    threshold=threshold,
                    is_significant=is_significant
                )
                changes.append(change)
        
        return changes
    
    def _parse_eval_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse evaluation results from file."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
            except ImportError:
                yaml = None
            
            if yaml is None:
                raise DriftError("YAML support not available - install PyYAML")
            
            try:
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                raise DriftError(f"Failed to load YAML file {file_path}: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _get_update_priority(self, change: DriftMetricChange) -> str:
        """Determine update priority based on change characteristics."""
        metric_name = change.metric_name
        delta = abs(change.delta)
        
        # Critical metrics that should have high priority
        critical_metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
        
        if metric_name in critical_metrics:
            if delta > 0.1:  # >10% change
                return "critical"
            elif delta > 0.05:  # >5% change
                return "high"
            else:
                return "medium"
        
        # Performance metrics
        performance_metrics = ['inference_time', 'inference_time_ms', 'latency']
        if metric_name in performance_metrics:
            if delta > 0.5:  # >50% change
                return "high"
            else:
                return "medium"
        
        # Loss metrics (improvement is good, degradation is bad)
        loss_metrics = ['loss', 'val_loss', 'training_loss']
        if metric_name in loss_metrics:
            if change.delta > 0:  # Loss increased (bad)
                return "high" if delta > 0.1 else "medium"
            else:  # Loss decreased (good)
                return "low"
        
        return "low"
    
    def check_against_history(
        self, 
        card: ModelCard, 
        history_file: Union[str, Path],
        lookback_period: int = 5
    ) -> Dict[str, Any]:
        """
        Check drift against historical metric values.
        
        Args:
            card: Current model card
            history_file: File containing historical metrics
            lookback_period: Number of previous versions to compare against
            
        Returns:
            Dictionary containing drift analysis results
        """
        history_path = Path(history_file)
        
        if not history_path.exists():
            return {"error": "History file not found", "has_drift": False}
        
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        
        # Get recent history
        recent_history = history_data[-lookback_period:] if len(history_data) > lookback_period else history_data
        
        current_metrics = {metric.name: metric.value for metric in card.evaluation_results}
        drift_analysis = {}
        
        for metric_name, current_value in current_metrics.items():
            historical_values = []
            for record in recent_history:
                if metric_name in record.get('metrics', {}):
                    historical_values.append(record['metrics'][metric_name])
            
            if historical_values:
                mean_value = sum(historical_values) / len(historical_values)
                variance = sum((x - mean_value) ** 2 for x in historical_values) / len(historical_values)
                std_dev = variance ** 0.5
                
                # Z-score based drift detection
                z_score = (current_value - mean_value) / std_dev if std_dev > 0 else 0
                
                drift_analysis[metric_name] = {
                    "current_value": current_value,
                    "historical_mean": mean_value,
                    "historical_std": std_dev,
                    "z_score": z_score,
                    "is_anomaly": abs(z_score) > 2.0,  # 2 standard deviations
                    "trend": "increasing" if current_value > mean_value else "decreasing"
                }
        
        has_anomalies = any(data["is_anomaly"] for data in drift_analysis.values())
        
        return {
            "has_drift": has_anomalies,
            "analysis": drift_analysis,
            "lookback_period": lookback_period,
            "history_records": len(recent_history)
        }
    
    def save_snapshot(self, card: ModelCard, snapshot_file: Union[str, Path]) -> None:
        """
        Save current card metrics as a snapshot for future drift detection.
        
        Args:
            card: Model card to save
            snapshot_file: File to save snapshot to
        """
        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "model_name": card.model_details.name,
            "model_version": card.model_details.version,
            "metrics": {metric.name: metric.value for metric in card.evaluation_results},
            "metadata": {
                "training_data": card.training_details.training_data,
                "framework": card.training_details.framework,
                "hyperparameters": card.training_details.hyperparameters
            }
        }
        
        snapshot_path = Path(snapshot_file)
        
        # Load existing snapshots if file exists
        if snapshot_path.exists():
            with open(snapshot_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new snapshot
        history.append(snapshot_data)
        
        # Keep only last 50 snapshots to prevent file from growing too large
        if len(history) > 50:
            history = history[-50:]
        
        # Save updated history
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved metric snapshot to {snapshot_path}")


class AuditableCard(ModelCard):
    """Model card with enhanced audit trail capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_audit_trail()
    
    def update_metric(self, name: str, value: float, reason: Optional[str] = None) -> None:
        """Update metric with audit trail."""
        old_value = None
        for metric in self.evaluation_results:
            if metric.name == name:
                old_value = metric.value
                break
        
        super().update_metric(name, value, reason)
        
        # Enhanced audit logging
        self._log_change("metric_updated", {
            "metric_name": name,
            "old_value": old_value,
            "new_value": value,
            "reason": reason,
            "change_type": "update" if old_value is not None else "add"
        })
    
    def add_limitation(self, limitation: str, reason: Optional[str] = None) -> None:
        """Add limitation with audit trail."""
        super().add_limitation(limitation)
        
        self._log_change("limitation_added", {
            "limitation": limitation,
            "reason": reason,
            "total_limitations": len(self.limitations.known_limitations)
        })
    
    def get_change_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of changes since a specific time."""
        if since:
            relevant_changes = [
                change for change in self.audit_trail 
                if datetime.fromisoformat(change['timestamp']) >= since
            ]
        else:
            relevant_changes = self.audit_trail
        
        summary = {
            "total_changes": len(relevant_changes),
            "metric_updates": len([c for c in relevant_changes if c['action'] == 'metric_updated']),
            "limitations_added": len([c for c in relevant_changes if c['action'] == 'limitation_added']),
            "sections_added": len([c for c in relevant_changes if c['action'].startswith('Added section')]),
            "last_change": relevant_changes[-1] if relevant_changes else None
        }
        
        return summary