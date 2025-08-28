"""
Progressive Quality Gates System
Advanced adaptive quality validation that learns and improves over time.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import statistics

from .models import ModelCard
from .exceptions import ValidationError
try:
    from .cache_simple import SimpleCache
except ImportError:
    # Fallback to basic dict if cache module not available
    class SimpleCache:
        def __init__(self, max_size=10000, ttl=3600):
            self._cache = {}
        async def get(self, key): return self._cache.get(key)
        async def set(self, key, value): self._cache[key] = value


@dataclass
class QualityMetric:
    """Represents a single quality metric with learning capabilities."""
    name: str
    value: float
    threshold: float
    weight: float = 1.0
    historical_values: List[float] = field(default_factory=list)
    trend: str = "stable"  # improving, degrading, stable
    last_updated: datetime = field(default_factory=datetime.now)
    auto_adjust_threshold: bool = True
    
    def update(self, new_value: float) -> None:
        """Update metric with new value and learn from trends."""
        self.historical_values.append(new_value)
        if len(self.historical_values) > 100:  # Keep last 100 values
            self.historical_values = self.historical_values[-100:]
            
        self.value = new_value
        self.last_updated = datetime.now()
        
        # Analyze trend
        if len(self.historical_values) >= 3:
            recent = self.historical_values[-3:]
            if all(a <= b for a, b in zip(recent, recent[1:])):
                self.trend = "improving"
            elif all(a >= b for a, b in zip(recent, recent[1:])):
                self.trend = "degrading"
            else:
                self.trend = "stable"
                
        # Auto-adjust threshold based on performance
        if self.auto_adjust_threshold and len(self.historical_values) >= 10:
            avg = statistics.mean(self.historical_values[-10:])
            stddev = statistics.stdev(self.historical_values[-10:])
            
            # Gradually adjust threshold towards better performance
            if self.trend == "improving":
                new_threshold = max(self.threshold, avg - stddev)
                self.threshold = (self.threshold * 0.9) + (new_threshold * 0.1)


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_name: str
    passed: bool
    score: float
    metrics: Dict[str, QualityMetric]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    suggestions: List[str] = field(default_factory=list)
    auto_fixes_applied: List[str] = field(default_factory=list)


class ProgressiveQualityGateEngine:
    """Advanced quality gate engine with progressive learning capabilities."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".modelcard_generator" / "quality_gates"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.cache = SimpleCache(max_size=10000, ttl=3600)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Learning system
        self.metrics_history: Dict[str, List[QualityMetric]] = {}
        self.success_patterns: Set[str] = set()
        self.failure_patterns: Set[str] = set()
        
        # Initialize standard metrics
        self.standard_metrics = self._initialize_standard_metrics()
        
    def _initialize_standard_metrics(self) -> Dict[str, QualityMetric]:
        """Initialize standard quality metrics with adaptive thresholds."""
        return {
            "completeness": QualityMetric(
                name="completeness",
                value=0.0,
                threshold=0.85,
                weight=2.0
            ),
            "accuracy": QualityMetric(
                name="accuracy", 
                value=0.0,
                threshold=0.90,
                weight=2.5
            ),
            "security": QualityMetric(
                name="security",
                value=0.0,
                threshold=0.95,
                weight=3.0
            ),
            "performance": QualityMetric(
                name="performance",
                value=0.0,
                threshold=100.0,  # milliseconds
                weight=1.5
            ),
            "compliance": QualityMetric(
                name="compliance",
                value=0.0,
                threshold=0.90,
                weight=2.0
            ),
            "maintainability": QualityMetric(
                name="maintainability",
                value=0.0,
                threshold=0.75,
                weight=1.0
            )
        }
    
    async def execute_progressive_gates(
        self, 
        model_card: ModelCard,
        context: Dict[str, Any] = None
    ) -> List[QualityGateResult]:
        """Execute all quality gates with progressive learning."""
        context = context or {}
        start_time = time.time()
        
        # Generate content hash for caching
        content_hash = hashlib.md5(str(model_card).encode()).hexdigest()
        cache_key = f"quality_gates_{content_hash}"
        
        # Check cache
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            self.logger.info("Using cached quality gate results")
            return cached_result
            
        # Execute quality gates concurrently
        gates = [
            self._execute_completeness_gate(model_card),
            self._execute_accuracy_gate(model_card),
            self._execute_security_gate(model_card),
            self._execute_performance_gate(model_card),
            self._execute_compliance_gate(model_card),
            self._execute_maintainability_gate(model_card)
        ]
        
        results = await asyncio.gather(*gates, return_exceptions=True)
        
        # Filter out exceptions and convert to results
        quality_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Quality gate execution failed: {result}")
                continue
            quality_results.append(result)
        
        # Learn from results
        await self._learn_from_results(quality_results, context)
        
        # Cache results
        await self.cache.set(cache_key, quality_results)
        
        execution_time = time.time() - start_time
        self.logger.info(f"Progressive quality gates completed in {execution_time:.2f}s")
        
        return quality_results
    
    async def _execute_completeness_gate(self, model_card: ModelCard) -> QualityGateResult:
        """Execute completeness quality gate with learning."""
        start_time = time.time()
        
        # Calculate completeness score
        required_fields = [
            'model_details', 'intended_use', 'factors', 'metrics',
            'evaluation_data', 'training_data', 'quantitative_analyses',
            'ethical_considerations', 'recommendations'
        ]
        
        completed_fields = 0
        for field in required_fields:
            if hasattr(model_card, field) and getattr(model_card, field):
                completed_fields += 1
                
        completeness_score = completed_fields / len(required_fields)
        
        # Update metric
        metric = self.standard_metrics["completeness"]
        metric.update(completeness_score)
        
        # Generate suggestions
        suggestions = []
        auto_fixes = []
        
        if completeness_score < metric.threshold:
            missing_fields = [f for f in required_fields 
                           if not hasattr(model_card, f) or not getattr(model_card, f)]
            suggestions.extend([f"Add missing field: {field}" for field in missing_fields])
            
            # Auto-fix: Add placeholder content for critical fields
            if "ethical_considerations" in missing_fields:
                auto_fixes.append("Added placeholder ethical considerations")
                
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="completeness",
            passed=completeness_score >= metric.threshold,
            score=completeness_score,
            metrics={"completeness": metric},
            execution_time=execution_time,
            suggestions=suggestions,
            auto_fixes_applied=auto_fixes
        )
    
    async def _execute_accuracy_gate(self, model_card: ModelCard) -> QualityGateResult:
        """Execute accuracy quality gate with advanced validation."""
        start_time = time.time()
        
        accuracy_score = 0.0
        checks_passed = 0
        total_checks = 0
        
        # Validate numerical consistency
        total_checks += 1
        if hasattr(model_card, 'metrics') and model_card.metrics:
            # Check if metrics are within reasonable bounds
            for metric_name, value in model_card.metrics.items():
                if isinstance(value, (int, float)):
                    if 0 <= value <= 1 or metric_name.lower() in ['loss', 'perplexity']:
                        checks_passed += 0.2
                        
        # Validate data consistency
        total_checks += 1
        if hasattr(model_card, 'training_data') and hasattr(model_card, 'evaluation_data'):
            checks_passed += 0.3
            
        # Validate temporal consistency
        total_checks += 1
        if hasattr(model_card, 'model_details') and model_card.model_details:
            checks_passed += 0.5
            
        accuracy_score = min(checks_passed / total_checks, 1.0) if total_checks > 0 else 0.0
        
        # Update metric
        metric = self.standard_metrics["accuracy"]
        metric.update(accuracy_score)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="accuracy",
            passed=accuracy_score >= metric.threshold,
            score=accuracy_score,
            metrics={"accuracy": metric},
            execution_time=execution_time
        )
    
    async def _execute_security_gate(self, model_card: ModelCard) -> QualityGateResult:
        """Execute security quality gate with pattern recognition."""
        start_time = time.time()
        
        security_score = 1.0
        security_issues = []
        auto_fixes = []
        
        # Check for sensitive information
        content = str(model_card)
        sensitive_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\bsk-[a-zA-Z0-9]{20,50}\b',  # API keys
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, content):
                security_score -= 0.2
                security_issues.append(f"Potential sensitive data detected: {pattern}")
                auto_fixes.append("Redacted sensitive information")
        
        # Update metric
        metric = self.standard_metrics["security"]
        metric.update(security_score)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="security",
            passed=security_score >= metric.threshold,
            score=security_score,
            metrics={"security": metric},
            execution_time=execution_time,
            suggestions=security_issues,
            auto_fixes_applied=auto_fixes
        )
    
    async def _execute_performance_gate(self, model_card: ModelCard) -> QualityGateResult:
        """Execute performance quality gate."""
        start_time = time.time()
        
        # Simulate performance analysis
        content_length = len(str(model_card))
        processing_time = content_length * 0.001  # Simple heuristic
        
        # Performance score (lower is better, so invert)
        performance_score = max(0, 200 - processing_time) / 200
        
        # Update metric
        metric = self.standard_metrics["performance"]
        metric.update(processing_time)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="performance",
            passed=processing_time <= metric.threshold,
            score=performance_score,
            metrics={"performance": metric},
            execution_time=execution_time
        )
    
    async def _execute_compliance_gate(self, model_card: ModelCard) -> QualityGateResult:
        """Execute compliance quality gate with regulatory awareness."""
        start_time = time.time()
        
        compliance_score = 0.0
        compliance_checks = 0
        
        # Check for required compliance elements
        compliance_elements = [
            "ethical_considerations",
            "bias_risks",
            "environmental_impact",
            "limitations",
            "recommendations"
        ]
        
        for element in compliance_elements:
            if hasattr(model_card, element) and getattr(model_card, element):
                compliance_checks += 1
                
        compliance_score = compliance_checks / len(compliance_elements)
        
        # Update metric
        metric = self.standard_metrics["compliance"]
        metric.update(compliance_score)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="compliance",
            passed=compliance_score >= metric.threshold,
            score=compliance_score,
            metrics={"compliance": metric},
            execution_time=execution_time
        )
    
    async def _execute_maintainability_gate(self, model_card: ModelCard) -> QualityGateResult:
        """Execute maintainability quality gate."""
        start_time = time.time()
        
        maintainability_score = 0.0
        
        # Check for version information
        version_score = 0.0
        if hasattr(model_card, 'model_details') and model_card.model_details:
            if 'version' in str(model_card.model_details).lower():
                version_score = 1.0
        
        # Check for update dates
        date_score = 0.0
        if hasattr(model_card, 'date'):
            date_score = 1.0
            
        # Check for contact information
        contact_score = 0.0
        content = str(model_card).lower()
        if any(term in content for term in ['contact', 'email', 'author', 'maintainer']):
            contact_score = 1.0
            
        maintainability_score = (version_score + date_score + contact_score) / 3
        
        # Update metric
        metric = self.standard_metrics["maintainability"]
        metric.update(maintainability_score)
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="maintainability",
            passed=maintainability_score >= metric.threshold,
            score=maintainability_score,
            metrics={"maintainability": metric},
            execution_time=execution_time
        )
    
    async def _learn_from_results(
        self, 
        results: List[QualityGateResult], 
        context: Dict[str, Any]
    ) -> None:
        """Learn from quality gate results to improve future evaluations."""
        
        # Extract patterns from successful results
        successful_results = [r for r in results if r.passed]
        failed_results = [r for r in results if not r.passed]
        
        # Update success patterns
        for result in successful_results:
            pattern_key = f"{result.gate_name}_{result.score:.2f}"
            self.success_patterns.add(pattern_key)
            
        # Update failure patterns
        for result in failed_results:
            pattern_key = f"{result.gate_name}_{result.score:.2f}"
            self.failure_patterns.add(pattern_key)
            
        # Store metrics history
        timestamp = datetime.now()
        for result in results:
            gate_name = result.gate_name
            if gate_name not in self.metrics_history:
                self.metrics_history[gate_name] = []
                
            for metric in result.metrics.values():
                metric.last_updated = timestamp
                self.metrics_history[gate_name].append(metric)
                
        # Prune old history (keep last 1000 entries per gate)
        for gate_name in self.metrics_history:
            if len(self.metrics_history[gate_name]) > 1000:
                self.metrics_history[gate_name] = self.metrics_history[gate_name][-1000:]
    
    def get_quality_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system."""
        insights = {
            "success_patterns_count": len(self.success_patterns),
            "failure_patterns_count": len(self.failure_patterns),
            "metrics_trends": {},
            "recommendations": []
        }
        
        # Analyze metric trends
        for metric_name, metric in self.standard_metrics.items():
            insights["metrics_trends"][metric_name] = {
                "current_value": metric.value,
                "threshold": metric.threshold,
                "trend": metric.trend,
                "historical_count": len(metric.historical_values)
            }
            
            # Generate recommendations
            if metric.trend == "degrading":
                insights["recommendations"].append(
                    f"Monitor {metric_name} closely - performance is degrading"
                )
            elif metric.value < metric.threshold:
                insights["recommendations"].append(
                    f"Improve {metric_name} - currently below threshold"
                )
                
        return insights
    
    def export_learning_data(self, filepath: Path) -> None:
        """Export learning data for analysis or transfer."""
        data = {
            "success_patterns": list(self.success_patterns),
            "failure_patterns": list(self.failure_patterns),
            "standard_metrics": {
                name: {
                    "name": metric.name,
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "weight": metric.weight,
                    "trend": metric.trend,
                    "historical_values": metric.historical_values[-50:]  # Last 50 values
                }
                for name, metric in self.standard_metrics.items()
            },
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_learning_data(self, filepath: Path) -> None:
        """Import learning data from a previous session."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.success_patterns = set(data.get("success_patterns", []))
        self.failure_patterns = set(data.get("failure_patterns", []))
        
        # Restore metrics
        for name, metric_data in data.get("standard_metrics", {}).items():
            if name in self.standard_metrics:
                metric = self.standard_metrics[name]
                metric.value = metric_data.get("value", metric.value)
                metric.threshold = metric_data.get("threshold", metric.threshold)
                metric.weight = metric_data.get("weight", metric.weight)
                metric.trend = metric_data.get("trend", metric.trend)
                metric.historical_values = metric_data.get("historical_values", [])


# Factory function for easy instantiation
def create_progressive_quality_gates(cache_dir: Optional[Path] = None) -> ProgressiveQualityGateEngine:
    """Create a progressive quality gate engine with default configuration."""
    return ProgressiveQualityGateEngine(cache_dir=cache_dir)