"""
Autonomous Learning System
Self-improving model card generation through continuous learning and adaptation.
"""

import asyncio
import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
import statistics
from collections import defaultdict, deque

from .models import ModelCard
from .progressive_quality_gates import QualityGateResult, ProgressiveQualityGateEngine


@dataclass
class LearningPattern:
    """Represents a learned pattern for model card generation."""
    pattern_id: str
    pattern_type: str  # "success", "failure", "optimization"
    context: Dict[str, Any]
    outcome_metrics: Dict[str, float]
    frequency: int = 0
    confidence: float = 0.0
    last_seen: datetime = field(default_factory=datetime.now)
    effectiveness: float = 0.0
    
    def update_effectiveness(self, new_metric: float) -> None:
        """Update pattern effectiveness with new observation."""
        self.effectiveness = (self.effectiveness * 0.8) + (new_metric * 0.2)
        self.frequency += 1
        self.last_seen = datetime.now()
        
        # Update confidence based on frequency and effectiveness
        self.confidence = min(0.95, (self.frequency / 100) * self.effectiveness)


@dataclass
class AdaptiveRecommendation:
    """Adaptive recommendation based on learned patterns."""
    recommendation_id: str
    description: str
    category: str
    confidence: float
    expected_improvement: float
    implementation_cost: str  # "low", "medium", "high"
    learned_from_pattern: str
    timestamp: datetime = field(default_factory=datetime.now)


class AutonomousLearningSystem:
    """
    Advanced learning system that continuously improves model card generation
    through pattern recognition, outcome prediction, and adaptive optimization.
    """
    
    def __init__(self, learning_dir: Optional[Path] = None, max_patterns: int = 10000):
        self.learning_dir = learning_dir or Path.home() / ".modelcard_generator" / "learning"
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.max_patterns = max_patterns
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Learning components
        self.patterns: Dict[str, LearningPattern] = {}
        self.pattern_sequences: deque = deque(maxlen=1000)
        self.outcome_history: deque = deque(maxlen=5000)
        
        # Adaptive systems
        self.feature_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.success_predictors: Dict[str, List[float]] = defaultdict(list)
        self.optimization_strategies: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.learning_metrics = {
            "patterns_discovered": 0,
            "successful_predictions": 0,
            "total_predictions": 0,
            "adaptation_count": 0,
            "avg_improvement": 0.0
        }
        
        # Load existing learning data
        self._load_learning_state()
    
    async def learn_from_generation(
        self,
        model_card: ModelCard,
        generation_context: Dict[str, Any],
        quality_results: List[QualityGateResult],
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> List[AdaptiveRecommendation]:
        """Learn from a model card generation session and provide adaptive recommendations."""
        
        # Extract features from the generation
        features = self._extract_features(model_card, generation_context, quality_results)
        
        # Calculate outcome quality
        overall_quality = self._calculate_overall_quality(quality_results)
        
        # Store outcome in history
        outcome = {
            "timestamp": datetime.now(),
            "features": features,
            "quality": overall_quality,
            "quality_results": quality_results,
            "user_feedback": user_feedback or {}
        }
        self.outcome_history.append(outcome)
        
        # Learn patterns
        await self._discover_patterns(features, overall_quality, quality_results)
        
        # Update feature weights
        self._update_feature_weights(features, overall_quality)
        
        # Generate adaptive recommendations
        recommendations = await self._generate_adaptive_recommendations(features, quality_results)
        
        # Update learning metrics
        self._update_learning_metrics(overall_quality, len(recommendations))
        
        # Periodically optimize the learning system
        if len(self.outcome_history) % 100 == 0:
            await self._optimize_learning_system()
        
        return recommendations
    
    def _extract_features(
        self,
        model_card: ModelCard,
        context: Dict[str, Any],
        quality_results: List[QualityGateResult]
    ) -> Dict[str, Any]:
        """Extract relevant features for learning."""
        
        features = {
            # Model card characteristics
            "content_length": len(str(model_card)),
            "section_count": self._count_sections(model_card),
            "has_metrics": bool(getattr(model_card, 'metrics', None)),
            "has_ethical_considerations": bool(getattr(model_card, 'ethical_considerations', None)),
            "has_limitations": bool(getattr(model_card, 'limitations', None)),
            
            # Context features
            "generation_time": context.get("generation_time", 0),
            "data_sources": len(context.get("data_sources", [])),
            "format_type": context.get("format", "unknown"),
            "use_case": context.get("use_case", "general"),
            
            # Quality features
            "avg_quality_score": np.mean([r.score for r in quality_results]),
            "gates_passed": sum(1 for r in quality_results if r.passed),
            "total_gates": len(quality_results),
            "execution_time": sum(r.execution_time for r in quality_results),
            
            # Advanced features
            "complexity_score": self._calculate_complexity(model_card),
            "completeness_score": self._calculate_completeness(model_card),
            "innovation_score": self._calculate_innovation(model_card, context)
        }
        
        return features
    
    def _count_sections(self, model_card: ModelCard) -> int:
        """Count the number of sections in a model card."""
        sections = [
            'model_details', 'intended_use', 'factors', 'metrics',
            'evaluation_data', 'training_data', 'quantitative_analyses',
            'ethical_considerations', 'recommendations', 'limitations'
        ]
        
        return sum(1 for section in sections 
                  if hasattr(model_card, section) and getattr(model_card, section))
    
    def _calculate_complexity(self, model_card: ModelCard) -> float:
        """Calculate the complexity score of a model card."""
        content = str(model_card)
        
        # Basic complexity metrics
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))
        avg_word_length = np.mean([len(word) for word in content.split()])
        
        # Technical complexity indicators
        technical_terms = ['algorithm', 'neural', 'transformer', 'attention', 'embedding',
                          'gradient', 'optimization', 'regularization', 'hyperparameter']
        tech_term_count = sum(1 for term in technical_terms if term in content.lower())
        
        # Normalize to 0-1 scale
        complexity_score = min(1.0, (
            (word_count / 1000) * 0.3 +
            (unique_words / word_count if word_count > 0 else 0) * 0.3 +
            (avg_word_length / 10) * 0.2 +
            (tech_term_count / len(technical_terms)) * 0.2
        ))
        
        return complexity_score
    
    def _calculate_completeness(self, model_card: ModelCard) -> float:
        """Calculate completeness score."""
        required_fields = [
            'model_details', 'intended_use', 'factors', 'metrics',
            'evaluation_data', 'training_data', 'quantitative_analyses',
            'ethical_considerations', 'recommendations'
        ]
        
        completed = sum(1 for field in required_fields 
                       if hasattr(model_card, field) and getattr(model_card, field))
        
        return completed / len(required_fields)
    
    def _calculate_innovation(self, model_card: ModelCard, context: Dict[str, Any]) -> float:
        """Calculate innovation score based on novel approaches or techniques."""
        content = str(model_card).lower()
        context_str = str(context).lower()
        
        # Innovation indicators
        innovation_terms = [
            'novel', 'breakthrough', 'state-of-the-art', 'cutting-edge',
            'innovative', 'pioneering', 'advanced', 'next-generation',
            'transformer', 'attention', 'self-supervised', 'few-shot',
            'zero-shot', 'multimodal', 'federated', 'meta-learning'
        ]
        
        innovation_count = sum(1 for term in innovation_terms 
                             if term in content or term in context_str)
        
        return min(1.0, innovation_count / len(innovation_terms))
    
    def _calculate_overall_quality(self, quality_results: List[QualityGateResult]) -> float:
        """Calculate overall quality score from quality gate results."""
        if not quality_results:
            return 0.0
            
        # Weight by gate importance
        gate_weights = {
            "completeness": 2.0,
            "accuracy": 2.5,
            "security": 3.0,
            "performance": 1.5,
            "compliance": 2.0,
            "maintainability": 1.0
        }
        
        weighted_scores = []
        for result in quality_results:
            weight = gate_weights.get(result.gate_name, 1.0)
            weighted_scores.append(result.score * weight)
            
        return sum(weighted_scores) / sum(gate_weights.values()) if weighted_scores else 0.0
    
    async def _discover_patterns(
        self,
        features: Dict[str, Any],
        quality: float,
        quality_results: List[QualityGateResult]
    ) -> None:
        """Discover and update patterns from the current generation."""
        
        # Create pattern signature
        pattern_key = self._create_pattern_key(features, quality > 0.8)
        
        # Determine pattern type
        pattern_type = "success" if quality > 0.8 else "failure"
        if quality > 0.9:
            pattern_type = "optimization"
        
        # Update or create pattern
        if pattern_key in self.patterns:
            pattern = self.patterns[pattern_key]
            pattern.update_effectiveness(quality)
        else:
            pattern = LearningPattern(
                pattern_id=pattern_key,
                pattern_type=pattern_type,
                context=features.copy(),
                outcome_metrics={
                    "quality": quality,
                    "gates_passed": features.get("gates_passed", 0),
                    "execution_time": features.get("execution_time", 0)
                },
                frequency=1,
                effectiveness=quality
            )
            self.patterns[pattern_key] = pattern
            self.learning_metrics["patterns_discovered"] += 1
        
        # Add to sequence for temporal pattern analysis
        self.pattern_sequences.append({
            "pattern_key": pattern_key,
            "timestamp": datetime.now(),
            "quality": quality
        })
        
        # Prune old patterns if we exceed max_patterns
        if len(self.patterns) > self.max_patterns:
            await self._prune_patterns()
    
    def _create_pattern_key(self, features: Dict[str, Any], is_success: bool) -> str:
        """Create a unique key for a pattern based on significant features."""
        
        # Select most significant features for pattern key
        significant_features = {
            "format_type": features.get("format_type", "unknown"),
            "use_case": features.get("use_case", "general"),
            "has_metrics": features.get("has_metrics", False),
            "has_ethical": features.get("has_ethical_considerations", False),
            "complexity_tier": "high" if features.get("complexity_score", 0) > 0.7 else "low",
            "completeness_tier": "high" if features.get("completeness_score", 0) > 0.8 else "low",
            "outcome": "success" if is_success else "failure"
        }
        
        # Create deterministic key
        key_parts = [f"{k}:{v}" for k, v in sorted(significant_features.items())]
        return "|".join(key_parts)
    
    def _update_feature_weights(self, features: Dict[str, Any], quality: float) -> None:
        """Update feature weights based on their correlation with quality."""
        
        # Simple adaptive weighting
        learning_rate = 0.01
        
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)):
                # Update weight based on quality correlation
                correlation = quality * (feature_value / (abs(feature_value) + 1e-6))
                
                current_weight = self.feature_weights[feature_name]
                self.feature_weights[feature_name] = current_weight + (learning_rate * correlation)
                
                # Clamp weights to reasonable bounds
                self.feature_weights[feature_name] = max(0.1, min(5.0, self.feature_weights[feature_name]))
    
    async def _generate_adaptive_recommendations(
        self,
        features: Dict[str, Any],
        quality_results: List[QualityGateResult]
    ) -> List[AdaptiveRecommendation]:
        """Generate adaptive recommendations based on learned patterns."""
        
        recommendations = []
        
        # Pattern-based recommendations
        similar_patterns = self._find_similar_patterns(features)
        for pattern in similar_patterns[:3]:  # Top 3 similar patterns
            if pattern.pattern_type == "success" and pattern.confidence > 0.6:
                rec = AdaptiveRecommendation(
                    recommendation_id=f"pattern_{pattern.pattern_id[:8]}",
                    description=f"Apply successful pattern: {self._describe_pattern(pattern)}",
                    category="pattern_optimization",
                    confidence=pattern.confidence,
                    expected_improvement=pattern.effectiveness * 0.1,
                    implementation_cost="medium",
                    learned_from_pattern=pattern.pattern_id
                )
                recommendations.append(rec)
        
        # Quality-based recommendations
        for result in quality_results:
            if not result.passed and result.suggestions:
                for suggestion in result.suggestions:
                    rec = AdaptiveRecommendation(
                        recommendation_id=f"quality_{result.gate_name}_{len(recommendations)}",
                        description=suggestion,
                        category=f"{result.gate_name}_improvement",
                        confidence=0.8,
                        expected_improvement=0.05,
                        implementation_cost="low",
                        learned_from_pattern="quality_gate"
                    )
                    recommendations.append(rec)
        
        # Feature-based recommendations
        feature_recs = self._generate_feature_recommendations(features)
        recommendations.extend(feature_recs)
        
        # Sort by expected impact
        recommendations.sort(key=lambda r: r.expected_improvement * r.confidence, reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _find_similar_patterns(self, features: Dict[str, Any]) -> List[LearningPattern]:
        """Find patterns similar to current features."""
        
        similar_patterns = []
        
        for pattern in self.patterns.values():
            similarity = self._calculate_pattern_similarity(features, pattern.context)
            if similarity > 0.7:  # Similarity threshold
                pattern_copy = pattern
                pattern_copy.confidence *= similarity  # Adjust confidence by similarity
                similar_patterns.append(pattern_copy)
        
        # Sort by confidence
        similar_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return similar_patterns
    
    def _calculate_pattern_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets."""
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            if isinstance(val1, bool) and isinstance(val2, bool):
                similarities.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, str) and isinstance(val2, str):
                similarities.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalized similarity for numeric values
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarities.append(1.0 - abs(val1 - val2) / max_val)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _describe_pattern(self, pattern: LearningPattern) -> str:
        """Generate human-readable description of a pattern."""
        
        context = pattern.context
        parts = []
        
        if context.get("format_type") != "unknown":
            parts.append(f"Use {context['format_type']} format")
        
        if context.get("has_metrics"):
            parts.append("Include comprehensive metrics")
        
        if context.get("has_ethical_considerations"):
            parts.append("Add ethical considerations")
        
        if context.get("complexity_tier") == "high":
            parts.append("Maintain high complexity")
        
        return ", ".join(parts) if parts else "Apply learned optimizations"
    
    def _generate_feature_recommendations(self, features: Dict[str, Any]) -> List[AdaptiveRecommendation]:
        """Generate recommendations based on feature analysis."""
        
        recommendations = []
        
        # Completeness recommendations
        if features.get("completeness_score", 0) < 0.8:
            recommendations.append(AdaptiveRecommendation(
                recommendation_id="completeness_improvement",
                description="Improve model card completeness by adding missing sections",
                category="completeness",
                confidence=0.9,
                expected_improvement=0.15,
                implementation_cost="medium",
                learned_from_pattern="feature_analysis"
            ))
        
        # Complexity recommendations
        if features.get("complexity_score", 0) < 0.3:
            recommendations.append(AdaptiveRecommendation(
                recommendation_id="complexity_enhancement",
                description="Add more technical detail and complexity to improve comprehensiveness",
                category="complexity",
                confidence=0.7,
                expected_improvement=0.08,
                implementation_cost="high",
                learned_from_pattern="feature_analysis"
            ))
        
        # Performance recommendations
        if features.get("execution_time", 0) > 5.0:
            recommendations.append(AdaptiveRecommendation(
                recommendation_id="performance_optimization",
                description="Optimize generation process to reduce execution time",
                category="performance",
                confidence=0.8,
                expected_improvement=0.05,
                implementation_cost="medium",
                learned_from_pattern="feature_analysis"
            ))
        
        return recommendations
    
    async def _optimize_learning_system(self) -> None:
        """Optimize the learning system based on accumulated data."""
        
        self.logger.info("Optimizing learning system...")
        
        # Prune ineffective patterns
        await self._prune_patterns()
        
        # Optimize feature weights
        self._optimize_feature_weights()
        
        # Update learning strategies
        self._update_optimization_strategies()
        
        # Save learning state
        self._save_learning_state()
        
        self.learning_metrics["adaptation_count"] += 1
        self.logger.info("Learning system optimization completed")
    
    async def _prune_patterns(self) -> None:
        """Remove old or ineffective patterns."""
        
        # Remove patterns older than 30 days with low confidence
        cutoff_date = datetime.now() - timedelta(days=30)
        
        patterns_to_remove = []
        for pattern_id, pattern in self.patterns.items():
            if (pattern.last_seen < cutoff_date and 
                pattern.confidence < 0.3 and 
                pattern.frequency < 5):
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
        
        self.logger.info(f"Pruned {len(patterns_to_remove)} ineffective patterns")
    
    def _optimize_feature_weights(self) -> None:
        """Optimize feature weights based on historical performance."""
        
        if len(self.outcome_history) < 50:
            return  # Not enough data
        
        # Calculate feature-outcome correlations
        feature_outcomes = defaultdict(list)
        
        for outcome in list(self.outcome_history)[-100:]:  # Last 100 outcomes
            quality = outcome["quality"]
            for feature_name, feature_value in outcome["features"].items():
                if isinstance(feature_value, (int, float)):
                    feature_outcomes[feature_name].append((feature_value, quality))
        
        # Update weights based on correlations
        for feature_name, value_quality_pairs in feature_outcomes.items():
            if len(value_quality_pairs) >= 10:
                values, qualities = zip(*value_quality_pairs)
                correlation = np.corrcoef(values, qualities)[0, 1]
                
                if not np.isnan(correlation):
                    # Adjust weight based on correlation strength
                    weight_adjustment = correlation * 0.1
                    self.feature_weights[feature_name] += weight_adjustment
                    self.feature_weights[feature_name] = max(0.1, min(5.0, self.feature_weights[feature_name]))
    
    def _update_optimization_strategies(self) -> None:
        """Update optimization strategies based on learning."""
        
        # Analyze successful patterns to derive strategies
        successful_patterns = [p for p in self.patterns.values() 
                             if p.pattern_type in ["success", "optimization"] and p.confidence > 0.7]
        
        # Extract common success factors
        strategy_features = defaultdict(int)
        for pattern in successful_patterns:
            for feature, value in pattern.context.items():
                if isinstance(value, bool) and value:
                    strategy_features[feature] += pattern.frequency
                elif isinstance(value, str):
                    strategy_features[f"{feature}_{value}"] += pattern.frequency
        
        # Create optimization strategies
        self.optimization_strategies = [
            {
                "name": feature,
                "weight": count / len(successful_patterns),
                "description": f"Optimize for {feature.replace('_', ' ')}"
            }
            for feature, count in strategy_features.items()
            if count >= len(successful_patterns) * 0.3  # At least 30% of successful patterns
        ]
        
        # Sort by weight
        self.optimization_strategies.sort(key=lambda s: s["weight"], reverse=True)
    
    def _update_learning_metrics(self, quality: float, recommendation_count: int) -> None:
        """Update learning system metrics."""
        
        self.learning_metrics["total_predictions"] += 1
        
        if quality > 0.8:
            self.learning_metrics["successful_predictions"] += 1
        
        # Update average improvement (simplified)
        current_avg = self.learning_metrics["avg_improvement"]
        new_avg = (current_avg * 0.9) + (quality * 0.1)
        self.learning_metrics["avg_improvement"] = new_avg
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the learning system's current state."""
        
        prediction_accuracy = (
            self.learning_metrics["successful_predictions"] / 
            max(self.learning_metrics["total_predictions"], 1)
        )
        
        return {
            "patterns_discovered": len(self.patterns),
            "prediction_accuracy": prediction_accuracy,
            "adaptation_count": self.learning_metrics["adaptation_count"],
            "average_improvement": self.learning_metrics["avg_improvement"],
            "top_features": dict(sorted(self.feature_weights.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]),
            "optimization_strategies": self.optimization_strategies[:5],
            "learning_health": "excellent" if prediction_accuracy > 0.8 else 
                             "good" if prediction_accuracy > 0.6 else "needs_improvement"
        }
    
    def _save_learning_state(self) -> None:
        """Save the current learning state to disk."""
        
        state_file = self.learning_dir / "learning_state.pkl"
        
        state = {
            "patterns": self.patterns,
            "feature_weights": dict(self.feature_weights),
            "optimization_strategies": self.optimization_strategies,
            "learning_metrics": self.learning_metrics,
            "timestamp": datetime.now()
        }
        
        try:
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            self.logger.error(f"Failed to save learning state: {e}")
    
    def _load_learning_state(self) -> None:
        """Load learning state from disk."""
        
        state_file = self.learning_dir / "learning_state.pkl"
        
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            self.patterns = state.get("patterns", {})
            self.feature_weights = defaultdict(lambda: 1.0, state.get("feature_weights", {}))
            self.optimization_strategies = state.get("optimization_strategies", [])
            self.learning_metrics = state.get("learning_metrics", self.learning_metrics)
            
            self.logger.info(f"Loaded learning state with {len(self.patterns)} patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")


# Factory function
def create_autonomous_learning_system(learning_dir: Optional[Path] = None) -> AutonomousLearningSystem:
    """Create an autonomous learning system with default configuration."""
    return AutonomousLearningSystem(learning_dir=learning_dir)