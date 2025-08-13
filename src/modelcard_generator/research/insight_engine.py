"""Intelligent insight generation engine for model card analysis."""

import asyncio
import json
import math
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.logging_config import get_logger
from ..core.models import ModelCard, PerformanceMetric

logger = get_logger(__name__)


@dataclass
class Insight:
    """A generated insight from model card analysis."""
    title: str
    description: str
    insight_type: str  # "performance", "trend", "anomaly", "opportunity", "risk"
    confidence: float  # 0.0 to 1.0
    impact_level: str  # "low", "medium", "high", "critical"
    evidence: List[str]
    actionable_recommendations: List[str]
    affected_stakeholders: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightCluster:
    """A cluster of related insights."""
    theme: str
    insights: List[Insight]
    cluster_confidence: float
    priority_score: float
    summary: str


class InsightEngine:
    """Advanced insight generation engine for model card analysis."""

    def __init__(self, insight_threshold: float = 0.6):
        self.insight_threshold = insight_threshold
        self.insight_history: List[Insight] = []
        self.pattern_database: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        
    async def generate_comprehensive_insights(
        self, 
        model_cards: List[ModelCard],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive insights from model card analysis."""
        
        try:
            logger.info(f"Generating insights from {len(model_cards)} model cards")
            
            context = context or {}
            
            # Initialize insight collection
            all_insights = []
            
            # 1. Performance insights
            performance_insights = await self._generate_performance_insights(model_cards, context)
            all_insights.extend(performance_insights)
            
            # 2. Trend insights
            trend_insights = await self._generate_trend_insights(model_cards, context)
            all_insights.extend(trend_insights)
            
            # 3. Anomaly detection insights
            anomaly_insights = await self._detect_anomaly_insights(model_cards, context)
            all_insights.extend(anomaly_insights)
            
            # 4. Risk assessment insights
            risk_insights = await self._generate_risk_insights(model_cards, context)
            all_insights.extend(risk_insights)
            
            # 5. Opportunity insights
            opportunity_insights = await self._generate_opportunity_insights(model_cards, context)
            all_insights.extend(opportunity_insights)
            
            # 6. Quality insights
            quality_insights = await self._generate_quality_insights(model_cards, context)
            all_insights.extend(quality_insights)
            
            # Filter by confidence threshold
            high_confidence_insights = [
                insight for insight in all_insights 
                if insight.confidence >= self.insight_threshold
            ]
            
            # Cluster related insights
            insight_clusters = self._cluster_insights(high_confidence_insights)
            
            # Prioritize insights
            prioritized_insights = self._prioritize_insights(high_confidence_insights)
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(insight_clusters, prioritized_insights)
            
            # Prepare comprehensive results
            results = {
                "total_insights_generated": len(all_insights),
                "high_confidence_insights": len(high_confidence_insights),
                "insight_clusters": [cluster.__dict__ for cluster in insight_clusters],
                "prioritized_insights": [insight.__dict__ for insight in prioritized_insights[:20]],  # Top 20
                "executive_summary": executive_summary,
                "insights_by_type": self._categorize_insights_by_type(high_confidence_insights),
                "insights_by_impact": self._categorize_insights_by_impact(high_confidence_insights),
                "actionable_recommendations": self._extract_actionable_recommendations(high_confidence_insights),
                "stakeholder_impact_analysis": self._analyze_stakeholder_impact(high_confidence_insights),
                "metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_cards_analyzed": len(model_cards),
                    "confidence_threshold": self.insight_threshold,
                    "processing_time": "async"
                }
            }
            
            # Store insights for future reference
            self.insight_history.extend(high_confidence_insights)
            
            logger.info(f"Generated {len(high_confidence_insights)} high-confidence insights")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive insight generation failed: {e}")
            raise

    async def _generate_performance_insights(self, model_cards: List[ModelCard], context: Dict[str, Any]) -> List[Insight]:
        """Generate insights about model performance patterns."""
        insights = []
        
        try:
            # Collect performance metrics
            all_metrics = defaultdict(list)
            for card in model_cards:
                for metric in card.evaluation_results:
                    all_metrics[metric.name].append((card.model_details.name, metric.value))
            
            # Analyze each metric
            for metric_name, model_values in all_metrics.items():
                if len(model_values) < 2:
                    continue
                
                values = [v for _, v in model_values]
                models = [m for m, _ in model_values]
                
                # Statistical analysis
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                min_val, max_val = min(values), max(values)
                
                # Performance distribution insight
                if std_val > 0.1 and mean_val > 0:  # High variance
                    high_performers = [models[i] for i, v in enumerate(values) if v > mean_val + std_val]
                    low_performers = [models[i] for i, v in enumerate(values) if v < mean_val - std_val]
                    
                    if high_performers and low_performers:
                        insights.append(Insight(
                            title=f"Performance Disparity in {metric_name.title()}",
                            description=f"Significant variation in {metric_name} across models (std: {std_val:.3f})",
                            insight_type="performance",
                            confidence=0.8,
                            impact_level="high",
                            evidence=[
                                f"Standard deviation: {std_val:.3f}",
                                f"Range: {min_val:.3f} - {max_val:.3f}",
                                f"High performers: {', '.join(high_performers[:3])}"
                            ],
                            actionable_recommendations=[
                                "Analyze high-performing models for best practices",
                                "Identify factors causing performance variation",
                                "Standardize training methodologies"
                            ],
                            affected_stakeholders=["ML Engineers", "Data Scientists", "Product Managers"],
                            metadata={"metric": metric_name, "variance": std_val}
                        ))
                
                # Performance ceiling insight
                if max_val > 0.95:
                    top_model = models[values.index(max_val)]
                    insights.append(Insight(
                        title=f"Exceptional {metric_name.title()} Achievement",
                        description=f"Model '{top_model}' achieves exceptional {metric_name}: {max_val:.4f}",
                        insight_type="performance",
                        confidence=0.9,
                        impact_level="high",
                        evidence=[
                            f"Peak performance: {max_val:.4f}",
                            f"Above 95th percentile threshold",
                            f"Model: {top_model}"
                        ],
                        actionable_recommendations=[
                            "Study model architecture and training approach",
                            "Consider as baseline for future development",
                            "Validate performance on additional test sets"
                        ],
                        affected_stakeholders=["Research Team", "ML Engineers", "Business Stakeholders"],
                        metadata={"metric": metric_name, "top_model": top_model, "peak_value": max_val}
                    ))
                
                # Performance floor insight
                if min_val < 0.5 and mean_val > 0.7:
                    weak_model = models[values.index(min_val)]
                    insights.append(Insight(
                        title=f"Performance Concern in {metric_name.title()}",
                        description=f"Model '{weak_model}' shows concerning {metric_name}: {min_val:.4f}",
                        insight_type="risk",
                        confidence=0.85,
                        impact_level="medium",
                        evidence=[
                            f"Low performance: {min_val:.4f}",
                            f"Below team average: {mean_val:.3f}",
                            f"Model: {weak_model}"
                        ],
                        actionable_recommendations=[
                            "Investigate root cause of poor performance",
                            "Consider model architecture review",
                            "Evaluate data quality and preprocessing"
                        ],
                        affected_stakeholders=["ML Engineers", "QA Team", "Product Managers"],
                        metadata={"metric": metric_name, "weak_model": weak_model, "low_value": min_val}
                    ))
            
            # Cross-metric performance consistency
            consistency_scores = {}
            for card in model_cards:
                if len(card.evaluation_results) > 1:
                    card_values = [m.value for m in card.evaluation_results]
                    consistency = 1 - (statistics.stdev(card_values) / statistics.mean(card_values)) if statistics.mean(card_values) > 0 else 0
                    consistency_scores[card.model_details.name] = consistency
            
            if consistency_scores:
                most_consistent = max(consistency_scores.items(), key=lambda x: x[1])
                least_consistent = min(consistency_scores.items(), key=lambda x: x[1])
                
                if most_consistent[1] > 0.8:
                    insights.append(Insight(
                        title="Exceptional Performance Consistency",
                        description=f"Model '{most_consistent[0]}' shows remarkable consistency across metrics",
                        insight_type="performance",
                        confidence=0.85,
                        impact_level="medium",
                        evidence=[
                            f"Consistency score: {most_consistent[1]:.3f}",
                            "Low variance across evaluation metrics",
                            f"Model: {most_consistent[0]}"
                        ],
                        actionable_recommendations=[
                            "Study training stability techniques",
                            "Consider as template for reliable model development",
                            "Analyze regularization and optimization strategies"
                        ],
                        affected_stakeholders=["ML Engineers", "Research Team"],
                        metadata={"consistency_champion": most_consistent[0], "score": most_consistent[1]}
                    ))
                
                if least_consistent[1] < 0.5 and most_consistent[1] - least_consistent[1] > 0.3:
                    insights.append(Insight(
                        title="Performance Inconsistency Alert",
                        description=f"Model '{least_consistent[0]}' shows inconsistent performance across metrics",
                        insight_type="risk",
                        confidence=0.75,
                        impact_level="medium",
                        evidence=[
                            f"Consistency score: {least_consistent[1]:.3f}",
                            "High variance across evaluation metrics",
                            f"Compared to most consistent: {most_consistent[1]:.3f}"
                        ],
                        actionable_recommendations=[
                            "Review model architecture for stability",
                            "Investigate training process irregularities",
                            "Consider additional regularization techniques"
                        ],
                        affected_stakeholders=["ML Engineers", "QA Team"],
                        metadata={"inconsistent_model": least_consistent[0], "score": least_consistent[1]}
                    ))
                        
        except Exception as e:
            logger.warning(f"Performance insight generation failed: {e}")
        
        return insights

    async def _generate_trend_insights(self, model_cards: List[ModelCard], context: Dict[str, Any]) -> List[Insight]:
        """Generate insights about trends over time."""
        insights = []
        
        try:
            if len(model_cards) < 3:
                return insights
            
            # Sort by creation time
            sorted_cards = sorted(model_cards, key=lambda x: x.created_at)
            
            # Analyze metric trends over time
            metric_timelines = defaultdict(list)
            for card in sorted_cards:
                for metric in card.evaluation_results:
                    metric_timelines[metric.name].append((card.created_at, metric.value))
            
            for metric_name, timeline in metric_timelines.items():
                if len(timeline) < 3:
                    continue
                
                # Calculate trend
                times = [(t - timeline[0][0]).days for t, _ in timeline]
                values = [v for _, v in timeline]
                
                if len(set(times)) > 1:  # Ensure time variation
                    trend_slope = self._calculate_simple_slope(times, values)
                    
                    # Improvement trend
                    if trend_slope > 0.001:  # Positive trend
                        time_span = (timeline[-1][0] - timeline[0][0]).days
                        improvement_rate = trend_slope * 30  # Monthly improvement
                        
                        insights.append(Insight(
                            title=f"Positive {metric_name.title()} Trend",
                            description=f"{metric_name} shows consistent improvement over {time_span} days",
                            insight_type="trend",
                            confidence=0.8,
                            impact_level="medium",
                            evidence=[
                                f"Improvement rate: {improvement_rate:.4f} per month",
                                f"Time span: {time_span} days",
                                f"Recent value: {values[-1]:.3f}"
                            ],
                            actionable_recommendations=[
                                "Continue current development practices",
                                "Document successful strategies",
                                "Set higher performance targets"
                            ],
                            affected_stakeholders=["ML Team", "Product Managers", "Leadership"],
                            metadata={"metric": metric_name, "trend_slope": trend_slope, "time_span": time_span}
                        ))
                    
                    # Declining trend
                    elif trend_slope < -0.001:  # Negative trend
                        time_span = (timeline[-1][0] - timeline[0][0]).days
                        decline_rate = abs(trend_slope) * 30  # Monthly decline
                        
                        insights.append(Insight(
                            title=f"Declining {metric_name.title()} Trend",
                            description=f"{metric_name} shows concerning decline over {time_span} days",
                            insight_type="risk",
                            confidence=0.85,
                            impact_level="high",
                            evidence=[
                                f"Decline rate: {decline_rate:.4f} per month",
                                f"Time span: {time_span} days",
                                f"Current value: {values[-1]:.3f}"
                            ],
                            actionable_recommendations=[
                                "Investigate causes of performance decline",
                                "Review recent changes in methodology",
                                "Consider reverting to previous successful approaches"
                            ],
                            affected_stakeholders=["ML Team", "QA Team", "Leadership"],
                            metadata={"metric": metric_name, "trend_slope": trend_slope, "time_span": time_span}
                        ))
            
            # Development velocity insights
            time_gaps = []
            for i in range(1, len(sorted_cards)):
                gap = (sorted_cards[i].created_at - sorted_cards[i-1].created_at).days
                time_gaps.append(gap)
            
            if time_gaps:
                avg_gap = statistics.mean(time_gaps)
                
                if avg_gap < 7:  # Very frequent development
                    insights.append(Insight(
                        title="Rapid Development Velocity",
                        description=f"High development frequency with avg {avg_gap:.1f} days between models",
                        insight_type="trend",
                        confidence=0.9,
                        impact_level="medium",
                        evidence=[
                            f"Average development gap: {avg_gap:.1f} days",
                            f"Total models: {len(sorted_cards)}",
                            "Rapid iteration cycle"
                        ],
                        actionable_recommendations=[
                            "Ensure quality control processes scale",
                            "Consider automated testing pipelines",
                            "Monitor for development burnout"
                        ],
                        affected_stakeholders=["Development Team", "QA Team", "Management"],
                        metadata={"avg_development_gap": avg_gap, "velocity": "high"}
                    ))
                
                elif avg_gap > 30:  # Slow development
                    insights.append(Insight(
                        title="Slow Development Velocity",
                        description=f"Extended development cycles with avg {avg_gap:.1f} days between models",
                        insight_type="opportunity",
                        confidence=0.8,
                        impact_level="medium",
                        evidence=[
                            f"Average development gap: {avg_gap:.1f} days",
                            f"Total models: {len(sorted_cards)}",
                            "Extended iteration cycles"
                        ],
                        actionable_recommendations=[
                            "Identify bottlenecks in development process",
                            "Consider parallel development streams",
                            "Implement more efficient testing procedures"
                        ],
                        affected_stakeholders=["Development Team", "Management", "Product Team"],
                        metadata={"avg_development_gap": avg_gap, "velocity": "low"}
                    ))
                        
        except Exception as e:
            logger.warning(f"Trend insight generation failed: {e}")
        
        return insights

    async def _detect_anomaly_insights(self, model_cards: List[ModelCard], context: Dict[str, Any]) -> List[Insight]:
        """Detect anomalous patterns in model cards."""
        insights = []
        
        try:
            # Metric value anomalies
            all_metrics = defaultdict(list)
            for card in model_cards:
                for metric in card.evaluation_results:
                    all_metrics[metric.name].append((card.model_details.name, metric.value))
            
            for metric_name, model_values in all_metrics.items():
                if len(model_values) < 3:
                    continue
                
                values = [v for _, v in model_values]
                models = [m for m, _ in model_values]
                
                # Statistical outlier detection using IQR method
                q1 = statistics.quantiles(values, n=4)[0] if len(values) >= 4 else min(values)
                q3 = statistics.quantiles(values, n=4)[2] if len(values) >= 4 else max(values)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [(models[i], v) for i, v in enumerate(values) 
                           if v < lower_bound or v > upper_bound]
                
                for model_name, outlier_value in outliers:
                    is_positive_outlier = outlier_value > upper_bound
                    
                    insights.append(Insight(
                        title=f"{'Exceptional' if is_positive_outlier else 'Anomalous'} {metric_name.title()} Value",
                        description=f"Model '{model_name}' shows {'exceptional' if is_positive_outlier else 'anomalous'} {metric_name}: {outlier_value:.4f}",
                        insight_type="anomaly",
                        confidence=0.85,
                        impact_level="high" if is_positive_outlier else "medium",
                        evidence=[
                            f"Value: {outlier_value:.4f}",
                            f"Expected range: {lower_bound:.3f} - {upper_bound:.3f}",
                            f"IQR: {iqr:.3f}"
                        ],
                        actionable_recommendations=[
                            "Investigate model architecture and training process" if is_positive_outlier else "Review data quality and preprocessing",
                            "Validate results on additional test sets",
                            "Document unique characteristics" if is_positive_outlier else "Consider retraining with improved data"
                        ],
                        affected_stakeholders=["ML Engineers", "Research Team", "QA Team"],
                        metadata={
                            "metric": metric_name, 
                            "model": model_name, 
                            "value": outlier_value,
                            "outlier_type": "positive" if is_positive_outlier else "negative"
                        }
                    ))
            
            # Architecture pattern anomalies
            architectures = [card.training_details.model_architecture or "unknown" 
                           for card in model_cards]
            arch_counter = Counter(architectures)
            
            # Find rare architectures
            rare_architectures = {arch: count for arch, count in arch_counter.items() 
                                if count == 1 and len(model_cards) > 3}
            
            for rare_arch in rare_architectures:
                if rare_arch != "unknown":
                    rare_models = [card.model_details.name for card in model_cards 
                                 if card.training_details.model_architecture == rare_arch]
                    
                    insights.append(Insight(
                        title="Unique Architecture Pattern",
                        description=f"Model uses rare architecture: {rare_arch}",
                        insight_type="anomaly",
                        confidence=0.7,
                        impact_level="medium",
                        evidence=[
                            f"Architecture: {rare_arch}",
                            f"Model(s): {', '.join(rare_models)}",
                            "No other models use this architecture"
                        ],
                        actionable_recommendations=[
                            "Document architectural choices and rationale",
                            "Evaluate performance compared to standard architectures",
                            "Consider knowledge transfer to team"
                        ],
                        affected_stakeholders=["Research Team", "ML Engineers"],
                        metadata={"architecture": rare_arch, "models": rare_models}
                    ))
            
            # Training time anomalies
            training_times = []
            for card in model_cards:
                if card.training_details.training_time:
                    try:
                        # Extract numeric training time (assuming it's documented)
                        time_str = str(card.training_details.training_time).lower()
                        if "hours" in time_str or "hrs" in time_str:
                            hours = float(''.join(filter(str.isdigit, time_str.split('h')[0])))
                            training_times.append((card.model_details.name, hours))
                        elif "minutes" in time_str or "mins" in time_str:
                            minutes = float(''.join(filter(str.isdigit, time_str.split('m')[0])))
                            training_times.append((card.model_details.name, minutes / 60))
                    except:
                        pass
            
            if len(training_times) > 2:
                times = [t for _, t in training_times]
                mean_time = statistics.mean(times)
                std_time = statistics.stdev(times) if len(times) > 1 else 0
                
                for model_name, time_val in training_times:
                    if abs(time_val - mean_time) > 2 * std_time and std_time > 0:
                        is_long_training = time_val > mean_time
                        
                        insights.append(Insight(
                            title=f"{'Extended' if is_long_training else 'Rapid'} Training Duration",
                            description=f"Model '{model_name}' has {'unusually long' if is_long_training else 'remarkably short'} training time: {time_val:.1f} hours",
                            insight_type="anomaly",
                            confidence=0.75,
                            impact_level="medium",
                            evidence=[
                                f"Training time: {time_val:.1f} hours",
                                f"Team average: {mean_time:.1f} hours",
                                f"Standard deviation: {std_time:.1f} hours"
                            ],
                            actionable_recommendations=[
                                "Analyze training efficiency factors" if is_long_training else "Document efficient training techniques",
                                "Review computational resources and optimization",
                                "Consider as baseline for future training" if not is_long_training else "Investigate potential optimization opportunities"
                            ],
                            affected_stakeholders=["ML Engineers", "Infrastructure Team"],
                            metadata={"model": model_name, "training_time": time_val, "type": "extended" if is_long_training else "rapid"}
                        ))
                        
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
        
        return insights

    async def _generate_risk_insights(self, model_cards: List[ModelCard], context: Dict[str, Any]) -> List[Insight]:
        """Generate insights about potential risks."""
        insights = []
        
        try:
            for card in model_cards:
                model_name = card.model_details.name
                
                # Missing documentation risks
                documentation_gaps = []
                if not card.model_details.description:
                    documentation_gaps.append("model description")
                if not card.intended_use:
                    documentation_gaps.append("intended use")
                if not card.limitations.known_limitations:
                    documentation_gaps.append("limitations")
                if not card.evaluation_results:
                    documentation_gaps.append("evaluation metrics")
                
                if len(documentation_gaps) > 2:
                    insights.append(Insight(
                        title=f"Documentation Completeness Risk - {model_name}",
                        description=f"Model has significant documentation gaps affecting deployment readiness",
                        insight_type="risk",
                        confidence=0.9,
                        impact_level="high",
                        evidence=[
                            f"Missing: {', '.join(documentation_gaps)}",
                            f"Total gaps: {len(documentation_gaps)}",
                            "Insufficient for production deployment"
                        ],
                        actionable_recommendations=[
                            "Complete missing documentation sections",
                            "Establish documentation review process",
                            "Delay deployment until documentation complete"
                        ],
                        affected_stakeholders=["ML Engineers", "Compliance Team", "Product Managers"],
                        metadata={"model": model_name, "gaps": documentation_gaps}
                    ))
                
                # Ethical considerations risks
                ethical_gaps = []
                if not card.ethical_considerations.bias_risks:
                    ethical_gaps.append("bias risk analysis")
                if not card.ethical_considerations.fairness_metrics:
                    ethical_gaps.append("fairness metrics")
                if not card.ethical_considerations.bias_mitigation:
                    ethical_gaps.append("bias mitigation strategies")
                
                if len(ethical_gaps) > 1:
                    insights.append(Insight(
                        title=f"Ethical AI Risk - {model_name}",
                        description=f"Model lacks comprehensive ethical considerations",
                        insight_type="risk",
                        confidence=0.85,
                        impact_level="high",
                        evidence=[
                            f"Missing ethical analysis: {', '.join(ethical_gaps)}",
                            "Potential regulatory compliance issues",
                            "Reputational risk"
                        ],
                        actionable_recommendations=[
                            "Conduct comprehensive bias analysis",
                            "Implement fairness testing",
                            "Develop bias mitigation strategies"
                        ],
                        affected_stakeholders=["Ethics Committee", "Legal Team", "ML Engineers"],
                        metadata={"model": model_name, "ethical_gaps": ethical_gaps}
                    ))
                
                # Performance volatility risk
                if len(card.evaluation_results) > 2:
                    metric_values = [m.value for m in card.evaluation_results]
                    cv = statistics.stdev(metric_values) / statistics.mean(metric_values) if statistics.mean(metric_values) > 0 else 0
                    
                    if cv > 0.3:  # High coefficient of variation
                        insights.append(Insight(
                            title=f"Performance Volatility Risk - {model_name}",
                            description=f"Model shows high performance variance across metrics",
                            insight_type="risk",
                            confidence=0.8,
                            impact_level="medium",
                            evidence=[
                                f"Coefficient of variation: {cv:.3f}",
                                f"Metric count: {len(metric_values)}",
                                "Inconsistent performance indicators"
                            ],
                            actionable_recommendations=[
                                "Investigate training stability",
                                "Review evaluation methodology",
                                "Consider ensemble approaches for reliability"
                            ],
                            affected_stakeholders=["ML Engineers", "QA Team"],
                            metadata={"model": model_name, "cv": cv, "volatility": "high"}
                        ))
                
                # Data freshness risk (if metadata available)
                if "data_collection_date" in card.metadata:
                    try:
                        data_date = datetime.fromisoformat(card.metadata["data_collection_date"])
                        days_old = (datetime.now() - data_date).days
                        
                        if days_old > 365:  # Data older than 1 year
                            insights.append(Insight(
                                title=f"Data Freshness Risk - {model_name}",
                                description=f"Model trained on data that is {days_old} days old",
                                insight_type="risk",
                                confidence=0.75,
                                impact_level="medium",
                                evidence=[
                                    f"Data age: {days_old} days",
                                    f"Data collection: {data_date.strftime('%Y-%m-%d')}",
                                    "Potential distribution drift"
                                ],
                                actionable_recommendations=[
                                    "Evaluate model performance on recent data",
                                    "Consider retraining with fresh data",
                                    "Implement data freshness monitoring"
                                ],
                                affected_stakeholders=["Data Engineers", "ML Engineers", "Product Team"],
                                metadata={"model": model_name, "data_age_days": days_old}
                            ))
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Risk insight generation failed: {e}")
        
        return insights

    async def _generate_opportunity_insights(self, model_cards: List[ModelCard], context: Dict[str, Any]) -> List[Insight]:
        """Generate insights about improvement opportunities."""
        insights = []
        
        try:
            # Performance improvement opportunities
            all_metrics = defaultdict(list)
            for card in model_cards:
                for metric in card.evaluation_results:
                    all_metrics[metric.name].append((card.model_details.name, metric.value))
            
            for metric_name, model_values in all_metrics.items():
                if len(model_values) > 1:
                    values = [v for _, v in model_values]
                    best_value = max(values)
                    worst_value = min(values)
                    
                    # Significant improvement potential
                    if best_value - worst_value > 0.1 and worst_value < 0.8:
                        improvement_potential = best_value - worst_value
                        worst_model = [m for m, v in model_values if v == worst_value][0]
                        best_model = [m for m, v in model_values if v == best_value][0]
                        
                        insights.append(Insight(
                            title=f"{metric_name.title()} Improvement Opportunity",
                            description=f"Potential for {improvement_potential:.3f} improvement in {metric_name}",
                            insight_type="opportunity",
                            confidence=0.8,
                            impact_level="medium",
                            evidence=[
                                f"Best performance: {best_value:.3f} ({best_model})",
                                f"Improvement potential: {improvement_potential:.3f}",
                                f"Lowest performer: {worst_model}"
                            ],
                            actionable_recommendations=[
                                f"Apply techniques from {best_model} to improve {worst_model}",
                                "Conduct comparative analysis of successful approaches",
                                "Implement knowledge transfer between model teams"
                            ],
                            affected_stakeholders=["ML Engineers", "Research Team"],
                            metadata={
                                "metric": metric_name,
                                "best_model": best_model,
                                "worst_model": worst_model,
                                "improvement_potential": improvement_potential
                            }
                        ))
            
            # Architecture diversification opportunities
            architectures = [card.training_details.model_architecture or "unknown" 
                           for card in model_cards]
            unique_architectures = len(set(architectures))
            
            if unique_architectures < len(model_cards) * 0.5 and len(model_cards) > 3:
                most_common_arch = max(set(architectures), key=architectures.count)
                
                insights.append(Insight(
                    title="Architecture Diversification Opportunity",
                    description=f"Limited architectural diversity with {unique_architectures} unique approaches",
                    insight_type="opportunity",
                    confidence=0.75,
                    impact_level="medium",
                    evidence=[
                        f"Unique architectures: {unique_architectures}/{len(model_cards)}",
                        f"Most common: {most_common_arch}",
                        "Potential for architectural exploration"
                    ],
                    actionable_recommendations=[
                        "Experiment with alternative architectures",
                        "Implement architecture search techniques",
                        "Benchmark different architectural approaches"
                    ],
                    affected_stakeholders=["Research Team", "ML Engineers"],
                    metadata={"architecture_diversity": unique_architectures, "most_common": most_common_arch}
                ))
            
            # Multi-model ensemble opportunity
            if len(model_cards) > 2:
                # Check if models have complementary strengths
                metric_leaders = {}
                for metric_name, model_values in all_metrics.items():
                    if model_values:
                        best_model = max(model_values, key=lambda x: x[1])[0]
                        metric_leaders[metric_name] = best_model
                
                unique_leaders = len(set(metric_leaders.values()))
                if unique_leaders > 1:
                    insights.append(Insight(
                        title="Ensemble Learning Opportunity",
                        description=f"Models show complementary strengths across {unique_leaders} different leaders",
                        insight_type="opportunity",
                        confidence=0.8,
                        impact_level="high",
                        evidence=[
                            f"Metric leaders: {unique_leaders}",
                            f"Leader distribution: {list(set(metric_leaders.values()))}",
                            "Complementary model strengths identified"
                        ],
                        actionable_recommendations=[
                            "Develop ensemble model combining strengths",
                            "Implement weighted voting based on metric expertise",
                            "Explore stacking and blending techniques"
                        ],
                        affected_stakeholders=["ML Engineers", "Research Team", "Product Team"],
                        metadata={"leaders": unique_leaders, "metric_leaders": metric_leaders}
                    ))
            
            # Automation opportunities
            manual_processes = 0
            for card in model_cards:
                # Check for manual indicators in descriptions or metadata
                card_text = str(card.__dict__).lower()
                if any(term in card_text for term in ["manual", "hand", "manually"]):
                    manual_processes += 1
            
            if manual_processes > len(model_cards) * 0.3:
                insights.append(Insight(
                    title="Process Automation Opportunity",
                    description=f"High prevalence of manual processes in {manual_processes} models",
                    insight_type="opportunity",
                    confidence=0.7,
                    impact_level="medium",
                    evidence=[
                        f"Manual process indicators: {manual_processes}/{len(model_cards)}",
                        "Potential for workflow automation",
                        "Efficiency improvement opportunity"
                    ],
                    actionable_recommendations=[
                        "Implement automated model training pipelines",
                        "Develop automated evaluation frameworks",
                        "Create automated documentation generation"
                    ],
                    affected_stakeholders=["DevOps Team", "ML Engineers", "Management"],
                    metadata={"manual_processes": manual_processes, "automation_potential": "high"}
                ))
                
        except Exception as e:
            logger.warning(f"Opportunity insight generation failed: {e}")
        
        return insights

    async def _generate_quality_insights(self, model_cards: List[ModelCard], context: Dict[str, Any]) -> List[Insight]:
        """Generate insights about model card and development quality."""
        insights = []
        
        try:
            # Documentation quality assessment
            quality_scores = {}
            for card in model_cards:
                score = 0
                total_possible = 10
                
                # Scoring criteria
                if card.model_details.description and len(card.model_details.description) > 50:
                    score += 2
                if card.intended_use and len(card.intended_use) > 30:
                    score += 1
                if card.evaluation_results:
                    score += 2
                if card.limitations.known_limitations:
                    score += 1
                if card.ethical_considerations.bias_risks:
                    score += 1
                if card.training_details.hyperparameters:
                    score += 1
                if card.model_details.datasets:
                    score += 1
                if card.compliance_info:
                    score += 1
                
                quality_scores[card.model_details.name] = score / total_possible
            
            if quality_scores:
                avg_quality = statistics.mean(quality_scores.values())
                highest_quality = max(quality_scores.items(), key=lambda x: x[1])
                lowest_quality = min(quality_scores.items(), key=lambda x: x[1])
                
                # High quality model recognition
                if highest_quality[1] > 0.8:
                    insights.append(Insight(
                        title="Documentation Excellence",
                        description=f"Model '{highest_quality[0]}' demonstrates exceptional documentation quality",
                        insight_type="quality",
                        confidence=0.9,
                        impact_level="medium",
                        evidence=[
                            f"Quality score: {highest_quality[1]:.2f}",
                            "Comprehensive documentation coverage",
                            "Above team average"
                        ],
                        actionable_recommendations=[
                            "Use as documentation template for other models",
                            "Share best practices with team",
                            "Consider for documentation standards"
                        ],
                        affected_stakeholders=["Documentation Team", "ML Engineers"],
                        metadata={"champion_model": highest_quality[0], "quality_score": highest_quality[1]}
                    ))
                
                # Quality improvement needed
                if lowest_quality[1] < 0.5 and highest_quality[1] - lowest_quality[1] > 0.3:
                    insights.append(Insight(
                        title="Documentation Quality Improvement Needed",
                        description=f"Model '{lowest_quality[0]}' requires documentation enhancement",
                        insight_type="risk",
                        confidence=0.85,
                        impact_level="medium",
                        evidence=[
                            f"Quality score: {lowest_quality[1]:.2f}",
                            f"Below team average: {avg_quality:.2f}",
                            "Missing critical documentation elements"
                        ],
                        actionable_recommendations=[
                            "Complete missing documentation sections",
                            "Review documentation requirements",
                            "Implement documentation quality gates"
                        ],
                        affected_stakeholders=["ML Engineers", "QA Team"],
                        metadata={"low_quality_model": lowest_quality[0], "quality_score": lowest_quality[1]}
                    ))
            
            # Evaluation methodology quality
            evaluation_completeness = {}
            expected_metrics = {"accuracy", "precision", "recall", "f1"}
            
            for card in model_cards:
                card_metrics = {m.name.lower() for m in card.evaluation_results}
                coverage = len(card_metrics.intersection(expected_metrics)) / len(expected_metrics)
                evaluation_completeness[card.model_details.name] = coverage
            
            if evaluation_completeness:
                avg_eval_coverage = statistics.mean(evaluation_completeness.values())
                
                if avg_eval_coverage < 0.5:
                    insights.append(Insight(
                        title="Evaluation Methodology Enhancement Needed",
                        description=f"Average evaluation coverage is low: {avg_eval_coverage:.2f}",
                        insight_type="quality",
                        confidence=0.8,
                        impact_level="medium",
                        evidence=[
                            f"Average metric coverage: {avg_eval_coverage:.2f}",
                            f"Expected metrics: {', '.join(expected_metrics)}",
                            "Inconsistent evaluation practices"
                        ],
                        actionable_recommendations=[
                            "Standardize evaluation metric requirements",
                            "Implement automated evaluation pipelines",
                            "Provide evaluation best practice guidelines"
                        ],
                        affected_stakeholders=["ML Engineers", "QA Team", "Research Team"],
                        metadata={"avg_coverage": avg_eval_coverage, "expected_metrics": list(expected_metrics)}
                    ))
                        
        except Exception as e:
            logger.warning(f"Quality insight generation failed: {e}")
        
        return insights

    def _cluster_insights(self, insights: List[Insight]) -> List[InsightCluster]:
        """Cluster related insights together."""
        clusters = []
        
        try:
            # Group insights by type and theme
            insight_groups = defaultdict(list)
            
            for insight in insights:
                # Create grouping key based on type and key terms
                key_terms = self._extract_key_terms(insight.title + " " + insight.description)
                group_key = f"{insight.insight_type}_{key_terms}"
                insight_groups[group_key].append(insight)
            
            # Create clusters from groups
            for group_key, group_insights in insight_groups.items():
                if len(group_insights) > 1:  # Only cluster if multiple insights
                    theme = self._generate_cluster_theme(group_insights)
                    cluster_confidence = statistics.mean([i.confidence for i in group_insights])
                    priority_score = self._calculate_cluster_priority(group_insights)
                    summary = self._generate_cluster_summary(group_insights)
                    
                    clusters.append(InsightCluster(
                        theme=theme,
                        insights=group_insights,
                        cluster_confidence=cluster_confidence,
                        priority_score=priority_score,
                        summary=summary
                    ))
            
            # Sort clusters by priority
            clusters.sort(key=lambda x: x.priority_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Insight clustering failed: {e}")
        
        return clusters

    def _prioritize_insights(self, insights: List[Insight]) -> List[Insight]:
        """Prioritize insights based on impact and confidence."""
        try:
            impact_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            
            def priority_score(insight):
                impact_score = impact_weights.get(insight.impact_level, 1)
                return insight.confidence * impact_score
            
            return sorted(insights, key=priority_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Insight prioritization failed: {e}")
            return insights

    def _generate_executive_summary(self, clusters: List[InsightCluster], prioritized_insights: List[Insight]) -> str:
        """Generate executive summary of insights."""
        try:
            summary_parts = []
            
            # Overview
            summary_parts.append(f"Analysis identified {len(prioritized_insights)} actionable insights across {len(clusters)} thematic areas.")
            
            # Top priorities
            if prioritized_insights:
                top_insight = prioritized_insights[0]
                summary_parts.append(f"Highest priority: {top_insight.title} (Impact: {top_insight.impact_level}, Confidence: {top_insight.confidence:.2f})")
            
            # Key themes
            if clusters:
                top_themes = [cluster.theme for cluster in clusters[:3]]
                summary_parts.append(f"Key focus areas: {', '.join(top_themes)}")
            
            # Impact distribution
            impact_distribution = Counter([insight.impact_level for insight in prioritized_insights])
            if impact_distribution:
                summary_parts.append(f"Impact distribution: {dict(impact_distribution)}")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            return "Executive summary generation encountered an error."

    def _categorize_insights_by_type(self, insights: List[Insight]) -> Dict[str, List[Dict]]:
        """Categorize insights by type."""
        categorized = defaultdict(list)
        for insight in insights:
            categorized[insight.insight_type].append(insight.__dict__)
        return dict(categorized)

    def _categorize_insights_by_impact(self, insights: List[Insight]) -> Dict[str, List[Dict]]:
        """Categorize insights by impact level."""
        categorized = defaultdict(list)
        for insight in insights:
            categorized[insight.impact_level].append(insight.__dict__)
        return dict(categorized)

    def _extract_actionable_recommendations(self, insights: List[Insight]) -> List[str]:
        """Extract all actionable recommendations."""
        all_recommendations = []
        for insight in insights:
            all_recommendations.extend(insight.actionable_recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:20]  # Top 20

    def _analyze_stakeholder_impact(self, insights: List[Insight]) -> Dict[str, Any]:
        """Analyze impact on different stakeholders."""
        stakeholder_analysis = defaultdict(lambda: {"insight_count": 0, "impact_levels": [], "types": []})
        
        for insight in insights:
            for stakeholder in insight.affected_stakeholders:
                stakeholder_analysis[stakeholder]["insight_count"] += 1
                stakeholder_analysis[stakeholder]["impact_levels"].append(insight.impact_level)
                stakeholder_analysis[stakeholder]["types"].append(insight.insight_type)
        
        # Summarize for each stakeholder
        summary = {}
        for stakeholder, data in stakeholder_analysis.items():
            summary[stakeholder] = {
                "total_insights": data["insight_count"],
                "high_impact_count": data["impact_levels"].count("high") + data["impact_levels"].count("critical"),
                "most_common_type": max(set(data["types"]), key=data["types"].count) if data["types"] else "none"
            }
        
        return summary

    # Helper methods
    def _calculate_simple_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate simple linear regression slope."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0
        
        try:
            n = len(x_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_x2 = sum(x * x for x in x_values)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0
            
            return (n * sum_xy - sum_x * sum_y) / denominator
            
        except Exception:
            return 0

    def _extract_key_terms(self, text: str) -> str:
        """Extract key terms from text for clustering."""
        # Simple keyword extraction
        key_terms = []
        words = text.lower().split()
        
        # Common important terms in ML context
        important_terms = {
            "performance", "accuracy", "model", "training", "evaluation", "bias", "fairness",
            "architecture", "metrics", "risk", "opportunity", "trend", "anomaly", "quality"
        }
        
        for word in words:
            if word in important_terms:
                key_terms.append(word)
        
        return "_".join(key_terms[:3]) if key_terms else "general"

    def _generate_cluster_theme(self, insights: List[Insight]) -> str:
        """Generate theme name for insight cluster."""
        # Extract common terms from insights
        all_terms = []
        for insight in insights:
            all_terms.extend((insight.title + " " + insight.description).lower().split())
        
        # Find most common meaningful terms
        term_counts = Counter(all_terms)
        common_terms = [term for term, count in term_counts.most_common(3) 
                       if len(term) > 3 and term not in {"the", "and", "for", "with"}]
        
        if common_terms:
            return " ".join(common_terms).title()
        else:
            return f"{insights[0].insight_type.title()} Analysis"

    def _calculate_cluster_priority(self, insights: List[Insight]) -> float:
        """Calculate priority score for insight cluster."""
        impact_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        
        total_score = 0
        for insight in insights:
            impact_score = impact_weights.get(insight.impact_level, 1)
            total_score += insight.confidence * impact_score
        
        return total_score / len(insights)

    def _generate_cluster_summary(self, insights: List[Insight]) -> str:
        """Generate summary for insight cluster."""
        if not insights:
            return "No insights available."
        
        # Count by impact level
        impact_counts = Counter([insight.impact_level for insight in insights])
        most_common_impact = impact_counts.most_common(1)[0][0]
        
        return f"Cluster of {len(insights)} {most_common_impact}-impact insights focusing on {insights[0].insight_type} analysis."