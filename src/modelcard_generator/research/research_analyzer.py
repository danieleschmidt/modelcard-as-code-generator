"""Advanced research analysis capabilities for model cards."""

import asyncio
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats

from ..core.logging_config import get_logger
from ..core.models import ModelCard, PerformanceMetric

logger = get_logger(__name__)


@dataclass
class ResearchFinding:
    """A research finding from model card analysis."""
    title: str
    description: str
    evidence: List[str]
    confidence_score: float
    research_area: str
    implications: List[str]
    recommendations: List[str]
    statistical_significance: Optional[float] = None
    effect_size: Optional[float] = None


@dataclass
class ComparativeAnalysis:
    """Results from comparing multiple model cards."""
    baseline_model: str
    comparison_models: List[str]
    metric_comparisons: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, Any]]
    research_gaps: List[str]
    innovation_opportunities: List[str]


@dataclass
class TrendAnalysis:
    """Analysis of trends across model cards over time."""
    metric_trends: Dict[str, List[Tuple[datetime, float]]]
    emerging_patterns: List[str]
    performance_evolution: Dict[str, Dict[str, Any]]
    future_predictions: Dict[str, Any]


class ResearchAnalyzer:
    """Advanced research analysis for model card data."""

    def __init__(self):
        self.analysis_cache: Dict[str, Any] = {}
        self.research_database: List[ResearchFinding] = []
        self.comparative_studies: List[ComparativeAnalysis] = []
        
    def conduct_comprehensive_research_analysis(
        self, 
        model_cards: List[ModelCard],
        research_questions: List[str] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive research analysis on model cards."""
        
        try:
            logger.info(f"Starting comprehensive research analysis on {len(model_cards)} model cards")
            
            if not research_questions:
                research_questions = self._generate_default_research_questions(model_cards)
            
            analysis_results = {
                "research_questions": research_questions,
                "findings": [],
                "comparative_analysis": {},
                "trend_analysis": {},
                "innovation_opportunities": [],
                "research_gaps": [],
                "statistical_summary": {},
                "recommendations": []
            }
            
            # 1. Individual model analysis
            for card in model_cards:
                findings = self._analyze_individual_model(card)
                analysis_results["findings"].extend(findings)
            
            # 2. Comparative analysis
            if len(model_cards) > 1:
                comparative_analysis = self._conduct_comparative_analysis(model_cards)
                analysis_results["comparative_analysis"] = comparative_analysis.__dict__
            
            # 3. Trend analysis
            trend_analysis = self._analyze_trends(model_cards)
            analysis_results["trend_analysis"] = trend_analysis.__dict__
            
            # 4. Statistical analysis
            statistical_summary = self._perform_statistical_analysis(model_cards)
            analysis_results["statistical_summary"] = statistical_summary
            
            # 5. Research gap identification
            research_gaps = self._identify_research_gaps(model_cards)
            analysis_results["research_gaps"] = research_gaps
            
            # 6. Innovation opportunities
            innovation_opportunities = self._identify_innovation_opportunities(model_cards)
            analysis_results["innovation_opportunities"] = innovation_opportunities
            
            # 7. Generate recommendations
            recommendations = self._generate_research_recommendations(analysis_results)
            analysis_results["recommendations"] = recommendations
            
            logger.info("Comprehensive research analysis completed")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Research analysis failed: {e}")
            raise

    def _analyze_individual_model(self, model_card: ModelCard) -> List[ResearchFinding]:
        """Analyze individual model for research findings."""
        findings = []
        
        try:
            # Performance analysis
            performance_findings = self._analyze_performance_characteristics(model_card)
            findings.extend(performance_findings)
            
            # Architecture analysis
            architecture_findings = self._analyze_architecture_novelty(model_card)
            findings.extend(architecture_findings)
            
            # Ethical considerations analysis
            ethical_findings = self._analyze_ethical_innovations(model_card)
            findings.extend(ethical_findings)
            
            # Training methodology analysis
            training_findings = self._analyze_training_innovations(model_card)
            findings.extend(training_findings)
            
        except Exception as e:
            logger.warning(f"Individual model analysis failed for {model_card.model_details.name}: {e}")
        
        return findings

    def _analyze_performance_characteristics(self, model_card: ModelCard) -> List[ResearchFinding]:
        """Analyze performance characteristics for research insights."""
        findings = []
        
        if not model_card.evaluation_results:
            return findings
        
        try:
            # Analyze metric distributions
            metric_values = [m.value for m in model_card.evaluation_results]
            
            # Statistical analysis
            mean_performance = statistics.mean(metric_values)
            std_performance = statistics.stdev(metric_values) if len(metric_values) > 1 else 0
            
            # High performance finding
            if mean_performance > 0.95:
                findings.append(ResearchFinding(
                    title="Exceptional Performance Achievement",
                    description=f"Model achieves exceptional performance with mean score of {mean_performance:.4f}",
                    evidence=[f"Mean performance: {mean_performance:.4f}", f"Standard deviation: {std_performance:.4f}"],
                    confidence_score=0.9,
                    research_area="performance_optimization",
                    implications=["Potential for production deployment", "Benchmark for future models"],
                    recommendations=["Investigate architectural innovations", "Study training methodology"]
                ))
            
            # Performance consistency finding
            if std_performance < 0.02 and len(metric_values) > 3:
                findings.append(ResearchFinding(
                    title="Consistent Performance Across Metrics",
                    description=f"Model demonstrates consistent performance with low variance ({std_performance:.4f})",
                    evidence=[f"Performance variance: {std_performance:.4f}", f"Metrics count: {len(metric_values)}"],
                    confidence_score=0.8,
                    research_area="model_reliability",
                    implications=["High reliability for production use", "Stable training methodology"],
                    recommendations=["Study regularization techniques", "Analyze training stability"]
                ))
            
            # Metric-specific analysis
            accuracy_metrics = [m for m in model_card.evaluation_results if 'accuracy' in m.name.lower()]
            f1_metrics = [m for m in model_card.evaluation_results if 'f1' in m.name.lower()]
            
            if accuracy_metrics and f1_metrics:
                acc_val = accuracy_metrics[0].value
                f1_val = f1_metrics[0].value
                
                if abs(acc_val - f1_val) < 0.01:
                    findings.append(ResearchFinding(
                        title="Balanced Precision-Recall Performance",
                        description=f"Model achieves balanced performance (Accuracy: {acc_val:.3f}, F1: {f1_val:.3f})",
                        evidence=[f"Accuracy: {acc_val}", f"F1 Score: {f1_val}", f"Difference: {abs(acc_val - f1_val):.4f}"],
                        confidence_score=0.85,
                        research_area="classification_balance",
                        implications=["Well-balanced model for classification tasks", "Effective handling of class imbalance"],
                        recommendations=["Investigate training techniques", "Study loss function optimization"]
                    ))
            
        except Exception as e:
            logger.warning(f"Performance analysis failed: {e}")
        
        return findings

    def _analyze_architecture_novelty(self, model_card: ModelCard) -> List[ResearchFinding]:
        """Analyze architectural innovations and novelty."""
        findings = []
        
        try:
            architecture = model_card.training_details.model_architecture
            if not architecture:
                return findings
            
            arch_lower = architecture.lower()
            
            # Novel architecture patterns
            novel_patterns = [
                ("transformer", "attention mechanisms"),
                ("diffusion", "generative modeling"),
                ("mixture of experts", "scalable architectures"),
                ("neural architecture search", "automated design"),
                ("federated", "distributed learning"),
                ("few-shot", "meta-learning"),
                ("self-supervised", "representation learning")
            ]
            
            for pattern, research_area in novel_patterns:
                if pattern in arch_lower:
                    findings.append(ResearchFinding(
                        title=f"Novel {pattern.title()} Architecture",
                        description=f"Model utilizes {pattern} architecture, representing advancement in {research_area}",
                        evidence=[f"Architecture: {architecture}"],
                        confidence_score=0.7,
                        research_area=research_area.replace(" ", "_"),
                        implications=[f"Advancement in {research_area}", "Potential for broader applications"],
                        recommendations=[f"Study {pattern} effectiveness", "Compare with baseline architectures"]
                    ))
            
            # Multi-modal analysis
            if any(term in arch_lower for term in ["multi-modal", "multimodal", "vision-language", "cross-modal"]):
                findings.append(ResearchFinding(
                    title="Multi-Modal Architecture Innovation",
                    description="Model implements multi-modal architecture for cross-domain learning",
                    evidence=[f"Architecture: {architecture}"],
                    confidence_score=0.8,
                    research_area="multi_modal_learning",
                    implications=["Cross-domain knowledge transfer", "Enhanced representation learning"],
                    recommendations=["Analyze modal fusion techniques", "Study cross-modal alignment"]
                ))
            
        except Exception as e:
            logger.warning(f"Architecture analysis failed: {e}")
        
        return findings

    def _analyze_ethical_innovations(self, model_card: ModelCard) -> List[ResearchFinding]:
        """Analyze ethical considerations for research insights."""
        findings = []
        
        try:
            ethical = model_card.ethical_considerations
            
            # Advanced bias mitigation
            if len(ethical.bias_mitigation) > 3:
                findings.append(ResearchFinding(
                    title="Comprehensive Bias Mitigation Strategy",
                    description=f"Model implements {len(ethical.bias_mitigation)} bias mitigation strategies",
                    evidence=ethical.bias_mitigation[:3],  # Show first 3
                    confidence_score=0.8,
                    research_area="algorithmic_fairness",
                    implications=["Advanced fairness considerations", "Ethical AI development"],
                    recommendations=["Evaluate mitigation effectiveness", "Study fairness metrics"]
                ))
            
            # Fairness metrics analysis
            if ethical.fairness_metrics:
                metric_count = len(ethical.fairness_metrics)
                if metric_count > 2:
                    findings.append(ResearchFinding(
                        title="Multi-Dimensional Fairness Evaluation",
                        description=f"Model evaluated on {metric_count} fairness dimensions",
                        evidence=[f"Fairness metrics: {list(ethical.fairness_metrics.keys())}"],
                        confidence_score=0.85,
                        research_area="fairness_evaluation",
                        implications=["Comprehensive fairness assessment", "Multi-stakeholder consideration"],
                        recommendations=["Analyze fairness trade-offs", "Study intersectional fairness"]
                    ))
            
            # Sensitive attributes handling
            if len(ethical.sensitive_attributes) > 0:
                findings.append(ResearchFinding(
                    title="Explicit Sensitive Attribute Consideration",
                    description=f"Model explicitly considers {len(ethical.sensitive_attributes)} sensitive attributes",
                    evidence=ethical.sensitive_attributes,
                    confidence_score=0.9,
                    research_area="protected_attributes",
                    implications=["Proactive bias prevention", "Regulatory compliance"],
                    recommendations=["Study attribute impact", "Evaluate protection mechanisms"]
                ))
            
        except Exception as e:
            logger.warning(f"Ethical analysis failed: {e}")
        
        return findings

    def _analyze_training_innovations(self, model_card: ModelCard) -> List[ResearchFinding]:
        """Analyze training methodology innovations."""
        findings = []
        
        try:
            training = model_card.training_details
            
            # Advanced optimization techniques
            hyperparams = training.hyperparameters
            if hyperparams:
                # Learning rate scheduling
                if any("schedule" in str(k).lower() for k in hyperparams.keys()):
                    findings.append(ResearchFinding(
                        title="Advanced Learning Rate Scheduling",
                        description="Model uses sophisticated learning rate scheduling techniques",
                        evidence=[f"Hyperparameters include scheduling: {list(hyperparams.keys())}"],
                        confidence_score=0.7,
                        research_area="optimization_techniques",
                        implications=["Improved convergence", "Training stability"],
                        recommendations=["Study scheduling strategies", "Compare with constant rates"]
                    ))
                
                # Regularization techniques
                reg_techniques = [k for k in hyperparams.keys() if any(term in str(k).lower() 
                                for term in ["dropout", "weight_decay", "l1", "l2", "regularization"])]
                if len(reg_techniques) > 1:
                    findings.append(ResearchFinding(
                        title="Multi-Technique Regularization Strategy",
                        description=f"Model employs {len(reg_techniques)} regularization techniques",
                        evidence=reg_techniques,
                        confidence_score=0.75,
                        research_area="regularization_methods",
                        implications=["Overfitting prevention", "Generalization improvement"],
                        recommendations=["Analyze regularization impact", "Study technique combinations"]
                    ))
            
            # Data augmentation analysis
            preprocessing = training.preprocessing
            if preprocessing and any(term in preprocessing.lower() for term in 
                                   ["augment", "synthetic", "transform", "mixup", "cutmix"]):
                findings.append(ResearchFinding(
                    title="Advanced Data Augmentation",
                    description="Model utilizes advanced data augmentation techniques",
                    evidence=[f"Preprocessing: {preprocessing}"],
                    confidence_score=0.8,
                    research_area="data_augmentation",
                    implications=["Enhanced data efficiency", "Improved robustness"],
                    recommendations=["Evaluate augmentation impact", "Study technique effectiveness"]
                ))
            
        except Exception as e:
            logger.warning(f"Training analysis failed: {e}")
        
        return findings

    def _conduct_comparative_analysis(self, model_cards: List[ModelCard]) -> ComparativeAnalysis:
        """Conduct comparative analysis across multiple model cards."""
        
        try:
            # Select baseline (first model or highest performing)
            baseline_idx = 0
            if len(model_cards) > 1:
                # Find model with highest mean performance
                mean_performances = []
                for card in model_cards:
                    if card.evaluation_results:
                        mean_perf = statistics.mean([m.value for m in card.evaluation_results])
                        mean_performances.append(mean_perf)
                    else:
                        mean_performances.append(0)
                baseline_idx = mean_performances.index(max(mean_performances))
            
            baseline = model_cards[baseline_idx]
            comparisons = [card for i, card in enumerate(model_cards) if i != baseline_idx]
            
            # Metric comparisons
            metric_comparisons = {}
            statistical_tests = {}
            
            baseline_metrics = {m.name: m.value for m in baseline.evaluation_results}
            
            for comp_card in comparisons:
                comp_name = comp_card.model_details.name
                metric_comparisons[comp_name] = {}
                statistical_tests[comp_name] = {}
                
                comp_metrics = {m.name: m.value for m in comp_card.evaluation_results}
                
                # Compare common metrics
                common_metrics = set(baseline_metrics.keys()) & set(comp_metrics.keys())
                for metric in common_metrics:
                    baseline_val = baseline_metrics[metric]
                    comp_val = comp_metrics[metric]
                    
                    metric_comparisons[comp_name][metric] = {
                        "baseline": baseline_val,
                        "comparison": comp_val,
                        "difference": comp_val - baseline_val,
                        "percent_change": ((comp_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                    }
                    
                    # Statistical significance test (if confidence intervals available)
                    baseline_metric = next((m for m in baseline.evaluation_results if m.name == metric), None)
                    comp_metric = next((m for m in comp_card.evaluation_results if m.name == metric), None)
                    
                    if (baseline_metric and baseline_metric.confidence_interval and 
                        comp_metric and comp_metric.confidence_interval):
                        
                        # Perform t-test approximation
                        t_stat, p_value = self._approximate_t_test(
                            baseline_val, baseline_metric.confidence_interval,
                            comp_val, comp_metric.confidence_interval
                        )
                        
                        statistical_tests[comp_name][metric] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05 if p_value is not None else False
                        }
            
            # Identify research gaps
            research_gaps = self._identify_comparative_research_gaps(model_cards)
            
            # Identify innovation opportunities
            innovation_opportunities = self._identify_comparative_innovations(model_cards)
            
            return ComparativeAnalysis(
                baseline_model=baseline.model_details.name,
                comparison_models=[card.model_details.name for card in comparisons],
                metric_comparisons=metric_comparisons,
                statistical_tests=statistical_tests,
                research_gaps=research_gaps,
                innovation_opportunities=innovation_opportunities
            )
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            raise

    def _analyze_trends(self, model_cards: List[ModelCard]) -> TrendAnalysis:
        """Analyze trends across model cards."""
        
        try:
            # Sort by creation date
            sorted_cards = sorted(model_cards, key=lambda x: x.created_at)
            
            # Analyze metric trends
            metric_trends = defaultdict(list)
            all_metric_names = set()
            
            for card in sorted_cards:
                for metric in card.evaluation_results:
                    metric_trends[metric.name].append((card.created_at, metric.value))
                    all_metric_names.add(metric.name)
            
            # Analyze performance evolution
            performance_evolution = {}
            for metric_name in all_metric_names:
                if len(metric_trends[metric_name]) > 1:
                    values = [v for _, v in metric_trends[metric_name]]
                    
                    # Calculate trend statistics
                    trend_slope = self._calculate_trend_slope(metric_trends[metric_name])
                    trend_correlation = self._calculate_trend_correlation(metric_trends[metric_name])
                    
                    performance_evolution[metric_name] = {
                        "trend_slope": trend_slope,
                        "trend_correlation": trend_correlation,
                        "improvement": trend_slope > 0,
                        "volatility": statistics.stdev(values) if len(values) > 1 else 0
                    }
            
            # Identify emerging patterns
            emerging_patterns = self._identify_emerging_patterns(sorted_cards)
            
            # Generate future predictions
            future_predictions = self._generate_future_predictions(metric_trends, performance_evolution)
            
            return TrendAnalysis(
                metric_trends=dict(metric_trends),
                emerging_patterns=emerging_patterns,
                performance_evolution=performance_evolution,
                future_predictions=future_predictions
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise

    def _perform_statistical_analysis(self, model_cards: List[ModelCard]) -> Dict[str, Any]:
        """Perform statistical analysis across model cards."""
        
        try:
            statistical_summary = {
                "model_count": len(model_cards),
                "metric_statistics": {},
                "correlation_analysis": {},
                "distribution_analysis": {},
                "outlier_analysis": {}
            }
            
            # Collect all metrics
            all_metrics = defaultdict(list)
            for card in model_cards:
                for metric in card.evaluation_results:
                    all_metrics[metric.name].append(metric.value)
            
            # Statistical analysis for each metric
            for metric_name, values in all_metrics.items():
                if len(values) > 1:
                    statistical_summary["metric_statistics"][metric_name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "std_dev": statistics.stdev(values),
                        "min": min(values),
                        "max": max(values),
                        "range": max(values) - min(values),
                        "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 0
                    }
                    
                    # Distribution analysis
                    try:
                        # Shapiro-Wilk test for normality (if scipy available)
                        if len(values) >= 3:
                            shapiro_stat, shapiro_p = stats.shapiro(values)
                            statistical_summary["distribution_analysis"][metric_name] = {
                                "shapiro_statistic": shapiro_stat,
                                "shapiro_p_value": shapiro_p,
                                "likely_normal": shapiro_p > 0.05
                            }
                    except Exception:
                        pass
                    
                    # Outlier detection using IQR method
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = [v for v in values if v < lower_bound or v > upper_bound]
                    if outliers:
                        statistical_summary["outlier_analysis"][metric_name] = {
                            "outliers": outliers,
                            "outlier_count": len(outliers),
                            "outlier_percentage": len(outliers) / len(values) * 100
                        }
            
            # Correlation analysis between metrics
            metric_names = list(all_metrics.keys())
            if len(metric_names) > 1:
                correlations = {}
                for i, metric1 in enumerate(metric_names):
                    for j, metric2 in enumerate(metric_names[i+1:], i+1):
                        values1 = all_metrics[metric1]
                        values2 = all_metrics[metric2]
                        
                        # Find common indices
                        min_len = min(len(values1), len(values2))
                        if min_len > 1:
                            correlation = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                            if not np.isnan(correlation):
                                correlations[f"{metric1}_vs_{metric2}"] = {
                                    "correlation": correlation,
                                    "strength": self._interpret_correlation_strength(correlation)
                                }
                
                statistical_summary["correlation_analysis"] = correlations
            
            return statistical_summary
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {"error": str(e)}

    def _identify_research_gaps(self, model_cards: List[ModelCard]) -> List[str]:
        """Identify research gaps from model card analysis."""
        gaps = []
        
        try:
            # Analyze common missing elements
            missing_elements = defaultdict(int)
            total_cards = len(model_cards)
            
            for card in model_cards:
                if not card.model_details.description:
                    missing_elements["model_description"] += 1
                if not card.evaluation_results:
                    missing_elements["evaluation_metrics"] += 1
                if not card.ethical_considerations.bias_risks:
                    missing_elements["bias_analysis"] += 1
                if not card.ethical_considerations.fairness_metrics:
                    missing_elements["fairness_evaluation"] += 1
                if not card.limitations.known_limitations:
                    missing_elements["limitation_analysis"] += 1
                if not card.training_details.hyperparameters:
                    missing_elements["hyperparameter_documentation"] += 1
                if not card.model_details.datasets:
                    missing_elements["dataset_documentation"] += 1
            
            # Identify gaps present in >50% of models
            for element, count in missing_elements.items():
                if count / total_cards > 0.5:
                    gaps.append(f"Insufficient {element.replace('_', ' ')} in {count}/{total_cards} models")
            
            # Domain-specific gaps
            domains = [self._infer_domain_from_card(card) for card in model_cards]
            domain_counter = Counter(domains)
            
            if "computer_vision" in domain_counter and domain_counter["computer_vision"] > 2:
                cv_cards = [card for card in model_cards if self._infer_domain_from_card(card) == "computer_vision"]
                if not any("robustness" in str(card.evaluation_results).lower() for card in cv_cards):
                    gaps.append("Missing robustness evaluation in computer vision models")
            
            if "nlp" in domain_counter and domain_counter["nlp"] > 2:
                nlp_cards = [card for card in model_cards if self._infer_domain_from_card(card) == "nlp"]
                if not any("multilingual" in str(card.model_details).lower() for card in nlp_cards):
                    gaps.append("Limited multilingual evaluation in NLP models")
            
            # Performance evaluation gaps
            common_metrics = set()
            for card in model_cards:
                card_metrics = {m.name.lower() for m in card.evaluation_results}
                if not common_metrics:
                    common_metrics = card_metrics
                else:
                    common_metrics &= card_metrics
            
            expected_metrics = {"accuracy", "precision", "recall", "f1"}
            missing_common_metrics = expected_metrics - common_metrics
            if missing_common_metrics:
                gaps.append(f"Missing common metrics across models: {', '.join(missing_common_metrics)}")
            
        except Exception as e:
            logger.warning(f"Research gap identification failed: {e}")
        
        return gaps

    def _identify_innovation_opportunities(self, model_cards: List[ModelCard]) -> List[str]:
        """Identify innovation opportunities from model card analysis."""
        opportunities = []
        
        try:
            # Analyze architectural patterns
            architectures = [card.training_details.model_architecture or "" for card in model_cards]
            arch_lower = [arch.lower() for arch in architectures if arch]
            
            # Emerging architecture opportunities
            if not any("attention" in arch for arch in arch_lower):
                opportunities.append("Explore attention mechanisms for improved performance")
            
            if not any("ensemble" in arch for arch in arch_lower):
                opportunities.append("Investigate ensemble methods for robustness")
            
            if not any("transfer" in arch or "pretrained" in arch for arch in arch_lower):
                opportunities.append("Leverage transfer learning and pre-trained models")
            
            # Multi-modal opportunities
            domains = [self._infer_domain_from_card(card) for card in model_cards]
            if len(set(domains)) > 1 and not any("multimodal" in arch for arch in arch_lower):
                opportunities.append("Develop multi-modal architectures for cross-domain learning")
            
            # Ethical AI opportunities
            ethical_coverage = []
            for card in model_cards:
                ethical_score = 0
                if card.ethical_considerations.bias_risks:
                    ethical_score += 1
                if card.ethical_considerations.fairness_metrics:
                    ethical_score += 1
                if card.ethical_considerations.bias_mitigation:
                    ethical_score += 1
                ethical_coverage.append(ethical_score)
            
            avg_ethical_coverage = statistics.mean(ethical_coverage)
            if avg_ethical_coverage < 2:
                opportunities.append("Enhance ethical AI practices and fairness evaluation")
            
            # Performance optimization opportunities
            all_metrics = defaultdict(list)
            for card in model_cards:
                for metric in card.evaluation_results:
                    all_metrics[metric.name].append(metric.value)
            
            for metric_name, values in all_metrics.items():
                if values and statistics.mean(values) < 0.85:
                    opportunities.append(f"Improve {metric_name} performance (current avg: {statistics.mean(values):.3f})")
            
            # Data efficiency opportunities
            dataset_sizes = []
            for card in model_cards:
                if "size" in card.metadata:
                    try:
                        size = int(card.metadata["size"])
                        dataset_sizes.append(size)
                    except:
                        pass
            
            if dataset_sizes and statistics.mean(dataset_sizes) > 1000000:
                opportunities.append("Explore data-efficient training methods for large datasets")
            
            # Deployment opportunities
            deployment_mentions = []
            for card in model_cards:
                card_text = str(card.__dict__).lower()
                if any(term in card_text for term in ["edge", "mobile", "embedded"]):
                    deployment_mentions.append("edge")
                if any(term in card_text for term in ["cloud", "server", "distributed"]):
                    deployment_mentions.append("cloud")
            
            if not deployment_mentions:
                opportunities.append("Develop deployment-optimized model variants")
            
        except Exception as e:
            logger.warning(f"Innovation opportunity identification failed: {e}")
        
        return opportunities

    def _generate_research_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate research recommendations based on analysis results."""
        recommendations = []
        
        try:
            # Based on research gaps
            research_gaps = analysis_results.get("research_gaps", [])
            for gap in research_gaps[:3]:  # Top 3 gaps
                if "bias_analysis" in gap:
                    recommendations.append("Implement comprehensive bias testing frameworks")
                elif "evaluation_metrics" in gap:
                    recommendations.append("Establish standardized evaluation protocols")
                elif "limitation_analysis" in gap:
                    recommendations.append("Develop systematic limitation assessment methodologies")
            
            # Based on innovation opportunities
            opportunities = analysis_results.get("innovation_opportunities", [])
            for opportunity in opportunities[:3]:  # Top 3 opportunities
                if "attention" in opportunity:
                    recommendations.append("Research attention mechanism applications in domain-specific contexts")
                elif "ensemble" in opportunity:
                    recommendations.append("Investigate ensemble learning for improved reliability")
                elif "ethical" in opportunity:
                    recommendations.append("Develop advanced fairness evaluation frameworks")
            
            # Based on statistical analysis
            stats = analysis_results.get("statistical_summary", {})
            correlation_analysis = stats.get("correlation_analysis", {})
            
            strong_correlations = [
                corr_name for corr_name, corr_data in correlation_analysis.items()
                if abs(corr_data.get("correlation", 0)) > 0.7
            ]
            
            if strong_correlations:
                recommendations.append("Investigate causal relationships between strongly correlated metrics")
            
            # Based on trend analysis
            trend_analysis = analysis_results.get("trend_analysis", {})
            performance_evolution = trend_analysis.get("performance_evolution", {})
            
            declining_metrics = [
                metric for metric, evolution in performance_evolution.items()
                if evolution.get("trend_slope", 0) < -0.01
            ]
            
            if declining_metrics:
                recommendations.append("Address performance decline trends in identified metrics")
            
            # General research recommendations
            recommendations.extend([
                "Establish longitudinal studies for model performance tracking",
                "Develop standardized benchmarks for cross-model comparison",
                "Create reproducibility frameworks for research validation",
                "Implement automated research insight generation systems"
            ])
            
        except Exception as e:
            logger.warning(f"Research recommendation generation failed: {e}")
        
        return recommendations[:10]  # Return top 10 recommendations

    # Helper methods
    def _generate_default_research_questions(self, model_cards: List[ModelCard]) -> List[str]:
        """Generate default research questions based on available data."""
        questions = [
            "What are the performance characteristics across different model architectures?",
            "How do ethical considerations vary across different domains?",
            "What training methodologies yield the most consistent results?",
            "Are there systematic biases in evaluation practices?",
            "What are the emerging trends in model development?",
            "How can model reliability be improved?",
            "What are the gaps in current evaluation frameworks?",
            "Which optimization strategies show the most promise?",
            "How do deployment considerations affect model design?",
            "What are the opportunities for cross-domain knowledge transfer?"
        ]
        return questions

    def _approximate_t_test(self, mean1: float, ci1: List[float], mean2: float, ci2: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Approximate t-test from means and confidence intervals."""
        try:
            # Estimate standard errors from confidence intervals (assuming 95% CI)
            se1 = (ci1[1] - ci1[0]) / (2 * 1.96) if ci1 and len(ci1) >= 2 else None
            se2 = (ci2[1] - ci2[0]) / (2 * 1.96) if ci2 and len(ci2) >= 2 else None
            
            if se1 is None or se2 is None:
                return None, None
            
            # Pooled standard error
            pooled_se = math.sqrt(se1**2 + se2**2)
            
            if pooled_se == 0:
                return None, None
            
            # T-statistic
            t_stat = (mean1 - mean2) / pooled_se
            
            # Approximate p-value (two-tailed)
            # Using normal approximation for large samples
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
            return t_stat, p_value
            
        except Exception:
            return None, None

    def _calculate_trend_slope(self, time_value_pairs: List[Tuple[datetime, float]]) -> float:
        """Calculate trend slope from time-value pairs."""
        if len(time_value_pairs) < 2:
            return 0
        
        try:
            # Convert datetime to numeric (days since first observation)
            base_time = time_value_pairs[0][0]
            x_values = [(tv[0] - base_time).days for tv in time_value_pairs]
            y_values = [tv[1] for tv in time_value_pairs]
            
            # Linear regression slope
            n = len(x_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_x2 = sum(x * x for x in x_values)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception:
            return 0

    def _calculate_trend_correlation(self, time_value_pairs: List[Tuple[datetime, float]]) -> float:
        """Calculate correlation coefficient for trend."""
        if len(time_value_pairs) < 2:
            return 0
        
        try:
            base_time = time_value_pairs[0][0]
            x_values = [(tv[0] - base_time).days for tv in time_value_pairs]
            y_values = [tv[1] for tv in time_value_pairs]
            
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            return correlation if not np.isnan(correlation) else 0
            
        except Exception:
            return 0

    def _identify_emerging_patterns(self, sorted_cards: List[ModelCard]) -> List[str]:
        """Identify emerging patterns in model development."""
        patterns = []
        
        try:
            if len(sorted_cards) < 3:
                return patterns
            
            # Recent vs older models
            recent_cutoff = len(sorted_cards) // 2
            recent_cards = sorted_cards[recent_cutoff:]
            older_cards = sorted_cards[:recent_cutoff]
            
            # Architecture pattern changes
            recent_archs = [card.training_details.model_architecture or "" for card in recent_cards]
            older_archs = [card.training_details.model_architecture or "" for card in older_cards]
            
            recent_arch_terms = set()
            older_arch_terms = set()
            
            for arch in recent_archs:
                recent_arch_terms.update(arch.lower().split())
            for arch in older_archs:
                older_arch_terms.update(arch.lower().split())
            
            emerging_terms = recent_arch_terms - older_arch_terms
            if emerging_terms:
                patterns.append(f"Emerging architectural terms: {', '.join(list(emerging_terms)[:5])}")
            
            # Performance improvement patterns
            recent_performance = []
            older_performance = []
            
            for card in recent_cards:
                if card.evaluation_results:
                    avg_perf = statistics.mean([m.value for m in card.evaluation_results])
                    recent_performance.append(avg_perf)
            
            for card in older_cards:
                if card.evaluation_results:
                    avg_perf = statistics.mean([m.value for m in card.evaluation_results])
                    older_performance.append(avg_perf)
            
            if recent_performance and older_performance:
                recent_avg = statistics.mean(recent_performance)
                older_avg = statistics.mean(older_performance)
                
                if recent_avg > older_avg * 1.05:
                    patterns.append("Performance improvement trend in recent models")
                elif recent_avg < older_avg * 0.95:
                    patterns.append("Performance decline trend in recent models")
            
            # Ethical considerations evolution
            recent_ethical_coverage = []
            older_ethical_coverage = []
            
            for card in recent_cards:
                coverage = len(card.ethical_considerations.bias_risks) + len(card.ethical_considerations.fairness_metrics)
                recent_ethical_coverage.append(coverage)
            
            for card in older_cards:
                coverage = len(card.ethical_considerations.bias_risks) + len(card.ethical_considerations.fairness_metrics)
                older_ethical_coverage.append(coverage)
            
            if recent_ethical_coverage and older_ethical_coverage:
                if statistics.mean(recent_ethical_coverage) > statistics.mean(older_ethical_coverage):
                    patterns.append("Increasing focus on ethical considerations")
            
        except Exception as e:
            logger.warning(f"Pattern identification failed: {e}")
        
        return patterns

    def _generate_future_predictions(self, metric_trends: Dict, performance_evolution: Dict) -> Dict[str, Any]:
        """Generate future predictions based on trends."""
        predictions = {}
        
        try:
            for metric_name, evolution in performance_evolution.items():
                if metric_name in metric_trends and len(metric_trends[metric_name]) > 2:
                    trend_slope = evolution.get("trend_slope", 0)
                    current_values = [v for _, v in metric_trends[metric_name]]
                    current_avg = statistics.mean(current_values[-3:]) if len(current_values) >= 3 else current_values[-1]
                    
                    # Simple linear extrapolation
                    predicted_6_months = current_avg + (trend_slope * 180)  # 180 days
                    predicted_1_year = current_avg + (trend_slope * 365)  # 365 days
                    
                    # Bound predictions to reasonable ranges
                    predicted_6_months = max(0, min(1, predicted_6_months))
                    predicted_1_year = max(0, min(1, predicted_1_year))
                    
                    predictions[metric_name] = {
                        "6_month_prediction": predicted_6_months,
                        "1_year_prediction": predicted_1_year,
                        "confidence": "low" if abs(trend_slope) < 0.001 else "medium" if abs(trend_slope) < 0.01 else "high"
                    }
            
        except Exception as e:
            logger.warning(f"Future prediction generation failed: {e}")
        
        return predictions

    def _identify_comparative_research_gaps(self, model_cards: List[ModelCard]) -> List[str]:
        """Identify research gaps from comparative analysis."""
        gaps = []
        
        # Implementation similar to _identify_research_gaps but focused on comparative aspects
        try:
            # Standardization gaps
            metric_sets = []
            for card in model_cards:
                metrics = {m.name for m in card.evaluation_results}
                metric_sets.append(metrics)
            
            if len(set(frozenset(ms) for ms in metric_sets)) > 1:
                gaps.append("Lack of standardized evaluation metrics across models")
            
            # Baseline comparison gaps
            if len(model_cards) > 2:
                baseline_mentions = sum(1 for card in model_cards if "baseline" in str(card.__dict__).lower())
                if baseline_mentions < len(model_cards) * 0.3:
                    gaps.append("Insufficient baseline comparisons in model evaluations")
            
        except Exception as e:
            logger.warning(f"Comparative research gap identification failed: {e}")
        
        return gaps

    def _identify_comparative_innovations(self, model_cards: List[ModelCard]) -> List[str]:
        """Identify innovation opportunities from comparative analysis."""
        innovations = []
        
        try:
            # Cross-model learning opportunities
            architectures = set()
            for card in model_cards:
                if card.training_details.model_architecture:
                    architectures.add(card.training_details.model_architecture.lower())
            
            if len(architectures) > 1:
                innovations.append("Cross-architecture knowledge distillation opportunities")
            
            # Multi-model ensemble opportunities
            if len(model_cards) > 2:
                innovations.append("Multi-model ensemble development for improved robustness")
            
        except Exception as e:
            logger.warning(f"Comparative innovation identification failed: {e}")
        
        return innovations

    def _infer_domain_from_card(self, model_card: ModelCard) -> str:
        """Infer domain from model card content."""
        text_content = " ".join([
            model_card.model_details.name.lower(),
            (model_card.model_details.description or "").lower(),
            " ".join(model_card.model_details.datasets).lower()
        ])
        
        if any(term in text_content for term in ["image", "vision", "detection", "segmentation"]):
            return "computer_vision"
        elif any(term in text_content for term in ["text", "nlp", "language", "sentiment"]):
            return "nlp"
        elif any(term in text_content for term in ["medical", "healthcare", "clinical"]):
            return "healthcare"
        elif any(term in text_content for term in ["financial", "finance", "trading"]):
            return "finance"
        else:
            return "general"

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very weak"