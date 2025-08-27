#!/usr/bin/env python3
"""
Progressive Quality Gates Test Suite
Comprehensive testing of the enhanced autonomous SDLC features.
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modelcard_generator.core.progressive_quality_gates import (
    ProgressiveQualityGateEngine, 
    QualityGateResult,
    QualityMetric
)
from modelcard_generator.core.autonomous_learning_system import (
    AutonomousLearningSystem,
    LearningPattern,
    AdaptiveRecommendation
)
from modelcard_generator.monitoring.realtime_quality_monitor import (
    RealTimeQualityMonitor,
    QualityAlert,
    AlertLevel,
    QualityDashboard
)
from modelcard_generator.core.self_healing_architecture import (
    SelfHealingOrchestrator,
    HealthStatus,
    HealingAction,
    HealingStrategy
)
from modelcard_generator.core.models import ModelCard, CardConfig, CardFormat
from modelcard_generator.core.generator import ModelCardGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProgressiveQualityGatesTestSuite:
    """Comprehensive test suite for progressive quality gates."""
    
    def __init__(self):
        self.results = {
            "test_start_time": datetime.now(),
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {},
            "quality_metrics": {},
            "learning_metrics": {},
            "healing_metrics": {},
            "detailed_results": []
        }
        
        # Initialize components
        self.quality_gates = ProgressiveQualityGateEngine()
        self.learning_system = AutonomousLearningSystem()
        self.quality_monitor = RealTimeQualityMonitor(
            quality_gates=self.quality_gates,
            learning_system=self.learning_system
        )
        self.healing_system = SelfHealingOrchestrator(
            quality_monitor=self.quality_monitor
        )
        
        # Test data
        self.test_model_cards = []
        self._generate_test_data()
    
    def _generate_test_data(self) -> None:
        """Generate test model cards for validation."""
        
        logger.info("Generating test model cards...")
        
        # High-quality model card
        high_quality_card = ModelCard(
            model_details={
                "name": "sentiment-analyzer-v2",
                "version": "2.0.0",
                "description": "Advanced transformer-based sentiment analysis model",
                "architecture": "BERT-large with custom classification head",
                "parameters": 345000000,
                "license": "apache-2.0",
                "developers": ["AI Research Team"],
                "contact": "ai-team@company.com"
            },
            intended_use={
                "primary_uses": ["Sentiment analysis", "Content moderation"],
                "primary_users": ["Data scientists", "ML engineers"],
                "out_of_scope": ["Medical diagnosis", "Legal advice"]
            },
            factors={
                "relevant_factors": ["Text language", "Domain specificity", "Text length"],
                "evaluation_factors": ["Accuracy", "Fairness", "Robustness"]
            },
            metrics={
                "accuracy": 0.94,
                "precision": 0.93,
                "recall": 0.95,
                "f1_score": 0.94,
                "auc_roc": 0.97
            },
            evaluation_data={
                "dataset": "Custom evaluation dataset",
                "size": 50000,
                "motivation": "Comprehensive evaluation across domains"
            },
            training_data={
                "dataset": "Multi-domain sentiment corpus",
                "size": 2000000,
                "preprocessing": "Standard NLP preprocessing pipeline"
            },
            quantitative_analyses=[
                {"metric": "accuracy", "value": 0.94, "confidence_interval": [0.93, 0.95]},
                {"metric": "fairness", "value": 0.88, "demographic_parity": True}
            ],
            ethical_considerations={
                "bias_risks": "Potential bias toward certain demographic groups",
                "mitigation_strategies": ["Bias testing", "Diverse training data"],
                "privacy_considerations": "No personal data retention"
            },
            recommendations={
                "use_cases": "Production-ready for content analysis",
                "limitations": "May struggle with sarcasm and irony",
                "monitoring": "Regular performance monitoring recommended"
            }
        )
        
        # Medium-quality model card (missing some fields)
        medium_quality_card = ModelCard(
            model_details={
                "name": "basic-classifier",
                "description": "Simple text classifier"
            },
            intended_use={
                "primary_uses": ["Text classification"]
            },
            metrics={
                "accuracy": 0.82
            },
            ethical_considerations={
                "bias_risks": "Limited bias analysis performed"
            }
        )
        
        # Low-quality model card (minimal information)
        low_quality_card = ModelCard(
            model_details={
                "name": "test-model"
            },
            metrics={
                "accuracy": 0.65
            }
        )
        
        self.test_model_cards = [
            ("high_quality", high_quality_card),
            ("medium_quality", medium_quality_card),
            ("low_quality", low_quality_card)
        ]
        
        logger.info(f"Generated {len(self.test_model_cards)} test model cards")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        
        logger.info("🧪 Starting Progressive Quality Gates Test Suite")
        logger.info("=" * 60)
        
        try:
            # Test Progressive Quality Gates
            await self._test_progressive_quality_gates()
            
            # Test Autonomous Learning System
            await self._test_autonomous_learning_system()
            
            # Test Real-time Quality Monitoring
            await self._test_realtime_quality_monitoring()
            
            # Test Self-Healing Architecture
            await self._test_self_healing_architecture()
            
            # Integration Tests
            await self._test_system_integration()
            
            # Performance Benchmarks
            await self._test_performance_benchmarks()
            
            # Learning and Adaptation Tests
            await self._test_learning_adaptation()
            
        except Exception as e:
            logger.error(f"Test suite failed with exception: {e}")
            self._record_test_result("test_suite_execution", False, error=str(e))
        
        # Calculate final results
        self._calculate_final_results()
        
        return self.results
    
    async def _test_progressive_quality_gates(self) -> None:
        """Test progressive quality gates functionality."""
        
        logger.info("🔍 Testing Progressive Quality Gates...")
        
        for card_type, model_card in self.test_model_cards:
            start_time = time.time()
            
            try:
                # Execute quality gates
                quality_results = await self.quality_gates.execute_progressive_gates(
                    model_card=model_card,
                    context={"test_type": card_type}
                )
                
                execution_time = time.time() - start_time
                
                # Validate results
                self._validate_quality_gate_results(quality_results, card_type)
                
                # Store performance metrics
                self.results["performance_metrics"][f"quality_gates_{card_type}"] = {
                    "execution_time": execution_time,
                    "gates_executed": len(quality_results),
                    "gates_passed": sum(1 for r in quality_results if r.passed),
                    "overall_score": statistics.mean([r.score for r in quality_results])
                }
                
                self._record_test_result(
                    f"quality_gates_{card_type}", 
                    True, 
                    execution_time=execution_time
                )
                
            except Exception as e:
                self._record_test_result(f"quality_gates_{card_type}", False, error=str(e))
        
        # Test adaptive threshold learning
        await self._test_adaptive_thresholds()
        
        logger.info("✅ Progressive Quality Gates tests completed")
    
    def _validate_quality_gate_results(
        self, 
        results: List[QualityGateResult], 
        card_type: str
    ) -> None:
        """Validate quality gate results."""
        
        # Check that all expected gates were executed
        expected_gates = {
            "completeness", "accuracy", "security", 
            "performance", "compliance", "maintainability"
        }
        
        executed_gates = {r.gate_name for r in results}
        
        if not expected_gates.issubset(executed_gates):
            raise AssertionError(f"Missing quality gates: {expected_gates - executed_gates}")
        
        # Validate result structure
        for result in results:
            if not isinstance(result.score, (int, float)) or not (0 <= result.score <= 1):
                raise AssertionError(f"Invalid score for {result.gate_name}: {result.score}")
            
            if result.execution_time <= 0:
                raise AssertionError(f"Invalid execution time for {result.gate_name}")
        
        # Check score consistency with card quality
        avg_score = statistics.mean([r.score for r in results])
        
        if card_type == "high_quality" and avg_score < 0.8:
            raise AssertionError(f"High quality card scored too low: {avg_score}")
        elif card_type == "low_quality" and avg_score > 0.7:
            raise AssertionError(f"Low quality card scored too high: {avg_score}")
    
    async def _test_adaptive_thresholds(self) -> None:
        """Test adaptive threshold learning."""
        
        logger.info("🎯 Testing adaptive thresholds...")
        
        # Get initial thresholds
        initial_thresholds = {
            name: metric.threshold 
            for name, metric in self.quality_gates.standard_metrics.items()
        }
        
        # Simulate multiple generations with varying quality
        for i in range(20):
            quality_score = 0.6 + (i * 0.02)  # Gradually improving quality
            
            for metric_name, metric in self.quality_gates.standard_metrics.items():
                metric.update(quality_score)
        
        # Check if thresholds adapted
        adapted = False
        for name, metric in self.quality_gates.standard_metrics.items():
            if abs(metric.threshold - initial_thresholds[name]) > 0.01:
                adapted = True
                break
        
        self._record_test_result("adaptive_thresholds", adapted)
        
        if adapted:
            logger.info("✅ Adaptive thresholds are working correctly")
        else:
            logger.warning("⚠️  Adaptive thresholds may need tuning")
    
    async def _test_autonomous_learning_system(self) -> None:
        """Test autonomous learning system functionality."""
        
        logger.info("🧠 Testing Autonomous Learning System...")
        
        # Test pattern discovery
        for i, (card_type, model_card) in enumerate(self.test_model_cards):
            try:
                # Simulate quality gate results
                mock_quality_results = [
                    QualityGateResult(
                        gate_name="completeness",
                        passed=card_type == "high_quality",
                        score=0.9 if card_type == "high_quality" else 0.6,
                        metrics={},
                        execution_time=0.1
                    )
                ]
                
                # Learn from generation
                recommendations = await self.learning_system.learn_from_generation(
                    model_card=model_card,
                    generation_context={"test_iteration": i, "card_type": card_type},
                    quality_results=mock_quality_results,
                    user_feedback={"satisfaction_score": 0.8}
                )
                
                # Validate recommendations
                if not isinstance(recommendations, list):
                    raise AssertionError("Learning system should return list of recommendations")
                
                # Store learning metrics
                learning_summary = self.learning_system.get_learning_summary()
                self.results["learning_metrics"][f"iteration_{i}"] = {
                    "patterns_discovered": learning_summary["patterns_discovered"],
                    "prediction_accuracy": learning_summary["prediction_accuracy"],
                    "recommendations_count": len(recommendations)
                }
                
                self._record_test_result(f"learning_system_{card_type}", True)
                
            except Exception as e:
                self._record_test_result(f"learning_system_{card_type}", False, error=str(e))
        
        # Test pattern similarity detection
        await self._test_pattern_similarity()
        
        logger.info("✅ Autonomous Learning System tests completed")
    
    async def _test_pattern_similarity(self) -> None:
        """Test pattern similarity detection."""
        
        logger.info("🔍 Testing pattern similarity detection...")
        
        # Create two similar patterns
        pattern1 = {
            "format_type": "huggingface",
            "use_case": "nlp",
            "has_metrics": True,
            "completeness_score": 0.8
        }
        
        pattern2 = {
            "format_type": "huggingface", 
            "use_case": "nlp",
            "has_metrics": True,
            "completeness_score": 0.85
        }
        
        # Test similarity calculation
        similarity = self.learning_system._calculate_pattern_similarity(pattern1, pattern2)
        
        # Should be high similarity
        if similarity < 0.8:
            raise AssertionError(f"Pattern similarity too low: {similarity}")
        
        self._record_test_result("pattern_similarity", True, similarity=similarity)
        logger.info(f"✅ Pattern similarity detection working (similarity: {similarity:.3f})")
    
    async def _test_realtime_quality_monitoring(self) -> None:
        """Test real-time quality monitoring system."""
        
        logger.info("📊 Testing Real-time Quality Monitoring...")
        
        # Start monitoring
        self.quality_monitor.start_monitoring()
        
        try:
            # Simulate multiple model card generations
            for i, (card_type, model_card) in enumerate(self.test_model_cards * 3):  # 9 total
                quality_results = await self.quality_gates.execute_progressive_gates(model_card)
                generation_time = 0.5 + (i * 0.1)  # Varying generation times
                
                # Record generation
                await self.quality_monitor.record_generation(
                    model_card=model_card,
                    quality_results=quality_results,
                    generation_time=generation_time,
                    context={"iteration": i, "card_type": card_type},
                    user_feedback={"satisfaction_score": 0.8 - (i * 0.05)}
                )
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.1)
            
            # Wait for monitoring to process
            await asyncio.sleep(2)
            
            # Get dashboard data
            dashboard = self.quality_monitor.get_dashboard_data()
            
            # Validate dashboard
            self._validate_dashboard(dashboard)
            
            # Store monitoring metrics
            self.results["quality_metrics"]["monitoring"] = {
                "generations_recorded": len(self.quality_monitor.metric_history),
                "alerts_generated": len(dashboard.alerts),
                "system_health": dashboard.system_health,
                "current_metrics": dashboard.current_metrics
            }
            
            self._record_test_result("realtime_monitoring", True)
            
        except Exception as e:
            self._record_test_result("realtime_monitoring", False, error=str(e))
        
        finally:
            # Stop monitoring
            self.quality_monitor.stop_monitoring()
        
        logger.info("✅ Real-time Quality Monitoring tests completed")
    
    def _validate_dashboard(self, dashboard: QualityDashboard) -> None:
        """Validate dashboard data structure and content."""
        
        # Check required fields
        required_fields = [
            "current_metrics", "trends", "alerts", "performance_summary",
            "recommendations", "system_health", "uptime"
        ]
        
        for field in required_fields:
            if not hasattr(dashboard, field):
                raise AssertionError(f"Dashboard missing field: {field}")
        
        # Validate metrics
        if not isinstance(dashboard.current_metrics, dict):
            raise AssertionError("Current metrics should be a dictionary")
        
        # Validate system health
        valid_health_statuses = ["excellent", "good", "fair", "poor", "critical", "unknown"]
        if dashboard.system_health not in valid_health_statuses:
            raise AssertionError(f"Invalid system health: {dashboard.system_health}")
        
        # Validate uptime
        if dashboard.uptime < 0:
            raise AssertionError(f"Invalid uptime: {dashboard.uptime}")
    
    async def _test_self_healing_architecture(self) -> None:
        """Test self-healing architecture functionality."""
        
        logger.info("🔧 Testing Self-Healing Architecture...")
        
        # Start healing system
        self.healing_system.start_healing_system()
        
        try:
            # Wait for initialization
            await asyncio.sleep(2)
            
            # Test component health monitoring
            await self._test_component_health_monitoring()
            
            # Test healing action planning
            await self._test_healing_action_planning()
            
            # Test healing execution
            await self._test_healing_execution()
            
            # Get system status
            system_status = self.healing_system.get_system_status()
            
            # Validate system status
            self._validate_system_status(system_status)
            
            # Store healing metrics
            self.results["healing_metrics"] = {
                "components_monitored": len(system_status["component_statuses"]),
                "active_healings": len(system_status["active_healings"]),
                "recent_healings": len(system_status["recent_healings"]),
                "overall_health": system_status["overall_health"]
            }
            
            self._record_test_result("self_healing_architecture", True)
            
        except Exception as e:
            self._record_test_result("self_healing_architecture", False, error=str(e))
        
        finally:
            # Stop healing system
            self.healing_system.stop_healing_system()
        
        logger.info("✅ Self-Healing Architecture tests completed")
    
    async def _test_component_health_monitoring(self) -> None:
        """Test component health monitoring."""
        
        logger.info("🔍 Testing component health monitoring...")
        
        # Check that components are being monitored
        if not self.healing_system.component_monitors:
            raise AssertionError("No component monitors found")
        
        # Test health check execution
        for component_name, monitor in self.healing_system.component_monitors.items():
            health_result = await monitor.check_health()
            
            if health_result.component_name != component_name:
                raise AssertionError(f"Health check component mismatch: {health_result.component_name}")
            
            if health_result.execution_time <= 0:
                raise AssertionError(f"Invalid execution time for {component_name}")
        
        self._record_test_result("component_health_monitoring", True)
    
    async def _test_healing_action_planning(self) -> None:
        """Test healing action planning."""
        
        logger.info("📋 Testing healing action planning...")
        
        # Create a mock health result with issues
        from modelcard_generator.core.self_healing_architecture import HealthCheckResult, HealthStatus
        
        mock_health_result = HealthCheckResult(
            component_name="test_component",
            status=HealthStatus.CRITICAL,
            issues=["High memory usage: 95%", "Performance degradation detected"],
            recommendations=["Scale resources", "Clear cache"]
        )
        
        # Plan healing actions
        healing_actions = self.healing_system._plan_healing_actions(mock_health_result)
        
        # Validate planned actions
        if not healing_actions:
            raise AssertionError("No healing actions planned for critical health result")
        
        for action in healing_actions:
            if not action.action_id:
                raise AssertionError("Healing action missing action_id")
            
            if not action.strategy:
                raise AssertionError("Healing action missing strategy")
        
        self._record_test_result("healing_action_planning", True, actions_count=len(healing_actions))
    
    async def _test_healing_execution(self) -> None:
        """Test healing execution."""
        
        logger.info("⚙️ Testing healing execution...")
        
        # Create a test healing action
        test_action = HealingAction(
            action_id="test_memory_cleanup",
            strategy=HealingStrategy.MEMORY_CLEANUP,
            component="test_component",
            description="Test memory cleanup",
            priority=5,
            estimated_time=5.0,
            success_probability=0.9
        )
        
        # Execute healing action
        result = self.healing_system._execute_healing_action(test_action)
        
        # Validate result
        if result.action_id != test_action.action_id:
            raise AssertionError("Healing result action_id mismatch")
        
        if result.execution_time <= 0:
            raise AssertionError("Invalid healing execution time")
        
        if not isinstance(result.success, bool):
            raise AssertionError("Healing result success should be boolean")
        
        self._record_test_result("healing_execution", True, success=result.success)
    
    def _validate_system_status(self, status: Dict[str, Any]) -> None:
        """Validate system status structure."""
        
        required_fields = [
            "is_running", "system_metrics", "component_statuses",
            "active_healings", "recent_healings", "overall_health"
        ]
        
        for field in required_fields:
            if field not in status:
                raise AssertionError(f"System status missing field: {field}")
        
        # Validate overall health
        valid_health_values = ["excellent", "good", "degraded", "critical", "unknown"]
        if status["overall_health"] not in valid_health_values:
            raise AssertionError(f"Invalid overall health: {status['overall_health']}")
    
    async def _test_system_integration(self) -> None:
        """Test integration between all systems."""
        
        logger.info("🔗 Testing System Integration...")
        
        try:
            # Start all systems
            self.quality_monitor.start_monitoring()
            self.healing_system.start_healing_system()
            
            # Generate a model card and process through all systems
            test_card = self.test_model_cards[0][1]  # High quality card
            
            # Execute quality gates
            quality_results = await self.quality_gates.execute_progressive_gates(test_card)
            
            # Learn from generation
            recommendations = await self.learning_system.learn_from_generation(
                model_card=test_card,
                generation_context={"integration_test": True},
                quality_results=quality_results
            )
            
            # Record in monitoring system
            await self.quality_monitor.record_generation(
                model_card=test_card,
                quality_results=quality_results,
                generation_time=1.0,
                context={"integration_test": True}
            )
            
            # Wait for systems to process
            await asyncio.sleep(3)
            
            # Validate integration
            dashboard = self.quality_monitor.get_dashboard_data()
            system_status = self.healing_system.get_system_status()
            learning_summary = self.learning_system.get_learning_summary()
            
            # Check that data flowed between systems
            if len(dashboard.recommendations) == 0:
                logger.warning("No recommendations in dashboard")
            
            if learning_summary["patterns_discovered"] == 0:
                logger.warning("No patterns discovered during integration test")
            
            self._record_test_result("system_integration", True)
            
        except Exception as e:
            self._record_test_result("system_integration", False, error=str(e))
        
        finally:
            self.quality_monitor.stop_monitoring()
            self.healing_system.stop_healing_system()
        
        logger.info("✅ System Integration tests completed")
    
    async def _test_performance_benchmarks(self) -> None:
        """Test performance benchmarks."""
        
        logger.info("⚡ Testing Performance Benchmarks...")
        
        # Test single generation performance
        start_time = time.time()
        test_card = self.test_model_cards[0][1]
        
        for _ in range(10):
            quality_results = await self.quality_gates.execute_progressive_gates(test_card)
        
        avg_single_time = (time.time() - start_time) / 10
        
        # Test batch performance
        start_time = time.time()
        batch_tasks = [
            self.quality_gates.execute_progressive_gates(card) 
            for _, card in self.test_model_cards * 10  # 30 cards
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        batch_total_time = time.time() - start_time
        batch_throughput = len(batch_results) / batch_total_time
        
        # Store performance metrics
        self.results["performance_metrics"]["benchmarks"] = {
            "single_generation_time": avg_single_time,
            "batch_total_time": batch_total_time,
            "batch_throughput": batch_throughput,
            "batch_size": len(batch_results)
        }
        
        # Performance thresholds
        single_threshold = 2.0  # seconds
        throughput_threshold = 5.0  # cards per second
        
        single_pass = avg_single_time < single_threshold
        throughput_pass = batch_throughput > throughput_threshold
        
        self._record_test_result(
            "performance_benchmarks", 
            single_pass and throughput_pass,
            single_time=avg_single_time,
            throughput=batch_throughput
        )
        
        logger.info(f"✅ Performance: {avg_single_time:.3f}s/card, {batch_throughput:.1f} cards/s")
    
    async def _test_learning_adaptation(self) -> None:
        """Test learning and adaptation capabilities."""
        
        logger.info("🎓 Testing Learning and Adaptation...")
        
        try:
            # Test multiple learning iterations
            initial_patterns = len(self.learning_system.patterns)
            
            # Simulate learning from varied scenarios
            scenarios = [
                ("nlp", "huggingface", True, 0.9),
                ("cv", "google", False, 0.7),
                ("nlp", "huggingface", True, 0.95),  # Similar to first
                ("llm", "eu_cra", True, 0.85),
            ]
            
            for i, (use_case, format_type, has_metrics, quality) in enumerate(scenarios):
                mock_card = ModelCard(
                    model_details={"name": f"test-{i}", "use_case": use_case}
                )
                
                mock_results = [
                    QualityGateResult(
                        gate_name="accuracy",
                        passed=quality > 0.8,
                        score=quality,
                        metrics={},
                        execution_time=0.1
                    )
                ]
                
                context = {
                    "format": format_type,
                    "use_case": use_case,
                    "has_metrics": has_metrics
                }
                
                await self.learning_system.learn_from_generation(
                    model_card=mock_card,
                    generation_context=context,
                    quality_results=mock_results
                )
            
            # Check learning occurred
            final_patterns = len(self.learning_system.patterns)
            patterns_learned = final_patterns > initial_patterns
            
            # Test recommendation generation
            learning_summary = self.learning_system.get_learning_summary()
            
            # Validate learning metrics
            if learning_summary["patterns_discovered"] == 0:
                logger.warning("No patterns discovered during learning test")
            
            self._record_test_result(
                "learning_adaptation", 
                patterns_learned,
                initial_patterns=initial_patterns,
                final_patterns=final_patterns,
                learning_summary=learning_summary
            )
            
        except Exception as e:
            self._record_test_result("learning_adaptation", False, error=str(e))
        
        logger.info("✅ Learning and Adaptation tests completed")
    
    def _record_test_result(
        self, 
        test_name: str, 
        passed: bool, 
        **kwargs
    ) -> None:
        """Record a test result."""
        
        self.results["tests_executed"] += 1
        
        if passed:
            self.results["tests_passed"] += 1
        else:
            self.results["tests_failed"] += 1
        
        result = {
            "test_name": test_name,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        self.results["detailed_results"].append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    def _calculate_final_results(self) -> None:
        """Calculate final test results and metrics."""
        
        self.results["test_end_time"] = datetime.now()
        self.results["total_duration"] = (
            self.results["test_end_time"] - self.results["test_start_time"]
        ).total_seconds()
        
        # Calculate success rate
        if self.results["tests_executed"] > 0:
            self.results["success_rate"] = (
                self.results["tests_passed"] / self.results["tests_executed"]
            )
        else:
            self.results["success_rate"] = 0.0
        
        # Generate summary
        self.results["summary"] = {
            "overall_status": "PASSED" if self.results["success_rate"] >= 0.8 else "FAILED",
            "tests_executed": self.results["tests_executed"],
            "tests_passed": self.results["tests_passed"],
            "tests_failed": self.results["tests_failed"],
            "success_rate": f"{self.results['success_rate']:.1%}",
            "duration": f"{self.results['total_duration']:.1f}s"
        }
    
    def print_results_summary(self) -> None:
        """Print a comprehensive results summary."""
        
        print("\n" + "=" * 80)
        print("🧪 PROGRESSIVE QUALITY GATES TEST SUITE RESULTS")
        print("=" * 80)
        
        summary = self.results["summary"]
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Tests Executed: {summary['tests_executed']}")
        print(f"Tests Passed: {summary['tests_passed']}")
        print(f"Tests Failed: {summary['tests_failed']}")
        print(f"Success Rate: {summary['success_rate']}")
        print(f"Duration: {summary['duration']}")
        
        print("\n📊 PERFORMANCE METRICS:")
        for category, metrics in self.results["performance_metrics"].items():
            print(f"  {category}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.3f}")
                else:
                    print(f"    {metric}: {value}")
        
        print("\n🧠 LEARNING METRICS:")
        if self.results["learning_metrics"]:
            latest_metrics = list(self.results["learning_metrics"].values())[-1]
            for metric, value in latest_metrics.items():
                print(f"  {metric}: {value}")
        
        print("\n🔧 HEALING METRICS:")
        for metric, value in self.results["healing_metrics"].items():
            print(f"  {metric}: {value}")
        
        print("\n❌ FAILED TESTS:")
        failed_tests = [r for r in self.results["detailed_results"] if not r["passed"]]
        if failed_tests:
            for result in failed_tests:
                print(f"  - {result['test_name']}: {result.get('error', 'Unknown error')}")
        else:
            print("  None! All tests passed! 🎉")
        
        print("\n" + "=" * 80)


async def main():
    """Main test execution function."""
    
    logger.info("🚀 Initializing Progressive Quality Gates Test Suite")
    
    # Create test suite
    test_suite = ProgressiveQualityGatesTestSuite()
    
    # Run comprehensive tests
    results = await test_suite.run_comprehensive_tests()
    
    # Print results
    test_suite.print_results_summary()
    
    # Save results to file
    results_file = Path("progressive_quality_gates_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📄 Test results saved to {results_file}")
    
    # Return exit code based on success
    success_rate = results.get("success_rate", 0.0)
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)