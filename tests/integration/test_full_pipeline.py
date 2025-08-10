"""Integration tests for complete model card generation pipeline."""

import asyncio
import json
import tempfile
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modelcard_generator.core.enhanced_generator import EnhancedModelCardGenerator
from modelcard_generator.core.models import CardConfig, CardFormat
from modelcard_generator.core.distributed_cache import MultiLevelCache
from modelcard_generator.core.async_executor import PriorityAsyncExecutor, TaskPriority
from modelcard_generator.core.resource_optimizer import ResourceManager, OptimizationStrategy


class TestFullPipeline(unittest.TestCase):
    """Test complete model card generation pipeline."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = CardConfig(
            format=CardFormat.HUGGINGFACE,
            auto_populate=True,
            include_ethical_considerations=True,
            include_carbon_footprint=True
        )
    
    def tearDown(self):
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end model card generation."""
        # Create test data files
        eval_data = {
            "model_name": "end-to-end-test",
            "accuracy": 0.94,
            "precision": 0.92,
            "recall": 0.90,
            "f1_score": 0.91,
            "inference_time_ms": 25,
            "dataset": ["train_set", "validation_set"]
        }
        
        config_data = {
            "name": "end-to-end-test",
            "version": "2.1.0",
            "description": "Test model for end-to-end pipeline",
            "license": "MIT",
            "authors": ["Test Author 1", "Test Author 2"],
            "framework": "PyTorch",
            "architecture": "transformer",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10
            }
        }
        
        dataset_data = {
            "datasets": ["custom_dataset_v1", "public_benchmark"],
            "training_data": ["training_corpus"],
            "preprocessing": "Tokenization, normalization, padding to 512 tokens",
            "bias_analysis": {
                "bias_risks": [
                    "Potential gender bias in text generation",
                    "Language model may reflect training data biases"
                ],
                "fairness_metrics": {
                    "demographic_parity": 0.95,
                    "equalized_odds": 0.93
                }
            }
        }
        
        # Write test files
        eval_file = self.temp_dir / "eval.json"
        config_file = self.temp_dir / "config.yaml"
        dataset_file = self.temp_dir / "dataset.json"
        
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f)
        
        # Generate model card
        generator = EnhancedModelCardGenerator(self.config)
        card = generator.generate(
            eval_results=str(eval_file),
            model_config=str(config_file),
            dataset_info=str(dataset_file)
        )
        
        # Verify card contents
        self.assertEqual(card.model_details.name, "end-to-end-test")
        self.assertEqual(card.model_details.version, "2.1.0")
        self.assertEqual(card.model_details.license, "MIT")
        self.assertEqual(len(card.model_details.authors), 2)
        self.assertEqual(card.training_details.framework, "PyTorch")
        self.assertEqual(card.training_details.model_architecture, "transformer")
        
        # Check metrics were extracted
        metric_names = [m.name for m in card.evaluation_results]
        self.assertIn("accuracy", metric_names)
        self.assertIn("f1_score", metric_names)
        self.assertIn("inference_time_ms", metric_names)
        
        # Check bias analysis was included
        self.assertTrue(len(card.ethical_considerations.bias_risks) >= 2)
        self.assertIn("demographic_parity", card.ethical_considerations.fairness_metrics)
        
        # Check datasets were added
        self.assertIn("custom_dataset_v1", card.model_details.datasets)
        self.assertIn("training_corpus", card.training_details.training_data)
        
        # Verify rendering
        markdown = card.render("markdown")
        self.assertIn("# end-to-end-test", markdown)
        self.assertIn("**accuracy**: 0.94", markdown)
        self.assertIn("PyTorch", markdown)
        
        json_output = card.render("json")
        json_data = json.loads(json_output)
        self.assertEqual(json_data["model_details"]["name"], "end-to-end-test")
    
    def test_async_pipeline(self):
        """Test async pipeline with multiple data sources."""
        async def run_async_test():
            # Create multiple test files
            files_data = [
                ("eval1.json", {"model_name": "async-1", "accuracy": 0.91}),
                ("eval2.json", {"model_name": "async-2", "accuracy": 0.88}),
                ("eval3.json", {"model_name": "async-3", "accuracy": 0.93})
            ]
            
            file_paths = []
            for filename, data in files_data:
                file_path = self.temp_dir / filename
                with open(file_path, 'w') as f:
                    json.dump(data, f)
                file_paths.append(str(file_path))
            
            # Generate cards asynchronously
            generator = EnhancedModelCardGenerator(self.config)
            tasks = [
                {"eval_results": path} for path in file_paths
            ]
            
            cards = await generator.generate_batch_async(tasks, max_concurrent=2)
            
            # Verify results
            self.assertEqual(len(cards), 3)
            for i, card in enumerate(cards):
                self.assertEqual(card.model_details.name, f"async-{i+1}")
                
                # Check that metrics were extracted
                accuracy_metrics = [m for m in card.evaluation_results if m.name == "accuracy"]
                self.assertEqual(len(accuracy_metrics), 1)
        
        asyncio.run(run_async_test())
    
    def test_error_recovery(self):
        """Test pipeline error recovery and partial success."""
        async def run_error_test():
            # Mix of valid and invalid data sources
            tasks = [
                {"eval_results": {"model_name": "good-1", "accuracy": 0.90}},  # Valid
                {"eval_results": "/nonexistent/file.json"},  # Invalid file
                {"eval_results": {"model_name": "good-2", "f1_score": 0.85}},  # Valid
                {"eval_results": "not-a-file-path"}  # Invalid
            ]
            
            generator = EnhancedModelCardGenerator(self.config)
            
            # This should process valid tasks and skip invalid ones
            cards = await generator.generate_batch_async(tasks, max_concurrent=2)
            
            # Should get only the valid cards
            self.assertGreater(len(cards), 0)  # At least some should succeed
            self.assertLessEqual(len(cards), 4)  # Not all may succeed
            
            # Check that valid cards are properly formed
            for card in cards:
                self.assertTrue(card.model_details.name)  # Should have a name
        
        asyncio.run(run_error_test())
    
    def test_performance_optimization(self):
        """Test pipeline with performance optimizations."""
        # Generate multiple cards to test performance features
        generator = EnhancedModelCardGenerator(self.config)
        
        # Generate several cards
        start_time = time.time()
        cards = []
        
        for i in range(5):
            card = generator.generate(
                eval_results={
                    "model_name": f"perf-test-{i}",
                    "accuracy": 0.90 + (i * 0.01),
                    "metrics_count": i * 10
                }
            )
            cards.append(card)
        
        generation_time = time.time() - start_time
        
        # Check that all cards were generated
        self.assertEqual(len(cards), 5)
        
        # Check generation statistics
        stats = generator.get_generation_statistics()
        self.assertEqual(stats["total_generated"], 5)
        self.assertEqual(stats["total_failures"], 0)
        self.assertEqual(stats["success_rate"], 1.0)
        self.assertGreater(stats["avg_generation_time"], 0)
        
        # Get performance report
        report = generator.get_performance_report()
        self.assertIn("generation_stats", report)
        self.assertIn("performance_stats", report)
        
        print(f"Generated {len(cards)} cards in {generation_time:.2f}s")
        print(f"Average time per card: {generation_time/len(cards):.3f}s")
    
    def test_multi_format_generation(self):
        """Test generation in multiple formats."""
        eval_data = {
            "model_name": "multi-format-test",
            "accuracy": 0.92,
            "precision": 0.89,
            "version": "1.0.0"
        }
        
        formats = [
            CardFormat.HUGGINGFACE,
            CardFormat.GOOGLE,
            CardFormat.EU_CRA
        ]
        
        for card_format in formats:
            config = CardConfig(format=card_format)
            generator = EnhancedModelCardGenerator(config)
            
            card = generator.generate(eval_results=eval_data)
            
            self.assertEqual(card.config.format, card_format)
            self.assertEqual(card.model_details.name, "multi-format-test")
            
            # Test rendering
            markdown = card.render("markdown")
            self.assertIn("multi-format-test", markdown)
            
            json_output = card.render("json")
            json_data = json.loads(json_output)
            self.assertEqual(json_data["model_details"]["name"], "multi-format-test")
    
    def test_compliance_features(self):
        """Test compliance and ethical considerations."""
        eval_data = {
            "model_name": "compliance-test",
            "accuracy": 0.94
        }
        
        # Test with ethical considerations enabled
        config = CardConfig(
            format=CardFormat.EU_CRA,
            include_ethical_considerations=True,
            include_carbon_footprint=True,
            regulatory_standard="eu_ai_act"
        )
        
        generator = EnhancedModelCardGenerator(config)
        card = generator.generate(eval_results=eval_data)
        
        # Should have ethical considerations
        self.assertTrue(len(card.ethical_considerations.bias_risks) > 0)
        self.assertTrue(len(card.ethical_considerations.bias_mitigation) > 0)
        
        # Should have compliance info
        self.assertIn(config.regulatory_standard, card.compliance_info)
        
        # Test rendering includes compliance sections
        markdown = card.render("markdown")
        self.assertIn("Ethical Considerations", markdown)
        self.assertIn("Compliance Information", markdown)


class TestScalabilityFeatures(unittest.TestCase):
    """Test scalability and performance features."""
    
    def test_caching_integration(self):
        """Test caching functionality."""
        # This would test the distributed cache integration
        # For now, we'll test the basic functionality
        
        cache = MultiLevelCache(l1_size=10)
        
        async def test_cache():
            # Test basic cache operations
            await cache.put("test_key", {"data": "test_value"})
            result = await cache.get("test_key")
            
            self.assertEqual(result["data"], "test_value")
            
            # Test cache stats
            stats = await cache.get_stats()
            self.assertIn("l1", stats)
            self.assertGreater(stats["overall_hit_rate"], 0)
        
        asyncio.run(test_cache())
    
    def test_async_executor(self):
        """Test async execution capabilities."""
        async def test_executor():
            executor = PriorityAsyncExecutor(max_concurrent=3)
            await executor.start()
            
            try:
                # Submit tasks with different priorities
                tasks = []
                for i in range(5):
                    priority = TaskPriority.HIGH if i < 2 else TaskPriority.NORMAL
                    task_id = await executor.submit(
                        self._dummy_async_task(f"task-{i}"),
                        priority=priority
                    )
                    tasks.append(task_id)
                
                # Wait for all tasks to complete
                results = []
                for task_id in tasks:
                    result = await executor.get_result(task_id, timeout=10)
                    results.append(result)
                
                self.assertEqual(len(results), 5)
                
                # Check executor stats
                stats = executor.get_stats()
                self.assertEqual(stats["total_submitted"], 5)
                self.assertTrue(stats["running"])
                
            finally:
                await executor.stop()
        
        asyncio.run(test_executor())
    
    async def _dummy_async_task(self, name: str) -> str:
        """Dummy async task for testing."""
        await asyncio.sleep(0.1)
        return f"completed_{name}"
    
    def test_resource_optimization(self):
        """Test resource optimization features."""
        # Test basic resource monitoring setup
        from modelcard_generator.core.resource_optimizer import ResourceMonitor, OptimizationStrategy
        
        strategy = OptimizationStrategy(
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            memory_high_threshold=75.0
        )
        
        monitor = ResourceMonitor(monitoring_interval=1, history_size=10)
        
        async def test_monitoring():
            await monitor.start_monitoring()
            
            # Let it collect a few data points
            await asyncio.sleep(2)
            
            current_metrics = monitor.get_current_metrics()
            self.assertIsNotNone(current_metrics)
            self.assertGreater(current_metrics.timestamp, 0)
            
            # Test trends analysis
            trends = monitor.get_resource_trends()
            # Should have trend data after collecting metrics
            
            await monitor.stop_monitoring()
        
        asyncio.run(test_monitoring())


import time

if __name__ == '__main__':
    unittest.main(verbosity=2)