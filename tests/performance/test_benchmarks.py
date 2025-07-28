"""Performance benchmarks for Model Card Generator.

These tests measure the performance characteristics of the model card generation
process to ensure it meets performance requirements and detect regressions.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from memory_profiler import memory_usage

# Import your model card generator classes here
# from modelcard_generator.core.generator import ModelCardGenerator
# from modelcard_generator.parsers.evaluation import EvaluationParser
# from modelcard_generator.templates.huggingface import HuggingFaceTemplate


class MockModelCardGenerator:
    """Mock generator for testing performance."""
    
    def __init__(self):
        self.generation_count = 0
    
    def generate(self, data: Dict[str, Any]) -> str:
        """Mock generate method with realistic delay."""
        # Simulate processing time
        time.sleep(0.1)
        self.generation_count += 1
        return f"# Model Card {self.generation_count}\n\nGenerated from data with {len(data)} fields."
    
    def validate(self, content: str) -> bool:
        """Mock validation with processing delay."""
        time.sleep(0.05)
        return len(content) > 10


@pytest.fixture
def large_dataset() -> Dict[str, Any]:
    """Generate large dataset for performance testing."""
    return {
        "model_name": "large-test-model",
        "metrics": {f"metric_{i}": 0.95 - (i * 0.001) for i in range(1000)},
        "training_data": {
            "samples": [f"sample_{i}" for i in range(10000)],
            "labels": list(range(1000)),
            "features": {f"feature_{i}": f"value_{i}" for i in range(500)}
        },
        "hyperparameters": {f"param_{i}": i * 0.01 for i in range(100)},
        "evaluation_results": {
            "confusion_matrix": [[100, 20], [15, 200]] * 100,
            "classification_report": {f"class_{i}": {"precision": 0.9, "recall": 0.85, "f1-score": 0.87} for i in range(50)}
        }
    }


@pytest.fixture
def memory_intensive_data() -> Dict[str, Any]:
    """Generate memory-intensive test data."""
    return {
        "model_name": "memory-test-model",
        "large_arrays": {
            f"array_{i}": list(range(10000)) for i in range(10)
        },
        "text_data": {
            "documents": ["This is a test document. " * 1000 for _ in range(100)],
            "tokenized": [["token"] * 1000 for _ in range(100)]
        },
        "nested_structure": {
            f"level_{i}": {
                f"sublevel_{j}": {
                    f"data_{k}": list(range(100))
                    for k in range(10)
                }
                for j in range(10)
            }
            for i in range(5)
        }
    }


@pytest.mark.performance
class TestGenerationPerformance:
    """Test performance of model card generation."""
    
    def test_single_card_generation_time(self, benchmark, sample_eval_results):
        """Benchmark time for single model card generation."""
        generator = MockModelCardGenerator()
        
        # Benchmark should complete in under 1 second
        result = benchmark.pedantic(
            generator.generate, 
            args=(sample_eval_results,),
            rounds=10,
            iterations=1
        )
        
        assert result is not None
        assert "Model Card" in result
        
        # Verify performance requirement
        assert benchmark.stats.mean < 1.0, "Generation should complete in under 1 second"
    
    def test_large_dataset_generation_time(self, benchmark, large_dataset):
        """Benchmark generation time with large dataset."""
        generator = MockModelCardGenerator()
        
        result = benchmark.pedantic(
            generator.generate,
            args=(large_dataset,),
            rounds=5,
            iterations=1
        )
        
        assert result is not None
        # Should handle large datasets within 5 seconds
        assert benchmark.stats.mean < 5.0, "Large dataset generation should complete in under 5 seconds"
    
    def test_batch_generation_performance(self, benchmark, sample_eval_results):
        """Benchmark batch generation of multiple cards."""
        generator = MockModelCardGenerator()
        
        def generate_batch(datasets: List[Dict[str, Any]]) -> List[str]:
            return [generator.generate(data) for data in datasets]
        
        datasets = [sample_eval_results.copy() for _ in range(10)]
        
        results = benchmark.pedantic(
            generate_batch,
            args=(datasets,),
            rounds=3,
            iterations=1
        )
        
        assert len(results) == 10
        # Batch of 10 should complete within 15 seconds
        assert benchmark.stats.mean < 15.0, "Batch generation should be efficient"
    
    def test_concurrent_generation_performance(self, benchmark, sample_eval_results):
        """Test performance under concurrent load."""
        import threading
        import queue
        
        generator = MockModelCardGenerator()
        results_queue = queue.Queue()
        
        def worker():
            result = generator.generate(sample_eval_results)
            results_queue.put(result)
        
        def run_concurrent_generation(num_threads: int = 5):
            threads = []
            for _ in range(num_threads):
                thread = threading.Thread(target=worker)
                threads.append(thread)
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
            
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            return results
        
        results = benchmark.pedantic(
            run_concurrent_generation,
            args=(5,),
            rounds=3,
            iterations=1
        )
        
        assert len(results) == 5
        # Concurrent generation should scale well
        assert benchmark.stats.mean < 10.0, "Concurrent generation should be efficient"


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage characteristics."""
    
    def test_memory_usage_single_generation(self, sample_eval_results):
        """Test memory usage for single card generation."""
        generator = MockModelCardGenerator()
        
        def generate_card():
            return generator.generate(sample_eval_results)
        
        # Measure memory usage
        memory_before = memory_usage()[0]
        memory_during = memory_usage((generate_card, ()))
        memory_after = memory_usage()[0]
        
        peak_memory = max(memory_during)
        memory_growth = peak_memory - memory_before
        
        # Memory growth should be reasonable (< 100MB for single card)
        assert memory_growth < 100, f"Memory growth {memory_growth}MB exceeds limit"
        
        # Memory should be released after generation
        memory_leak = memory_after - memory_before
        assert abs(memory_leak) < 10, f"Potential memory leak: {memory_leak}MB"
    
    def test_memory_usage_large_dataset(self, memory_intensive_data):
        """Test memory usage with memory-intensive data."""
        generator = MockModelCardGenerator()
        
        def generate_large_card():
            return generator.generate(memory_intensive_data)
        
        memory_before = memory_usage()[0]
        memory_during = memory_usage((generate_large_card, ()))
        memory_after = memory_usage()[0]
        
        peak_memory = max(memory_during)
        memory_growth = peak_memory - memory_before
        
        # Even with large datasets, memory should stay under 500MB
        assert memory_growth < 500, f"Memory usage {memory_growth}MB exceeds limit for large datasets"
    
    def test_memory_usage_batch_processing(self, sample_eval_results):
        """Test memory usage during batch processing."""
        generator = MockModelCardGenerator()
        
        def generate_batch():
            datasets = [sample_eval_results.copy() for _ in range(20)]
            return [generator.generate(data) for data in datasets]
        
        memory_before = memory_usage()[0]
        memory_during = memory_usage((generate_batch, ()))
        memory_after = memory_usage()[0]
        
        peak_memory = max(memory_during)
        memory_growth = peak_memory - memory_before
        
        # Batch processing should have linear memory growth
        assert memory_growth < 200, f"Batch memory usage {memory_growth}MB exceeds limit"
        
        # Check for memory leaks in batch processing
        memory_leak = memory_after - memory_before
        assert abs(memory_leak) < 20, f"Batch processing memory leak: {memory_leak}MB"


@pytest.mark.performance
class TestScalabilityTests:
    """Test scalability characteristics."""
    
    def test_linear_scaling_with_data_size(self, benchmark):
        """Test that generation time scales linearly with data size."""
        generator = MockModelCardGenerator()
        
        # Test different data sizes
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            data = {
                "model_name": f"test-model-{size}",
                "metrics": {f"metric_{i}": 0.95 for i in range(size)}
            }
            
            # Use benchmark to measure time
            result = benchmark.pedantic(
                generator.generate,
                args=(data,),
                rounds=3,
                iterations=1
            )
            
            times.append(benchmark.stats.mean)
            assert result is not None
        
        # Check that scaling is roughly linear (allow for some variance)
        # Time should not increase exponentially
        for i in range(1, len(times)):
            scaling_factor = times[i] / times[0]
            size_factor = sizes[i] / sizes[0]
            
            # Scaling should be roughly proportional (within 2x factor)
            assert scaling_factor < size_factor * 2, f"Non-linear scaling detected: {scaling_factor} vs {size_factor}"
    
    def test_file_size_scaling(self, benchmark, temp_dir):
        """Test performance with different file sizes."""
        generator = MockModelCardGenerator()
        
        file_sizes = [1000, 10000, 100000]  # bytes
        
        for size in file_sizes:
            # Create test file of specific size
            test_data = {"content": "x" * size}
            
            result = benchmark.pedantic(
                generator.generate,
                args=(test_data,),
                rounds=3,
                iterations=1
            )
            
            assert result is not None
            # Even large files should process within reasonable time
            assert benchmark.stats.mean < 10.0, f"File size {size} took too long: {benchmark.stats.mean}s"


@pytest.mark.performance
class TestValidationPerformance:
    """Test performance of validation operations."""
    
    def test_validation_speed(self, benchmark):
        """Benchmark validation performance."""
        generator = MockModelCardGenerator()
        test_content = "# Test Model Card\n\n" + "Test content.\n" * 1000
        
        result = benchmark.pedantic(
            generator.validate,
            args=(test_content,),
            rounds=10,
            iterations=1
        )
        
        assert result is True
        # Validation should be fast
        assert benchmark.stats.mean < 0.5, "Validation should complete quickly"
    
    def test_large_content_validation(self, benchmark):
        """Test validation performance with large content."""
        generator = MockModelCardGenerator()
        
        # Generate large model card content
        large_content = "# Large Model Card\n\n" + "Content line.\n" * 10000
        
        result = benchmark.pedantic(
            generator.validate,
            args=(large_content,),
            rounds=5,
            iterations=1
        )
        
        assert result is True
        # Even large content should validate quickly
        assert benchmark.stats.mean < 2.0, "Large content validation should be efficient"


@pytest.mark.performance
class TestRegressionTests:
    """Performance regression tests."""
    
    def test_generation_time_regression(self, benchmark, sample_eval_results):
        """Ensure generation time hasn't regressed."""
        generator = MockModelCardGenerator()
        
        result = benchmark.pedantic(
            generator.generate,
            args=(sample_eval_results,),
            rounds=10,
            iterations=1
        )
        
        assert result is not None
        
        # Baseline performance requirement (adjust based on your actual performance)
        baseline_time = 1.0  # seconds
        assert benchmark.stats.mean < baseline_time, f"Performance regression detected: {benchmark.stats.mean}s > {baseline_time}s"
    
    def test_memory_regression(self, sample_eval_results):
        """Ensure memory usage hasn't regressed."""
        generator = MockModelCardGenerator()
        
        def generate_and_measure():
            return generator.generate(sample_eval_results)
        
        memory_usage_values = memory_usage((generate_and_measure, ()))
        peak_memory = max(memory_usage_values)
        baseline_memory = memory_usage()[0]
        
        memory_delta = peak_memory - baseline_memory
        
        # Baseline memory requirement (adjust based on your actual usage)
        baseline_memory_limit = 50  # MB
        assert memory_delta < baseline_memory_limit, f"Memory regression detected: {memory_delta}MB > {baseline_memory_limit}MB"


@pytest.mark.performance
class TestStressTests:
    """Stress tests for extreme conditions."""
    
    def test_rapid_successive_generations(self, sample_eval_results):
        """Test rapid successive model card generations."""
        generator = MockModelCardGenerator()
        
        start_time = time.time()
        results = []
        
        # Generate 100 cards rapidly
        for i in range(100):
            data = sample_eval_results.copy()
            data["iteration"] = i
            result = generator.generate(data)
            results.append(result)
        
        total_time = time.time() - start_time
        
        assert len(results) == 100
        # Should handle rapid generation efficiently
        assert total_time < 60, f"Rapid generation took too long: {total_time}s"
        
        # Check average time per generation
        avg_time = total_time / 100
        assert avg_time < 0.6, f"Average generation time too high: {avg_time}s"
    
    def test_extremely_large_input(self):
        """Test with extremely large input data."""
        generator = MockModelCardGenerator()
        
        # Create extremely large dataset
        huge_data = {
            "model_name": "stress-test-model",
            "massive_metrics": {f"metric_{i}": 0.95 for i in range(50000)},
            "huge_text": "Large text content. " * 100000,
            "big_arrays": [list(range(1000)) for _ in range(100)]
        }
        
        start_time = time.time()
        result = generator.generate(huge_data)
        generation_time = time.time() - start_time
        
        assert result is not None
        # Even extremely large inputs should complete within reasonable time
        assert generation_time < 30, f"Extremely large input took too long: {generation_time}s"


# Utility functions for performance testing
def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def measure_memory_usage(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    import tracemalloc
    
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, peak / 1024 / 1024  # Convert to MB


def profile_function(func, *args, **kwargs):
    """Profile a function and return timing and memory stats."""
    import cProfile
    import pstats
    import io
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Get profiling stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    return result, s.getvalue()
