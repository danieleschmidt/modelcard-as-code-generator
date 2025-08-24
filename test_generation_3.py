#!/usr/bin/env python3
"""Test Generation 3: Performance Optimization and Scaling Features."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.modelcard_generator import ModelCardGenerator, CardConfig, CardFormat
from src.modelcard_generator.core.quantum_performance_optimizer import QuantumInspiredOptimizer
from src.modelcard_generator.core.distributed_processing import DistributedProcessor
from src.modelcard_generator.core.intelligent_cache import IntelligentCacheManager
from src.modelcard_generator.core.resource_optimizer import ResourceOptimizer


async def test_performance_optimization():
    """Test performance optimization features."""
    print("⚡ Testing Generation 3: Performance Optimization & Scaling")
    
    # Initialize performance components with proper configuration
    from src.modelcard_generator.core.quantum_performance_optimizer import OptimizationConfig
    
    opt_config = OptimizationConfig()
    optimizer = QuantumInspiredOptimizer(opt_config)
    cache_manager = IntelligentCacheManager()
    resource_optimizer = ResourceOptimizer()
    
    # Test 1: Batch Processing Performance
    print("\n📦 Test 1: Batch Processing Performance")
    
    # Create multiple model card generation tasks
    tasks = []
    for i in range(10):
        task = {
            "eval_results": "test_data/eval_results.json",
            "model_config": "test_data/model_config.json", 
            "dataset_info": "test_data/dataset_info.json",
            "model_name": f"model-{i}",
            "model_version": f"1.{i}.0"
        }
        tasks.append(task)
    
    config = CardConfig(format=CardFormat.HUGGINGFACE, auto_populate=True)
    generator = ModelCardGenerator(config)
    
    # Test batch generation
    start_time = time.time()
    batch_results = generator.generate_batch(tasks, max_workers=4)
    batch_time = time.time() - start_time
    
    print(f"✅ Generated {len(batch_results)} model cards in {batch_time:.2f}s")
    print(f"📈 Throughput: {len(batch_results)/batch_time:.1f} cards/second")
    
    # Test 2: Intelligent Caching
    print("\n🧠 Test 2: Intelligent Caching Performance")
    
    # First generation (cold cache)
    start_time = time.time()
    card1 = generator.generate(eval_results="test_data/eval_results.json")
    cold_time = time.time() - start_time
    
    # Second generation (warm cache)
    start_time = time.time() 
    card2 = generator.generate(eval_results="test_data/eval_results.json")
    warm_time = time.time() - start_time
    
    speedup = cold_time / warm_time if warm_time > 0 else float('inf')
    print(f"🧊 Cold cache: {cold_time*1000:.1f}ms")
    print(f"🔥 Warm cache: {warm_time*1000:.1f}ms")
    print(f"🚀 Cache speedup: {speedup:.1f}x")
    
    # Test 3: Resource Optimization
    print("\n🔧 Test 3: Resource Optimization")
    
    # Get resource metrics
    metrics = resource_optimizer.get_resource_metrics()
    print(f"💾 Memory usage: {metrics.get('memory_usage_mb', 0):.1f} MB")
    print(f"⚙️  CPU usage: {metrics.get('cpu_percent', 0):.1f}%")
    print(f"🗂️  Open files: {metrics.get('open_files', 0)}")
    
    # Test optimization
    optimization_result = resource_optimizer.optimize_resources()
    print(f"🎯 Optimization applied: {optimization_result['optimizations_applied']}")
    
    # Test 4: Concurrent Processing
    print("\n🔄 Test 4: Concurrent Processing")
    
    async def generate_card_async(task_id):
        await asyncio.sleep(0.1)  # Simulate async work
        card = generator.generate(
            eval_results="test_data/eval_results.json",
            model_name=f"async-model-{task_id}"
        )
        return card
    
    # Test concurrent generation
    start_time = time.time()
    concurrent_tasks = [generate_card_async(i) for i in range(20)]
    concurrent_results = await asyncio.gather(*concurrent_tasks)
    concurrent_time = time.time() - start_time
    
    print(f"✅ Generated {len(concurrent_results)} cards concurrently in {concurrent_time:.2f}s")
    print(f"📊 Concurrent throughput: {len(concurrent_results)/concurrent_time:.1f} cards/second")
    
    # Test 5: Performance Monitoring
    print("\n📊 Test 5: Performance Monitoring")
    
    # Get performance insights
    perf_insights = optimizer.get_performance_insights()
    print(f"⏱️  Average generation time: {perf_insights.get('avg_generation_time_ms', 0):.1f}ms")
    print(f"🎯 Cache hit rate: {perf_insights.get('cache_hit_rate', 0):.1%}")
    print(f"📈 Throughput trend: {perf_insights.get('throughput_trend', 'stable')}")
    
    # Test 6: Memory Management
    print("\n🧮 Test 6: Memory Management")
    
    # Force garbage collection and optimize
    import gc
    gc.collect()
    
    # Test memory efficiency
    memory_before = resource_optimizer.get_memory_usage()
    
    # Generate many cards to test memory management
    for i in range(50):
        card = generator.generate(
            eval_results="test_data/eval_results.json",
            model_name=f"memory-test-{i}"
        )
        del card  # Explicit cleanup
    
    memory_after = resource_optimizer.get_memory_usage()
    print(f"🔢 Memory before: {memory_before:.1f} MB")
    print(f"🔢 Memory after: {memory_after:.1f} MB")
    print(f"📊 Memory delta: {memory_after - memory_before:+.1f} MB")
    
    # Test 7: Scale Testing
    print("\n📏 Test 7: Scale Testing")
    
    # Test with larger batches
    large_tasks = [
        {
            "eval_results": "test_data/eval_results.json",
            "model_name": f"scale-model-{i}",
            "model_version": f"2.{i}.0"
        }
        for i in range(100)
    ]
    
    start_time = time.time()
    large_batch_results = generator.generate_batch(large_tasks, max_workers=8)
    large_batch_time = time.time() - start_time
    
    print(f"🎯 Large batch: {len(large_batch_results)} cards in {large_batch_time:.2f}s")
    print(f"⚡ Large batch throughput: {len(large_batch_results)/large_batch_time:.1f} cards/second")
    
    # Performance Summary
    print("\n📋 Performance Summary")
    print("="*50)
    print(f"✅ Batch processing: {len(batch_results)/batch_time:.1f} cards/second")
    print(f"🚀 Cache speedup: {speedup:.1f}x faster")
    print(f"🔄 Concurrent processing: {len(concurrent_results)/concurrent_time:.1f} cards/second")
    print(f"📏 Large scale: {len(large_batch_results)/large_batch_time:.1f} cards/second")
    print(f"💾 Memory efficiency: {memory_after - memory_before:+.1f} MB delta")
    print("="*50)
    print("🎉 Generation 3 Performance Testing Complete!")
    
    return {
        "batch_throughput": len(batch_results)/batch_time,
        "cache_speedup": speedup,
        "concurrent_throughput": len(concurrent_results)/concurrent_time,
        "large_scale_throughput": len(large_batch_results)/large_batch_time,
        "memory_delta": memory_after - memory_before
    }


if __name__ == "__main__":
    results = asyncio.run(test_performance_optimization())
    
    # Save performance results
    with open("test_data/performance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Performance results saved to test_data/performance_results.json")