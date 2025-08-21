#!/usr/bin/env python3
"""Batch Processing Example."""

import time
from modelcard_generator import ModelCardGenerator, CardConfig, CardFormat

def main():
    print("🔄 Batch Processing Example")
    print("="*40)
    
    # Configure generator for batch processing
    config = CardConfig(
        format=CardFormat.HUGGINGFACE,
        auto_populate=True,
        include_ethical_considerations=True
    )
    
    generator = ModelCardGenerator(config)
    
    # Create batch tasks
    tasks = []
    for i in range(20):
        task = {
            "eval_results": {
                "accuracy": 0.90 + (i * 0.002),  # Gradually increasing
                "f1_score": 0.88 + (i * 0.003),
                "precision": 0.89 + (i * 0.001),
                "recall": 0.91 + (i * 0.001)
            },
            "model_name": f"batch-model-{i:02d}",
            "model_version": f"1.{i}.0",
            "authors": ["Batch Processing Team"],
            "license": "apache-2.0",
            "intended_use": f"Model {i} for batch processing demonstration"
        }
        tasks.append(task)
    
    print(f"📦 Processing {len(tasks)} model cards...")
    
    # Measure performance
    start_time = time.time()
    
    # Generate all cards in batch
    results = generator.generate_batch(tasks, max_workers=4)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n📊 Batch Processing Results:")
    print(f"✅ Generated: {len(results)} model cards")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Throughput: {len(results)/total_time:.1f} cards/second")
    print(f"⚡ Average time per card: {total_time/len(results)*1000:.1f}ms")
    
    # Save all cards
    for i, card in enumerate(results):
        card.save(f"examples/batch/MODEL_CARD_{i:02d}.md")
    
    print(f"\n💾 Saved {len(results)} model cards to examples/batch/")
    
    # Performance analysis
    print("\n📈 Performance Analysis:")
    print(f"   - Batch size: {len(tasks)}")
    print(f"   - Workers: 4")
    print(f"   - Total time: {total_time:.2f}s")
    print(f"   - Throughput: {len(results)/total_time:.1f} cards/sec")
    
    # Demonstrate sequential vs parallel performance
    print("\n🔄 Comparing sequential vs parallel processing...")
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for task in tasks[:5]:  # Just first 5 for comparison
        card = generator.generate(**task)
        sequential_results.append(card)
    sequential_time = time.time() - start_time
    
    # Parallel processing  
    start_time = time.time()
    parallel_results = generator.generate_batch(tasks[:5], max_workers=4)
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
    
    print(f"\n⚡ Performance Comparison (5 cards):")
    print(f"   Sequential: {sequential_time:.3f}s ({5/sequential_time:.1f} cards/sec)")
    print(f"   Parallel:   {parallel_time:.3f}s ({5/parallel_time:.1f} cards/sec)")
    print(f"   Speedup:    {speedup:.1f}x faster")
    
    print("\n🎉 Batch processing example completed!")

if __name__ == "__main__":
    # Create batch directory
    import os
    os.makedirs("examples/batch", exist_ok=True)
    
    main()
