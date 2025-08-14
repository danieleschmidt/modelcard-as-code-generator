"""Advanced feature tests for production capabilities."""

import json
import tempfile
import sys
from pathlib import Path
import time

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcard_generator.core.generator import ModelCardGenerator
from modelcard_generator.core.models import CardConfig, CardFormat
from modelcard_generator.core.drift_detector import DriftDetector
from modelcard_generator.core.cache_simple import cache_manager


def test_performance_monitoring():
    """Test performance monitoring and metrics."""
    config = CardConfig(format=CardFormat.HUGGINGFACE)
    generator = ModelCardGenerator(config)
    
    # Time the generation process
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "model_name": "performance-test-model",
        }
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        card = generator.generate(eval_results=temp_file)
        generation_time = time.time() - start_time
        
        assert card is not None
        assert generation_time < 5.0  # Should complete in under 5 seconds
        print(f"✅ Generation completed in {generation_time:.2f}s")
        
    finally:
        Path(temp_file).unlink()


def test_caching_functionality():
    """Test intelligent caching system."""
    cache = cache_manager.get_cache()
    
    # Test cache operations
    test_key = "test_key"
    test_value = {"test": "data"}
    
    # Put and get
    cache.put(test_key, test_value, ttl_seconds=300)
    retrieved = cache.get(test_key)
    
    assert retrieved == test_value
    print("✅ Cache functionality working")


def test_batch_processing():
    """Test batch processing capabilities."""
    config = CardConfig(format=CardFormat.HUGGINGFACE)
    generator = ModelCardGenerator(config)
    
    # Create multiple test tasks
    tasks = []
    temp_files = []
    
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "accuracy": 0.9 + i * 0.01,
                "f1_score": 0.85 + i * 0.02,
                "model_name": f"batch-model-{i}",
            }
            json.dump(test_data, f)
            temp_files.append(f.name)
            tasks.append({"eval_results": f.name})
    
    try:
        start_time = time.time()
        results = generator.generate_batch(tasks, max_workers=2)
        batch_time = time.time() - start_time
        
        assert len(results) == 3
        assert all(card is not None for card in results)
        assert batch_time < 10.0  # Should complete efficiently
        
        print(f"✅ Batch processing completed {len(results)} cards in {batch_time:.2f}s")
        
    finally:
        for temp_file in temp_files:
            Path(temp_file).unlink()


def test_drift_detection():
    """Test drift detection capabilities."""
    # Create initial card
    config = CardConfig(format=CardFormat.HUGGINGFACE)
    generator = ModelCardGenerator(config)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        initial_data = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "model_name": "drift-test-model",
        }
        json.dump(initial_data, f)
        initial_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        new_data = {
            "accuracy": 0.88,  # Significant drop
            "f1_score": 0.90,
            "model_name": "drift-test-model",
        }
        json.dump(new_data, f)
        new_file = f.name
    
    try:
        card = generator.generate(eval_results=initial_file)
        
        # Test drift detection
        detector = DriftDetector()
        drift_report = detector.check(
            card=card,
            new_eval_results=new_file,
            thresholds={"accuracy": 0.02, "f1_score": 0.02}
        )
        
        assert drift_report.has_drift
        assert len(drift_report.significant_changes) > 0
        print("✅ Drift detection working correctly")
        
    finally:
        Path(initial_file).unlink()
        Path(new_file).unlink()


def test_security_features():
    """Test security scanning and sanitization."""
    config = CardConfig(format=CardFormat.HUGGINGFACE)
    generator = ModelCardGenerator(config)
    
    # Test with potentially malicious content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {
            "accuracy": 0.95,
            "model_name": "security-test-model",
            "password": "should_be_removed",  # Should be sanitized
            "api_key": "sk-1234567890",       # Should be detected
        }
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        card = generator.generate(eval_results=temp_file)
        content = card.render()
        
        # Check that sensitive data is not in output
        assert "should_be_removed" not in content
        assert "sk-1234567890" not in content
        print("✅ Security sanitization working")
        
    except Exception as e:
        # Security scan may prevent generation entirely
        if "Security scan failed" in str(e):
            print("✅ Security scanner blocked malicious content")
        else:
            raise
    finally:
        Path(temp_file).unlink()


if __name__ == "__main__":
    print("Running advanced feature tests...")
    
    try:
        test_performance_monitoring()
        test_caching_functionality()
        test_batch_processing()
        test_drift_detection()
        test_security_features()
        
        print("🎉 All advanced tests passed!")
        
    except Exception as e:
        print(f"❌ Advanced test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)