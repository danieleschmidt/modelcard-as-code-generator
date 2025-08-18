"""
Property-based tests for Model Card Generator using Hypothesis.

These tests generate random inputs to discover edge cases and ensure
the system behaves correctly under all conditions.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from hypothesis import given, strategies as st, assume, example
from hypothesis import HealthCheck, settings

from modelcard_generator.core.generator import ModelCardGenerator
from modelcard_generator.core.validator import ModelCardValidator
from modelcard_generator.core.models import ModelCardData


# Custom strategies for model card data
@st.composite
def model_metrics(draw):
    """Generate realistic model metrics."""
    return {
        "accuracy": draw(st.floats(min_value=0.0, max_value=1.0)),
        "precision": draw(st.floats(min_value=0.0, max_value=1.0)),
        "recall": draw(st.floats(min_value=0.0, max_value=1.0)),
        "f1_score": draw(st.floats(min_value=0.0, max_value=1.0)),
        "loss": draw(st.floats(min_value=0.0, max_value=10.0)),
    }


@st.composite
def model_name_strategy(draw):
    """Generate valid model names."""
    name = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1,
        max_size=50
    ).filter(lambda x: x[0].isalpha() if x else False))
    return name.replace(" ", "-").lower()


@st.composite
def eval_results_strategy(draw):
    """Generate evaluation results with realistic constraints."""
    return {
        "model_name": draw(model_name_strategy()),
        "model_version": draw(st.text(
            alphabet="0123456789.",
            min_size=3,
            max_size=10
        ).filter(lambda x: x.count(".") <= 2)),
        "metrics": draw(model_metrics()),
        "dataset": {
            "name": draw(st.text(min_size=1, max_size=50)),
            "size": draw(st.integers(min_value=100, max_value=1000000)),
            "splits": draw(st.lists(
                st.sampled_from(["train", "validation", "test", "dev"]),
                min_size=1,
                max_size=4,
                unique=True
            ))
        },
        "training_time": draw(st.integers(min_value=1, max_value=86400)),
        "inference_time_ms": draw(st.floats(min_value=0.1, max_value=10000.0))
    }


class TestModelCardGenerationProperties:
    """Property-based tests for model card generation."""
    
    @given(eval_results_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_generated_card_is_valid_markdown(self, eval_results):
        """Generated cards should always be valid markdown."""
        generator = ModelCardGenerator()
        card = generator.generate(eval_results=eval_results, format="huggingface")
        
        # Should not be empty
        assert len(card.content.strip()) > 0
        
        # Should contain markdown headers
        assert any(line.startswith("#") for line in card.content.split("\n"))
        
        # Should not contain unrendered template variables
        assert "{{" not in card.content
        assert "}}" not in card.content
    
    @given(eval_results_strategy())
    def test_generation_deterministic(self, eval_results):
        """Same inputs should produce identical outputs."""
        generator = ModelCardGenerator()
        
        card1 = generator.generate(eval_results=eval_results, format="huggingface")
        card2 = generator.generate(eval_results=eval_results, format="huggingface")
        
        assert card1.content == card2.content
    
    @given(eval_results_strategy())
    def test_metrics_preserved_in_output(self, eval_results):
        """All metrics should appear in the generated card."""
        generator = ModelCardGenerator()
        card = generator.generate(eval_results=eval_results, format="huggingface")
        
        for metric_name, metric_value in eval_results["metrics"].items():
            # Metric name should appear in card
            assert metric_name.replace("_", " ").title() in card.content
            
            # Metric value should appear (allowing for formatting differences)
            if isinstance(metric_value, float):
                # Check for percentage or decimal representation
                percentage = f"{metric_value * 100:.1f}%"
                decimal = f"{metric_value:.3f}"
                assert percentage in card.content or decimal in card.content
    
    @given(
        eval_results_strategy(),
        st.sampled_from(["huggingface", "google", "eu_cra"])
    )
    def test_all_formats_generate_valid_cards(self, eval_results, format_type):
        """All supported formats should generate valid cards."""
        generator = ModelCardGenerator()
        
        try:
            card = generator.generate(eval_results=eval_results, format=format_type)
            
            # Basic validity checks
            assert isinstance(card.content, str)
            assert len(card.content.strip()) > 0
            assert card.format == format_type
            
            # Format-specific checks
            if format_type == "huggingface":
                assert "---" in card.content  # YAML frontmatter
            elif format_type == "google":
                assert "Model Card" in card.content
            elif format_type == "eu_cra":
                assert "Intended Purpose" in card.content or "intended purpose" in card.content
                
        except NotImplementedError:
            # Some formats might not be fully implemented yet
            pytest.skip(f"Format {format_type} not yet implemented")
    
    @given(st.text(min_size=0, max_size=1000))
    def test_model_name_sanitization(self, raw_name):
        """Model names should be properly sanitized."""
        assume(len(raw_name.strip()) > 0)  # Skip empty names
        
        eval_results = {
            "model_name": raw_name,
            "model_version": "1.0.0",
            "metrics": {"accuracy": 0.95}
        }
        
        generator = ModelCardGenerator()
        card = generator.generate(eval_results=eval_results, format="huggingface")
        
        # Generated card should not contain problematic characters
        problematic_chars = ["<", ">", "&", "\"", "'"]
        for char in problematic_chars:
            if char in raw_name:
                # Should be escaped or replaced in output
                assert char not in card.content or f"&{char};" in card.content


class TestModelCardValidationProperties:
    """Property-based tests for model card validation."""
    
    @given(eval_results_strategy())
    def test_generated_cards_pass_validation(self, eval_results):
        """All generated cards should pass their format validation."""
        generator = ModelCardGenerator()
        validator = ModelCardValidator()
        
        for format_type in ["huggingface", "google"]:
            try:
                card = generator.generate(eval_results=eval_results, format=format_type)
                result = validator.validate(card.content, format=format_type)
                
                assert result.is_valid, f"Generated {format_type} card failed validation: {result.errors}"
                
            except NotImplementedError:
                pytest.skip(f"Format {format_type} not yet implemented")
    
    @given(st.text())
    def test_validator_handles_malformed_input(self, malformed_content):
        """Validator should gracefully handle any input without crashing."""
        validator = ModelCardValidator()
        
        # Should not raise exceptions, even with malformed input
        try:
            result = validator.validate(malformed_content, format="huggingface")
            assert isinstance(result.is_valid, bool)
            assert isinstance(result.errors, list)
        except Exception as e:
            pytest.fail(f"Validator crashed on input: {repr(malformed_content[:100])} - {e}")


class TestDataStructureProperties:
    """Property-based tests for data structure handling."""
    
    @given(st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.integers(),
            st.text()
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=10),
            st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=10)
        ),
        max_leaves=50
    ))
    def test_data_serialization_roundtrip(self, data):
        """Data should survive serialization/deserialization cycles."""
        from modelcard_generator.core.serializer import serialize_data, deserialize_data
        
        # Skip problematic data types that can't be JSON serialized
        assume(self._is_json_serializable(data))
        
        serialized = serialize_data(data)
        deserialized = deserialize_data(serialized)
        
        assert deserialized == data
    
    @staticmethod
    def _is_json_serializable(obj) -> bool:
        """Check if an object can be JSON serialized."""
        try:
            import json
            json.dumps(obj)
            return True
        except (TypeError, ValueError, OverflowError):
            return False


class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""
    
    @given(st.integers(min_value=1, max_value=1000))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_generation_time_scales_reasonably(self, data_size):
        """Generation time should scale reasonably with input size."""
        import time
        
        # Generate proportionally sized data
        large_metrics = {f"metric_{i}": 0.95 for i in range(data_size)}
        
        eval_results = {
            "model_name": "test-model",
            "model_version": "1.0.0",
            "metrics": large_metrics
        }
        
        generator = ModelCardGenerator()
        
        start_time = time.perf_counter()
        card = generator.generate(eval_results=eval_results, format="huggingface")
        end_time = time.perf_counter()
        
        generation_time = end_time - start_time
        
        # Should complete within reasonable time (allowing for test environment variability)
        max_time = 5.0 + (data_size * 0.001)  # Base time + linear scaling
        assert generation_time < max_time, f"Generation took {generation_time:.2f}s for {data_size} metrics"
        
        # Should produce non-empty output
        assert len(card.content) > 100
    
    @given(st.lists(eval_results_strategy(), min_size=1, max_size=10))
    def test_memory_usage_bounded(self, eval_results_list):
        """Memory usage should be bounded for batch operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        generator = ModelCardGenerator()
        cards = []
        
        for eval_results in eval_results_list:
            card = generator.generate(eval_results=eval_results, format="huggingface")
            cards.append(card)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (allowing for test environment)
        # Rough estimate: each card shouldn't use more than 1MB
        max_expected_increase = len(eval_results_list) * 1024 * 1024  # 1MB per card
        assert memory_increase < max_expected_increase * 2, f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB for {len(eval_results_list)} cards"


# Example-based tests for edge cases discovered through property testing
class TestDiscoveredEdgeCases:
    """Tests for specific edge cases discovered through property-based testing."""
    
    def test_empty_metrics_dict(self):
        """Handle empty metrics gracefully."""
        eval_results = {
            "model_name": "test-model",
            "model_version": "1.0.0",
            "metrics": {}
        }
        
        generator = ModelCardGenerator()
        card = generator.generate(eval_results=eval_results, format="huggingface")
        
        # Should generate valid card even without metrics
        assert len(card.content) > 0
        assert "# " in card.content  # Should have at least one header
    
    def test_unicode_in_model_name(self):
        """Handle Unicode characters in model names."""
        eval_results = {
            "model_name": "测试模型-émojï-🤖",
            "model_version": "1.0.0",
            "metrics": {"accuracy": 0.95}
        }
        
        generator = ModelCardGenerator()
        card = generator.generate(eval_results=eval_results, format="huggingface")
        
        # Should handle Unicode without errors
        assert len(card.content) > 0
        # Unicode should be preserved or safely encoded
        assert "测试模型" in card.content or "test" in card.content.lower()
    
    @example(0.0)
    @example(1.0)
    @example(0.5)
    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_edge_metric_values(self, metric_value):
        """Handle edge cases in metric values."""
        eval_results = {
            "model_name": "test-model",
            "model_version": "1.0.0",
            "metrics": {"accuracy": metric_value}
        }
        
        generator = ModelCardGenerator()
        card = generator.generate(eval_results=eval_results, format="huggingface")
        
        # Should handle all valid metric values
        assert len(card.content) > 0
        assert str(metric_value) in card.content or f"{metric_value:.1%}" in card.content