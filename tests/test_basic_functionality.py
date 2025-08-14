"""Basic functionality tests for the model card generator."""

import json
import tempfile
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcard_generator.core.generator import ModelCardGenerator
from modelcard_generator.core.models import CardConfig, CardFormat


def test_basic_generator_creation():
    """Test that we can create a basic generator."""
    config = CardConfig(format=CardFormat.HUGGINGFACE)
    generator = ModelCardGenerator(config)
    assert generator is not None
    assert generator.config.format == CardFormat.HUGGINGFACE


def test_basic_card_generation():
    """Test basic card generation without external dependencies."""
    config = CardConfig(format=CardFormat.HUGGINGFACE)
    generator = ModelCardGenerator(config)
    
    # Create minimal test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "model_name": "test-model",
            "version": "1.0.0"
        }
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        # Generate card
        card = generator.generate(eval_results=temp_file)
        assert card is not None
        assert card.model_details.name == "test-model"
        
        # Test rendering
        content = card.render()
        assert "test-model" in content
        assert "1.0.0" in content
        
    finally:
        Path(temp_file).unlink()


def test_cli_import():
    """Test that CLI module can be imported."""
    from modelcard_generator.cli.main import cli
    assert cli is not None


if __name__ == "__main__":
    print("Running basic functionality tests...")
    
    try:
        test_basic_generator_creation()
        print("✅ Generator creation test passed")
        
        test_basic_card_generation()
        print("✅ Basic card generation test passed")
        
        test_cli_import()
        print("✅ CLI import test passed")
        
        print("🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)