"""Comprehensive tests for enhanced model card generator."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modelcard_generator.core.enhanced_generator import EnhancedModelCardGenerator, EnhancedDataSourceParser
from modelcard_generator.core.models import ModelCard, CardConfig, CardFormat
from modelcard_generator.core.exceptions import ModelCardError, DataSourceError


class TestEnhancedDataSourceParser(unittest.TestCase):
    """Test enhanced data source parser."""
    
    def setUp(self):
        self.parser = EnhancedDataSourceParser()
    
    def tearDown(self):
        if hasattr(self.parser, 'executor'):
            self.parser.executor.shutdown(wait=True)
    
    def test_json_parsing_basic(self):
        """Test basic JSON parsing."""
        test_data = {
            "model_name": "test-model",
            "accuracy": 0.95,
            "f1_score": 0.93
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            result = self.parser.parse_json(temp_path)
            self.assertEqual(result['model_name'], 'test-model')
            self.assertEqual(result['accuracy'], 0.95)
            self.assertEqual(result['f1_score'], 0.93)
        finally:
            Path(temp_path).unlink()
    
    def test_json_parsing_async(self):
        """Test async JSON parsing."""
        async def run_test():
            test_data = {"test": "value", "score": 0.88}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_data, f)
                temp_path = f.name
            
            try:
                result = await self.parser.parse_json_async(temp_path)
                self.assertEqual(result['test'], 'value')
                self.assertEqual(result['score'], 0.88)
            finally:
                Path(temp_path).unlink()
        
        asyncio.run(run_test())
    
    def test_invalid_file_format(self):
        """Test handling of invalid file formats."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'not json content')
            temp_path = f.name
        
        try:
            with self.assertRaises(DataSourceError):
                self.parser.parse_json(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{ invalid json')
            temp_path = f.name
        
        try:
            with self.assertRaises(DataSourceError):
                self.parser.parse_json(temp_path)
        finally:
            Path(temp_path).unlink()


class TestEnhancedModelCardGenerator(unittest.TestCase):
    """Test enhanced model card generator."""
    
    def setUp(self):
        self.config = CardConfig(
            format=CardFormat.HUGGINGFACE,
            auto_populate=True,
            include_ethical_considerations=True
        )
        self.generator = EnhancedModelCardGenerator(self.config)
    
    def test_basic_card_generation(self):
        """Test basic model card generation."""
        eval_data = {
            "model_name": "test-classifier",
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.87,
            "f1_score": 0.88
        }
        
        card = self.generator.generate(eval_results=eval_data)
        
        self.assertIsInstance(card, ModelCard)
        self.assertEqual(card.model_details.name, "test-classifier")
        self.assertEqual(len(card.evaluation_results), 4)
        
        # Check metrics were added
        metric_names = [m.name for m in card.evaluation_results]
        self.assertIn("accuracy", metric_names)
        self.assertIn("precision", metric_names)
        self.assertIn("recall", metric_names)
        self.assertIn("f1_score", metric_names)
    
    def test_async_generation(self):
        """Test async model card generation."""
        async def run_test():
            eval_data = {
                "model_name": "async-model",
                "accuracy": 0.95,
                "loss": 0.05
            }
            
            card = await self.generator.generate_async(eval_results=eval_data)
            
            self.assertIsInstance(card, ModelCard)
            self.assertEqual(card.model_details.name, "async-model")
            self.assertTrue(len(card.evaluation_results) >= 2)
        
        asyncio.run(run_test())
    
    def test_batch_generation(self):
        """Test batch model card generation."""
        async def run_test():
            tasks = [
                {"eval_results": {"model_name": "model-1", "accuracy": 0.91}},
                {"eval_results": {"model_name": "model-2", "accuracy": 0.89}},
                {"eval_results": {"model_name": "model-3", "accuracy": 0.93}}
            ]
            
            cards = await self.generator.generate_batch_async(tasks, max_concurrent=2)
            
            self.assertEqual(len(cards), 3)
            for i, card in enumerate(cards):
                self.assertEqual(card.model_details.name, f"model-{i+1}")
        
        asyncio.run(run_test())
    
    def test_auto_population(self):
        """Test auto-population of missing sections."""
        # Generate card with minimal data
        card = self.generator.generate()
        
        # Check auto-populated fields
        self.assertTrue(card.model_details.name)  # Should have default name
        self.assertTrue(card.model_details.version)  # Should have default version
        self.assertTrue(card.intended_use)  # Should have default intended use
        self.assertTrue(card.limitations.known_limitations)  # Should have default limitations
    
    def test_error_handling(self):
        """Test comprehensive error handling."""
        # Test with invalid eval results path
        with self.assertRaises(ModelCardError):
            self.generator.generate(eval_results="/nonexistent/file.json")
    
    def test_generation_statistics(self):
        """Test generation statistics tracking."""
        # Generate a few cards
        for i in range(3):
            card = self.generator.generate(eval_results={"model_name": f"test-{i}"})
            self.assertIsInstance(card, ModelCard)
        
        stats = self.generator.get_generation_statistics()
        self.assertEqual(stats["total_generated"], 3)
        self.assertEqual(stats["total_failures"], 0)
        self.assertGreater(stats["success_rate"], 0.0)
        self.assertGreater(stats["avg_generation_time"], 0.0)
    
    def test_metadata_application(self):
        """Test application of additional metadata."""
        metadata = {
            "model_name": "metadata-test",
            "authors": "Test Author",
            "license": "MIT",
            "description": "Test description"
        }
        
        card = self.generator.generate(**metadata)
        
        self.assertEqual(card.model_details.name, "metadata-test")
        self.assertEqual(card.model_details.authors, ["Test Author"])
        self.assertEqual(card.model_details.license, "MIT")
        self.assertEqual(card.model_details.description, "Test description")
    
    def test_performance_report(self):
        """Test performance report generation."""
        # Generate some cards to have data
        for i in range(2):
            self.generator.generate(eval_results={"accuracy": 0.9 + i * 0.01})
        
        report = self.generator.get_performance_report()
        
        self.assertIn("generation_stats", report)
        self.assertIn("performance_stats", report)
        self.assertIn("circuit_breaker_status", report)
        self.assertIn("rate_limiter_stats", report)
        
        # Check generation stats
        gen_stats = report["generation_stats"]
        self.assertEqual(gen_stats["total_generated"], 2)
        self.assertEqual(gen_stats["total_failures"], 0)


class TestModelCardRendering(unittest.TestCase):
    """Test model card rendering capabilities."""
    
    def setUp(self):
        self.generator = EnhancedModelCardGenerator()
    
    def test_markdown_rendering(self):
        """Test Markdown rendering."""
        eval_data = {
            "model_name": "render-test",
            "accuracy": 0.94,
            "version": "1.2.0"
        }
        
        card = self.generator.generate(eval_results=eval_data, 
                                      authors="Test Author",
                                      license="Apache-2.0")
        
        markdown = card.render("markdown")
        
        self.assertIn("# render-test", markdown)
        self.assertIn("**accuracy**: 0.94", markdown)
        self.assertIn("**Version**: 1.2.0", markdown)
        self.assertIn("**Authors**: Test Author", markdown)
        self.assertIn("**License**: Apache-2.0", markdown)
    
    def test_json_rendering(self):
        """Test JSON rendering."""
        eval_data = {
            "model_name": "json-test",
            "f1_score": 0.87
        }
        
        card = self.generator.generate(eval_results=eval_data)
        json_output = card.render("json")
        
        # Parse JSON to verify structure
        data = json.loads(json_output)
        
        self.assertEqual(data["model_details"]["name"], "json-test")
        self.assertIn("evaluation_results", data)
        self.assertIn("metadata", data)
    
    def test_html_rendering(self):
        """Test HTML rendering."""
        card = self.generator.generate(eval_results={"model_name": "html-test"})
        html_output = card.render("html")
        
        self.assertIn("<!DOCTYPE html>", html_output)
        self.assertIn("<title>html-test</title>", html_output)
        self.assertIn("<h1>html-test", html_output)


class TestCardFormats(unittest.TestCase):
    """Test different card formats."""
    
    def test_huggingface_format(self):
        """Test Hugging Face format."""
        config = CardConfig(format=CardFormat.HUGGINGFACE)
        generator = EnhancedModelCardGenerator(config)
        
        card = generator.generate(eval_results={
            "model_name": "hf-test",
            "accuracy": 0.92,
            "language": ["en", "fr"]
        })
        
        self.assertEqual(card.config.format, CardFormat.HUGGINGFACE)
        self.assertEqual(card.model_details.name, "hf-test")
    
    def test_google_format(self):
        """Test Google format."""
        config = CardConfig(format=CardFormat.GOOGLE)
        generator = EnhancedModelCardGenerator(config)
        
        card = generator.generate(eval_results={
            "model_name": "google-test",
            "precision": 0.88
        })
        
        self.assertEqual(card.config.format, CardFormat.GOOGLE)
        self.assertEqual(card.model_details.name, "google-test")
    
    def test_eu_cra_format(self):
        """Test EU CRA format."""
        config = CardConfig(format=CardFormat.EU_CRA)
        generator = EnhancedModelCardGenerator(config)
        
        card = generator.generate(eval_results={
            "model_name": "eu-cra-test",
            "recall": 0.85
        })
        
        self.assertEqual(card.config.format, CardFormat.EU_CRA)
        self.assertEqual(card.model_details.name, "eu-cra-test")


if __name__ == '__main__':
    unittest.main(verbosity=2)