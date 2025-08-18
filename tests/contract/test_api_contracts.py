"""
Contract tests for Model Card Generator API.

These tests ensure that API contracts between components are maintained
and backward compatibility is preserved across versions.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

from pact import Consumer, Provider, Term, Like, EachLike
from pact.verifier import Verifier

# Mock for demonstration - would use actual Pact library in practice
class MockPact:
    """Mock Pact implementation for contract testing."""
    
    def __init__(self, consumer: str, provider: str):
        self.consumer = consumer
        self.provider = provider
        self.interactions = []
    
    def given(self, state: str):
        self.current_interaction = {"state": state}
        return self
    
    def upon_receiving(self, description: str):
        self.current_interaction["description"] = description
        return self
    
    def with_request(self, method: str, path: str, headers: Dict = None, body: Any = None):
        self.current_interaction["request"] = {
            "method": method,
            "path": path,
            "headers": headers or {},
            "body": body
        }
        return self
    
    def will_respond_with(self, status: int, headers: Dict = None, body: Any = None):
        self.current_interaction["response"] = {
            "status": status,
            "headers": headers or {},
            "body": body
        }
        self.interactions.append(self.current_interaction.copy())
        return self


class TestModelCardGeneratorContracts:
    """Contract tests for the main ModelCardGenerator interface."""
    
    @pytest.fixture
    def generator_consumer_pact(self):
        """Pact between CLI consumer and generator provider."""
        return MockPact(
            consumer="ModelCardCLI",
            provider="ModelCardGenerator"
        )
    
    def test_generate_basic_huggingface_card_contract(self, generator_consumer_pact):
        """Test contract for basic Hugging Face card generation."""
        expected_eval_results = {
            "model_name": "test-sentiment-classifier",
            "model_version": "1.0.0",
            "metrics": {
                "accuracy": 0.92,
                "f1_score": 0.91
            }
        }
        
        expected_response = {
            "content": str,  # Should be markdown string
            "format": "huggingface",
            "metadata": dict,
            "generated_at": str
        }
        
        (generator_consumer_pact
         .given("valid evaluation results exist")
         .upon_receiving("a request to generate a Hugging Face model card")
         .with_request(
             method="POST",
             path="/api/v1/generate",
             headers={"Content-Type": "application/json"},
             body={
                 "eval_results": expected_eval_results,
                 "format": "huggingface"
             }
         )
         .will_respond_with(
             status=200,
             headers={"Content-Type": "application/json"},
             body=expected_response
         ))
        
        # Verify contract
        from modelcard_generator.core.generator import ModelCardGenerator
        generator = ModelCardGenerator()
        
        card = generator.generate(
            eval_results=expected_eval_results,
            format="huggingface"
        )
        
        # Verify response structure matches contract
        assert hasattr(card, 'content')
        assert hasattr(card, 'format')
        assert hasattr(card, 'metadata')
        assert card.format == "huggingface"
        assert isinstance(card.content, str)
        assert len(card.content) > 0
    
    def test_validation_contract(self, generator_consumer_pact):
        """Test contract for model card validation."""
        (generator_consumer_pact
         .given("a model card exists")
         .upon_receiving("a request to validate the model card")
         .with_request(
             method="POST",
             path="/api/v1/validate",
             headers={"Content-Type": "application/json"},
             body={
                 "card_content": str,
                 "format": "huggingface"
             }
         )
         .will_respond_with(
             status=200,
             headers={"Content-Type": "application/json"},
             body={
                 "is_valid": bool,
                 "errors": list,
                 "warnings": list,
                 "score": float
             }
         ))
        
        # Test the actual contract
        from modelcard_generator.core.validator import ModelCardValidator
        validator = ModelCardValidator()
        
        sample_card = """---
language: en
tags:
- sentiment-analysis
---

# Test Model

## Model Description
This is a test model.
"""
        
        result = validator.validate(sample_card, format="huggingface")
        
        # Verify response structure
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)


class TestMLPlatformIntegrationContracts:
    """Contract tests for ML platform integrations."""
    
    @pytest.fixture
    def mlflow_integration_pact(self):
        """Pact between ModelCardGenerator and MLflow."""
        return MockPact(
            consumer="ModelCardGenerator",
            provider="MLflowAPI"
        )
    
    def test_mlflow_experiment_retrieval_contract(self, mlflow_integration_pact):
        """Test contract for retrieving MLflow experiment data."""
        experiment_id = "1"
        
        (mlflow_integration_pact
         .given(f"experiment {experiment_id} exists")
         .upon_receiving("a request for experiment details")
         .with_request(
             method="GET",
             path=f"/api/2.0/mlflow/experiments/get?experiment_id={experiment_id}",
             headers={"Accept": "application/json"}
         )
         .will_respond_with(
             status=200,
             headers={"Content-Type": "application/json"},
             body={
                 "experiment": {
                     "experiment_id": experiment_id,
                     "name": str,
                     "artifact_location": str,
                     "lifecycle_stage": str,
                     "creation_time": int,
                     "last_update_time": int
                 }
             }
         ))
        
        # Test with mock MLflow integration
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "experiment": {
                    "experiment_id": "1",
                    "name": "test-experiment",
                    "artifact_location": "/tmp/artifacts",
                    "lifecycle_stage": "active",
                    "creation_time": 1635724800000,
                    "last_update_time": 1635724800000
                }
            }
            mock_get.return_value = mock_response
            
            from modelcard_generator.integrations.mlflow_integration import MLflowIntegration
            integration = MLflowIntegration(tracking_uri="http://localhost:5000")
            
            experiment = integration.get_experiment("1")
            
            # Verify contract compliance
            assert "experiment_id" in experiment
            assert "name" in experiment
            assert "artifact_location" in experiment
            assert experiment["experiment_id"] == "1"
    
    def test_wandb_runs_contract(self):
        """Test contract for Weights & Biases run data."""
        with patch('wandb.Api') as mock_wandb_api:
            mock_run = MagicMock()
            mock_run.id = "test-run-123"
            mock_run.name = "test-run"
            mock_run.state = "finished"
            mock_run.config = {"learning_rate": 0.001, "batch_size": 32}
            mock_run.summary = {"accuracy": 0.92, "loss": 0.08}
            mock_run.tags = ["experiment", "baseline"]
            
            mock_api = MagicMock()
            mock_api.run.return_value = mock_run
            mock_wandb_api.return_value = mock_api
            
            from modelcard_generator.integrations.wandb_integration import WandbIntegration
            integration = WandbIntegration()
            
            run_data = integration.get_run("test-project/test-run-123")
            
            # Verify expected structure
            required_fields = ["id", "name", "state", "config", "summary"]
            for field in required_fields:
                assert field in run_data, f"Missing required field: {field}"
            
            assert isinstance(run_data["config"], dict)
            assert isinstance(run_data["summary"], dict)


class TestSchemaVersioningContracts:
    """Contract tests for schema versioning and backward compatibility."""
    
    def test_schema_v1_compatibility(self):
        """Test that v1 schema is still supported."""
        v1_eval_results = {
            "model_name": "test-model",
            "accuracy": 0.92,  # Old flat structure
            "precision": 0.89,
            "recall": 0.94
        }
        
        from modelcard_generator.core.generator import ModelCardGenerator
        generator = ModelCardGenerator()
        
        # Should handle old schema without errors
        card = generator.generate(eval_results=v1_eval_results, format="huggingface")
        
        assert card is not None
        assert len(card.content) > 0
        assert "0.92" in card.content or "92%" in card.content
    
    def test_schema_v2_features(self):
        """Test that v2 schema features work correctly."""
        v2_eval_results = {
            "schema_version": "2.0",
            "model": {
                "name": "test-model",
                "version": "1.0.0",
                "architecture": "transformer"
            },
            "evaluation": {
                "metrics": {
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.94
                },
                "dataset": {
                    "name": "test-dataset",
                    "version": "1.0",
                    "size": 10000
                }
            },
            "training": {
                "framework": "pytorch",
                "framework_version": "1.9.0",
                "python_version": "3.8.10"
            }
        }
        
        from modelcard_generator.core.generator import ModelCardGenerator
        generator = ModelCardGenerator()
        
        card = generator.generate(eval_results=v2_eval_results, format="huggingface")
        
        assert card is not None
        assert "transformer" in card.content.lower()
        assert "pytorch" in card.content.lower()
    
    def test_forward_compatibility(self):
        """Test handling of future schema versions gracefully."""
        future_eval_results = {
            "schema_version": "3.0",  # Future version
            "model_name": "test-model",
            "metrics": {"accuracy": 0.92},
            "new_future_field": {"some": "data"},  # Unknown field
            "another_new_section": {
                "quantum_metrics": {"entanglement": 0.85}  # Hypothetical future feature
            }
        }
        
        from modelcard_generator.core.generator import ModelCardGenerator
        generator = ModelCardGenerator()
        
        # Should handle gracefully, not crash
        card = generator.generate(eval_results=future_eval_results, format="huggingface")
        
        assert card is not None
        assert len(card.content) > 0
        # Should at least include known fields
        assert "0.92" in card.content or "92%" in card.content


class TestAPIBreakingChanges:
    """Tests to detect potentially breaking API changes."""
    
    def test_generator_public_interface_stability(self):
        """Ensure public interface of ModelCardGenerator remains stable."""
        from modelcard_generator.core.generator import ModelCardGenerator
        
        generator = ModelCardGenerator()
        
        # These methods should always exist for backward compatibility
        required_methods = [
            'generate',
            'validate',
            'get_supported_formats'
        ]
        
        for method_name in required_methods:
            assert hasattr(generator, method_name), f"Missing required method: {method_name}"
            method = getattr(generator, method_name)
            assert callable(method), f"Method {method_name} is not callable"
    
    def test_configuration_backward_compatibility(self):
        """Test that old configuration formats still work."""
        old_config = {
            "output_format": "huggingface",  # Old key name
            "template_dir": "./templates",   # Old key name
            "validation_level": "strict"    # Old key name
        }
        
        from modelcard_generator.core.config import Config
        
        # Should handle old configuration without errors
        config = Config(old_config)
        
        # Should map old keys to new ones
        assert hasattr(config, 'format') or hasattr(config, 'output_format')
        assert hasattr(config, 'template_directory') or hasattr(config, 'template_dir')
    
    def test_cli_interface_stability(self):
        """Test that CLI interface remains backward compatible."""
        from modelcard_generator.cli.main import create_cli
        
        cli = create_cli()
        
        # Essential commands should exist
        essential_commands = ['generate', 'validate', 'version']
        
        for cmd_name in essential_commands:
            # Check if command exists in CLI
            assert any(cmd.name == cmd_name for cmd in cli.commands.values()), \
                f"Missing essential command: {cmd_name}"


class TestDataContractValidation:
    """Tests for data contract validation between components."""
    
    def test_template_data_contract(self):
        """Test that template data follows expected contract."""
        from modelcard_generator.templates.library import TemplateLibrary
        
        template = TemplateLibrary.get("huggingface")
        
        # Template should define required data fields
        required_fields = template.get_required_fields()
        
        # Should include essential fields
        essential_fields = ["model_name", "metrics", "description"]
        for field in essential_fields:
            assert field in required_fields or any(
                field in req_field for req_field in required_fields
            ), f"Template missing essential field: {field}"
    
    def test_format_output_contract(self):
        """Test that all formats produce consistent output structure."""
        eval_results = {
            "model_name": "test-model",
            "model_version": "1.0.0",
            "metrics": {"accuracy": 0.92}
        }
        
        from modelcard_generator.core.generator import ModelCardGenerator
        generator = ModelCardGenerator()
        
        formats = ["huggingface", "google"]  # Add more as they're implemented
        
        for format_name in formats:
            try:
                card = generator.generate(eval_results=eval_results, format=format_name)
                
                # All formats should have consistent interface
                assert hasattr(card, 'content'), f"Format {format_name} missing content attribute"
                assert hasattr(card, 'format'), f"Format {format_name} missing format attribute"
                assert hasattr(card, 'metadata'), f"Format {format_name} missing metadata attribute"
                
                assert card.format == format_name
                assert isinstance(card.content, str)
                assert len(card.content.strip()) > 0
                
            except NotImplementedError:
                pytest.skip(f"Format {format_name} not implemented yet")


# Helper functions for contract testing
def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate that data matches expected JSON schema."""
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return True
    except ImportError:
        # Fallback validation if jsonschema not available
        return simple_schema_validation(data, schema)
    except jsonschema.ValidationError:
        return False


def simple_schema_validation(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Simple schema validation without external dependencies."""
    if "required" in schema:
        for required_field in schema["required"]:
            if required_field not in data:
                return False
    
    if "properties" in schema:
        for field_name, field_schema in schema["properties"].items():
            if field_name in data:
                field_value = data[field_name]
                expected_type = field_schema.get("type")
                
                if expected_type == "string" and not isinstance(field_value, str):
                    return False
                elif expected_type == "number" and not isinstance(field_value, (int, float)):
                    return False
                elif expected_type == "array" and not isinstance(field_value, list):
                    return False
                elif expected_type == "object" and not isinstance(field_value, dict):
                    return False
    
    return True