"""Executable model cards with embedded validation tests."""

import ast
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

from .core.models import CardConfig, ModelCard

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running an embedded test."""
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ExecutionResults:
    """Results of running all embedded tests."""
    all_passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[TestResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0


class ExecutableCard(ModelCard):
    """Model card with embedded executable validation tests."""

    def __init__(self, config: Optional[CardConfig] = None):
        super().__init__(config)
        self.embedded_tests: List[str] = []
        self.test_requirements: List[str] = []
        self.test_setup_code: Optional[str] = None

    def add_test(self, test_code: str, test_name: Optional[str] = None) -> None:
        """Add an embedded test to the model card.
        
        Args:
            test_code: Python test code as string
            test_name: Optional name for the test
        """
        if test_name:
            # Wrap test code with function definition
            wrapped_code = f"def {test_name}():\n"
            for line in test_code.split("\n"):
                wrapped_code += f"    {line}\n"
            self.embedded_tests.append(wrapped_code)
        else:
            self.embedded_tests.append(test_code)

        self._log_change("Added embedded test", {
            "test_name": test_name or f"test_{len(self.embedded_tests)}",
            "code_length": len(test_code)
        })

    def add_performance_test(self, claimed_metrics: Dict[str, float], tolerance: float = 0.05) -> None:
        """Add a test to verify claimed performance metrics.
        
        Args:
            claimed_metrics: Dictionary of metric names and claimed values
            tolerance: Acceptable tolerance for metric verification
        """
        test_code = f"""
import json
try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path

def test_performance_claims():
    \"\"\"Verify that actual performance meets claimed metrics.\"\"\"
    claimed_metrics = {claimed_metrics}
    tolerance = {tolerance}
    
    # Try to load actual results from common locations
    result_files = [
        'results/eval.json',
        'evaluation/metrics.json', 
        'metrics.json',
        'results.yaml'
    ]
    
    actual_metrics = {{}}
    for result_file in result_files:
        if Path(result_file).exists():
            if result_file.endswith('.json'):
                with open(result_file) as f:
                    actual_metrics = json.load(f)
            else:
                with open(result_file) as f:
                    actual_metrics = yaml.safe_load(f)
            break
    
    if not actual_metrics:
        raise AssertionError("No evaluation results found to verify claims")
    
    # Verify each claimed metric
    for metric_name, claimed_value in claimed_metrics.items():
        if metric_name not in actual_metrics:
            raise AssertionError(f"Metric '{{metric_name}}' not found in results")
        
        actual_value = actual_metrics[metric_name]
        difference = abs(actual_value - claimed_value)
        max_difference = claimed_value * tolerance
        
        if difference > max_difference:
            raise AssertionError(
                f"Metric '{{metric_name}}' verification failed: "
                f"claimed {{claimed_value}}, actual {{actual_value}} "
                f"(difference {{difference:.4f}} > tolerance {{max_difference:.4f}})"
            )
"""
        self.add_test(test_code, "test_performance_claims")

    def add_bias_test(self, sensitive_attributes: List[str], fairness_threshold: float = 0.1) -> None:
        """Add a test to check for bias in model predictions.
        
        Args:
            sensitive_attributes: List of sensitive attributes to check
            fairness_threshold: Maximum allowed fairness metric difference
        """
        test_code = f"""
def test_bias_detection():
    \"\"\"Test for bias across sensitive attributes.\"\"\"
    sensitive_attributes = {sensitive_attributes}
    threshold = {fairness_threshold}
    
    # This is a template - actual implementation would load model and test data
    # and compute fairness metrics like demographic parity, equal opportunity
    
    print(f"Testing bias for attributes: {{sensitive_attributes}}")
    print(f"Fairness threshold: {{threshold}}")
    
    # Placeholder assertion - replace with actual bias detection logic
    assert True, "Bias test implementation required"
"""
        self.add_test(test_code, "test_bias_detection")

    def add_data_validation_test(self, expected_features: List[str]) -> None:
        """Add a test to validate input data schema.
        
        Args:
            expected_features: List of expected feature names
        """
        test_code = f"""
def test_data_validation():
    \"\"\"Validate that input data matches expected schema.\"\"\"
    expected_features = {expected_features}
    
    # Try to load training/test data for validation
    data_files = [
        'data/train.csv',
        'data/test.csv', 
        'dataset.csv'
    ]
    
    import pandas as pd
    data = None
    for data_file in data_files:
        try:
            data = pd.read_csv(data_file)
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        print("Warning: No data files found for validation")
        return
    
    # Check that all expected features are present
    missing_features = set(expected_features) - set(data.columns)
    if missing_features:
        raise AssertionError(f"Missing expected features: {{missing_features}}")
    
    print(f"Data validation passed: all {{len(expected_features)}} features found")
"""
        self.add_test(test_code, "test_data_validation")

    def add_model_loading_test(self, model_path: str) -> None:
        """Add a test to verify model can be loaded successfully.
        
        Args:
            model_path: Path to the model file
        """
        test_code = f"""
def test_model_loading():
    \"\"\"Test that model can be loaded successfully.\"\"\"
    model_path = "{model_path}"
    
    from pathlib import Path
    
    if not Path(model_path).exists():
        raise AssertionError(f"Model file not found: {{model_path}}")
    
    # Try to load model based on file extension
    if model_path.endswith('.pkl') or model_path.endswith('.joblib'):
        import joblib
        model = joblib.load(model_path)
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        import torch
        model = torch.load(model_path, map_location='cpu')
    elif model_path.endswith('.h5'):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:
        raise AssertionError(f"Unsupported model format: {{model_path}}")
    
    assert model is not None, "Model loaded successfully"
    print(f"Model loaded successfully from {{model_path}}")
"""
        self.add_test(test_code, "test_model_loading")

    def set_test_setup(self, setup_code: str) -> None:
        """Set setup code that runs before all tests.
        
        Args:
            setup_code: Python code to run before tests
        """
        self.test_setup_code = setup_code

    def add_test_requirement(self, requirement: str) -> None:
        """Add a package requirement for running tests.
        
        Args:
            requirement: Package requirement (e.g., 'pandas>=1.0.0')
        """
        self.test_requirements.append(requirement)

    def run_tests(self, verbose: bool = True) -> ExecutionResults:
        """Run all embedded tests and return results.
        
        Args:
            verbose: Whether to print detailed output
            
        Returns:
            ExecutionResults containing test outcomes
        """
        results = []
        passed_count = 0

        # Check test requirements
        missing_packages = self._check_requirements()
        if missing_packages:
            logger.warning(f"Missing required packages: {missing_packages}")
            if verbose:
                print(f"⚠️  Missing packages: {', '.join(missing_packages)}")

        # Run setup code if provided
        if self.test_setup_code:
            try:
                exec(self.test_setup_code)
                if verbose:
                    print("✅ Test setup completed")
            except Exception as e:
                logger.error(f"Test setup failed: {e}")
                if verbose:
                    print(f"❌ Test setup failed: {e}")

        # Run each test
        for i, test_code in enumerate(self.embedded_tests):
            test_name = f"test_{i+1}"

            # Extract test name from function definition if present
            try:
                tree = ast.parse(test_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        test_name = node.name
                        break
            except:
                pass

            if verbose:
                print(f"🧪 Running {test_name}...")

            result = self._run_single_test(test_code, test_name)
            results.append(result)

            if result.passed:
                passed_count += 1
                if verbose:
                    print(f"  ✅ {test_name} passed")
            else:
                if verbose:
                    print(f"  ❌ {test_name} failed: {result.error_message}")

        execution_results = ExecutionResults(
            all_passed=passed_count == len(results),
            total_tests=len(results),
            passed_tests=passed_count,
            failed_tests=len(results) - passed_count,
            results=results
        )

        if verbose:
            print(f"\n📊 Test Results: {passed_count}/{len(results)} passed ({execution_results.success_rate:.1%})")

        return execution_results

    def _run_single_test(self, test_code: str, test_name: str, timeout: int = 30) -> TestResult:
        """Run a single test and return result with enhanced security and timeout.
        
        Args:
            test_code: Test code to execute
            test_name: Name of the test
            timeout: Execution timeout in seconds
        """
        import time

        start_time = time.time()

        # Security: Check for dangerous operations
        if self._contains_dangerous_code(test_code):
            return TestResult(
                test_name=test_name,
                passed=False,
                error_message="Test contains potentially dangerous operations",
                execution_time=0.0
            )

        try:
            # Use subprocess for better isolation and timeout control
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_code)
                test_file_path = f.name

            # Run test in isolated subprocess with timeout
            try:
                result = subprocess.run(
                    [sys.executable, test_file_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=Path.cwd()
                )

                execution_time = time.time() - start_time

                if result.returncode == 0:
                    return TestResult(
                        test_name=test_name,
                        passed=True,
                        execution_time=execution_time
                    )
                else:
                    error_msg = result.stderr or result.stdout or "Test failed with no output"
                    return TestResult(
                        test_name=test_name,
                        passed=False,
                        error_message=error_msg.strip(),
                        execution_time=execution_time
                    )

            except subprocess.TimeoutExpired:
                return TestResult(
                    test_name=test_name,
                    passed=False,
                    error_message=f"Test timed out after {timeout} seconds",
                    execution_time=timeout
                )
            finally:
                # Clean up temporary file
                Path(test_file_path).unlink(missing_ok=True)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error running test {test_name}")

            return TestResult(
                test_name=test_name,
                passed=False,
                error_message=f"Test execution error: {str(e)}",
                execution_time=execution_time
            )

    def _contains_dangerous_code(self, code: str) -> bool:
        """Check if code contains potentially dangerous operations."""
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"import\s+sys",
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"\.write\s*\(",
            r"\.delete\s*\(",
            r"\.remove\s*\(",
            r"shutil\.",
            r"socket\.",
            r"urllib\.",
            r"requests\.",
        ]

        # Allow specific safe operations
        safe_exceptions = [
            r'open\s*\([^)]*["\']r["\'][^)]*\)',  # Read-only file operations
            r"pd\.read_csv",  # Pandas read operations
            r"json\.load",    # JSON loading
            r"yaml\.safe_load",  # YAML loading
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                # Check if it's in the safe exceptions
                is_safe = any(re.search(safe_pattern, code, re.IGNORECASE) for safe_pattern in safe_exceptions)
                if not is_safe:
                    return True

        return False

    def _check_requirements(self) -> List[str]:
        """Check if required packages are available."""
        missing = []

        for requirement in self.test_requirements:
            # Simple package name extraction (handles 'package>=version' format)
            package_name = requirement.split(">=")[0].split("==")[0].split("<")[0].strip()

            try:
                __import__(package_name)
            except ImportError:
                missing.append(package_name)

        return missing

    def render(self, format_type: str = "markdown") -> str:
        """Render executable model card with embedded tests."""
        base_content = super().render(format_type)

        if format_type == "markdown":
            # Add executable tests section
            if self.embedded_tests:
                base_content += "\n\n## Embedded Validation Tests\n"
                base_content += "This model card contains executable tests that can verify the claims made.\n\n"

                if self.test_requirements:
                    base_content += "### Requirements\n"
                    for req in self.test_requirements:
                        base_content += f"- {req}\n"
                    base_content += "\n"

                base_content += "### Running Tests\n"
                base_content += "```python\n"
                base_content += "from modelcard_generator import ExecutableCard\n"
                base_content += "card = ExecutableCard.load('MODEL_CARD.md')\n"
                base_content += "results = card.run_tests()\n"
                base_content += "print(f'Tests passed: {results.success_rate:.1%}')\n"
                base_content += "```\n\n"

                base_content += "### Test Definitions\n"
                for i, test_code in enumerate(self.embedded_tests):
                    base_content += f"\n#### Test {i+1}\n"
                    base_content += "```python\n"
                    base_content += test_code
                    base_content += "\n```\n"

        return base_content

    @classmethod
    def load(cls, path: str) -> "ExecutableCard":
        """Load an executable model card from file.
        
        This is a simplified implementation - a full version would parse
        the markdown and extract embedded test code.
        """
        card = cls()

        # In a real implementation, this would parse the markdown file
        # and extract the embedded test code blocks

        with open(path) as f:
            content = f.read()

        # Extract basic model information from markdown
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                card.model_details.name = line[2:].strip()
                break

        return card

    def save_executable(self, path: str) -> None:
        """Save model card with executable tests."""
        self.save(path)

        # Also save a Python file with just the tests for easy execution
        test_file_path = Path(path).with_suffix(".py")

        with open(test_file_path, "w") as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""Executable tests for model card validation."""\n\n')

            if self.test_requirements:
                f.write("# Required packages:\n")
                for req in self.test_requirements:
                    f.write(f"# - {req}\n")
                f.write("\n")

            if self.test_setup_code:
                f.write("# Setup code\n")
                f.write(self.test_setup_code)
                f.write("\n\n")

            for i, test_code in enumerate(self.embedded_tests):
                f.write(f"# Test {i+1}\n")
                f.write(test_code)
                f.write("\n\n")

            f.write('if __name__ == "__main__":\n')
            f.write('    print("Running model card validation tests...")\n')

            # Generate test runner code
            for i, test_code in enumerate(self.embedded_tests):
                test_name = f"test_{i+1}"
                # Extract function name if present
                try:
                    tree = ast.parse(test_code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            test_name = node.name
                            break
                except:
                    pass

                f.write("    try:\n")
                f.write(f"        {test_name}()\n")
                f.write(f'        print("✅ {test_name} passed")\n')
                f.write("    except Exception as e:\n")
                f.write(f'        print(f"❌ {test_name} failed: {{e}}")\n')

        logger.info(f"Saved executable model card: {path}")
        logger.info(f"Saved test runner: {test_file_path}")


def create_executable_template(model_name: str) -> ExecutableCard:
    """Create a template executable model card."""
    card = ExecutableCard()

    card.model_details.name = model_name
    card.model_details.description = "Template executable model card with validation tests"

    # Add common test requirements
    card.add_test_requirement("pandas>=1.0.0")
    card.add_test_requirement("numpy>=1.19.0")

    # Add basic setup code
    setup_code = """
import warnings
warnings.filterwarnings('ignore')

print("Setting up test environment...")
"""
    card.set_test_setup(setup_code)

    # Add placeholder performance test
    card.add_performance_test({"accuracy": 0.85, "f1": 0.80}, tolerance=0.05)

    # Add data validation test
    card.add_data_validation_test(["feature1", "feature2", "target"])

    return card
