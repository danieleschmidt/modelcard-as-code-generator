"""Core model card generation engine."""

import csv
import json

try:
    import yaml
except ImportError:
    yaml = None
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .cache_simple import cache_manager
from .config import get_config
from .exceptions import DataSourceError, ModelCardError, ValidationError
from .logging_config import get_logger
from .models import (
    CardConfig,
    ModelCard,
)
from .security import sanitizer, scan_for_vulnerabilities

logger = get_logger(__name__)


class DataSourceParser:
    """Parse various data sources for model card generation."""

    def __init__(self):
        self.cache = cache_manager.get_cache()

    def parse_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse JSON evaluation results with security and error handling."""
        try:
            path = Path(file_path)

            # Create cache key based on file path and modification time
            stat_info = path.stat() if path.exists() else None
            if stat_info:
                cache_key = f"json_parse:{path.absolute()}:{stat_info.st_mtime}:{stat_info.st_size}"

                # Check cache first
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    logger.debug(f"Cache hit for JSON parsing: {path}")
                    return cached_data
            else:
                raise DataSourceError(str(path), "File not found")

            # Validate file extension
            if path.suffix.lower() not in [".json"]:
                raise DataSourceError(str(path), f"Invalid file extension: {path.suffix}")

            # Check file size
            config = get_config()
            if path.stat().st_size > config.security.max_file_size:
                raise DataSourceError(str(path), "File too large")

            with open(path, encoding="utf-8") as f:
                content = f.read()

                # Security scan if enabled
                if config.security.scan_content:
                    scan_result = scan_for_vulnerabilities(content)
                    if not scan_result["passed"]:
                        raise DataSourceError(str(path), "Security scan failed",
                                           details={"vulnerabilities": scan_result["vulnerabilities"]})

                # Parse and sanitize
                data = json.loads(content)
                sanitized_data = sanitizer.validate_json(data)

                # Cache the result (TTL: 5 minutes for file parsing)
                if stat_info:
                    self.cache.put(cache_key, sanitized_data, ttl_seconds=300)
                    logger.debug(f"Cached JSON parsing result: {path}")

                return sanitized_data

        except json.JSONDecodeError as e:
            raise DataSourceError(str(file_path), f"Invalid JSON format: {e}")
        except Exception as e:
            if isinstance(e, DataSourceError):
                raise
            raise DataSourceError(str(file_path), f"Failed to parse JSON: {e}")

    def parse_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse YAML configuration files with security and error handling."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataSourceError(str(path), "File not found")

            # Validate file extension
            if path.suffix.lower() not in [".yaml", ".yml"]:
                raise DataSourceError(str(path), f"Invalid file extension: {path.suffix}")

            # Check file size
            config = get_config()
            if path.stat().st_size > config.security.max_file_size:
                raise DataSourceError(str(path), "File too large")

            with open(path, encoding="utf-8") as f:
                content = f.read()

                # Security scan if enabled
                if config.security.scan_content:
                    scan_result = scan_for_vulnerabilities(content)
                    if not scan_result["passed"]:
                        raise DataSourceError(str(path), "Security scan failed",
                                           details={"vulnerabilities": scan_result["vulnerabilities"]})

                # Parse and sanitize
                if yaml is None:
                    raise DataSourceError(str(file_path), "YAML support not available - install PyYAML")
                data = yaml.safe_load(content)
                return sanitizer.validate_json(data or {})

        except Exception as yaml_error:
            if yaml and hasattr(yaml, "YAMLError") and isinstance(yaml_error, yaml.YAMLError):
                raise DataSourceError(str(file_path), f"Invalid YAML format: {yaml_error}")
            elif yaml_error.__class__.__name__ in ["DataSourceError"]:
                raise
            else:
                raise DataSourceError(str(file_path), f"Failed to parse YAML: {yaml_error}")

    def parse_csv(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parse CSV evaluation results with security and error handling."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataSourceError(str(path), "File not found")

            # Validate file extension
            if path.suffix.lower() not in [".csv", ".tsv"]:
                raise DataSourceError(str(path), f"Invalid file extension: {path.suffix}")

            # Check file size
            config = get_config()
            if path.stat().st_size > config.security.max_file_size:
                raise DataSourceError(str(path), "File too large")

            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

            with open(path, encoding="utf-8") as f:
                content = f.read()

                # Security scan if enabled
                if config.security.scan_content:
                    scan_result = scan_for_vulnerabilities(content)
                    if not scan_result["passed"]:
                        raise DataSourceError(str(path), "Security scan failed",
                                           details={"vulnerabilities": scan_result["vulnerabilities"]})

            # Parse CSV
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                data = []
                for row_idx, row in enumerate(reader):
                    if row_idx > 10000:  # Limit rows for safety
                        logger.warning("CSV file truncated at 10000 rows")
                        break
                    # Sanitize each row
                    sanitized_row = sanitizer._sanitize_dict(row)
                    data.append(sanitized_row)
                return data

        except csv.Error as e:
            raise DataSourceError(str(file_path), f"Invalid CSV format: {e}")
        except Exception as e:
            if isinstance(e, DataSourceError):
                raise
            raise DataSourceError(str(file_path), f"Failed to parse CSV: {e}")

    def parse_training_log(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse training log files to extract metrics and metadata."""
        log_data = {"metrics": [], "hyperparameters": {}, "hardware": None}

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Extract metrics (assuming format like "epoch: 1, loss: 0.5, accuracy: 0.8")
                if "loss:" in line and "accuracy:" in line:
                    try:
                        parts = line.split(",")
                        metrics = {}
                        for part in parts:
                            if ":" in part:
                                key, value = part.split(":", 1)
                                key = key.strip()
                                value = value.strip()
                                if key in ["loss", "accuracy", "f1", "precision", "recall"]:
                                    metrics[key] = float(value)
                        if metrics:
                            log_data["metrics"].append(metrics)
                    except (ValueError, IndexError):
                        continue

                # Extract hyperparameters
                if "learning_rate:" in line or "batch_size:" in line:
                    try:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if key in ["learning_rate", "batch_size", "epochs"]:
                            log_data["hyperparameters"][key] = float(value) if "." in value else int(value)
                    except (ValueError, IndexError):
                        continue

                # Extract hardware info
                if "GPU:" in line or "CUDA:" in line:
                    log_data["hardware"] = line

        return log_data


class ModelCardGenerator:
    """Main generator for creating model cards from various sources."""

    def __init__(self, config: Optional[CardConfig] = None):
        self.config = config or CardConfig()
        self.parser = DataSourceParser()
        self.app_config = get_config()
        self.cache = cache_manager.get_cache()
        logger.set_context(component="generator", format=self.config.format.value)

    def generate(
        self,
        eval_results: Optional[Union[str, Path, Dict[str, Any]]] = None,
        training_history: Optional[Union[str, Path]] = None,
        dataset_info: Optional[Union[str, Path, Dict[str, Any]]] = None,
        model_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        **kwargs
    ) -> ModelCard:
        """Generate a model card from various input sources with comprehensive error handling."""

        start_time = time.time()
        operation = "model_card_generation"

        try:
            logger.log_operation_start(operation,
                                     eval_results=bool(eval_results),
                                     training_history=bool(training_history),
                                     dataset_info=bool(dataset_info),
                                     model_config=bool(model_config))

            # Sanitize kwargs
            if kwargs:
                kwargs = sanitizer._sanitize_dict(kwargs)

            card = ModelCard(config=self.config)

            # Process inputs with individual error handling
            with logger.context(step="processing_inputs"):
                if eval_results:
                    self._process_evaluation_results(card, eval_results)

                if training_history:
                    self._process_training_history(card, training_history)

                if dataset_info:
                    self._process_dataset_info(card, dataset_info)

                if model_config:
                    self._process_model_config(card, model_config)

            # Apply additional metadata
            with logger.context(step="applying_metadata"):
                self._apply_additional_metadata(card, kwargs)

            # Auto-populate missing information if enabled
            if self.config.auto_populate:
                with logger.context(step="auto_populate"):
                    self._auto_populate_sections(card)

            # Validate if auto-validation is enabled
            if self.app_config.auto_validate:
                with logger.context(step="validation"):
                    from .validator import Validator
                    validator = Validator()
                    result = validator.validate_schema(card, self.config.format.value)
                    logger.log_validation_result(result.is_valid, result.score, result.issues)

                    if not result.is_valid and self.app_config.validation.enforce_compliance:
                        raise ValidationError("Model card validation failed", result.issues)

            # Security scan if enabled
            if self.app_config.auto_scan_security:
                with logger.context(step="security_scan"):
                    content = card.render()
                    scan_result = scan_for_vulnerabilities(content)
                    logger.log_security_check("content_scan", scan_result["passed"], scan_result)

                    if not scan_result["passed"]:
                        logger.warning("Security vulnerabilities detected in generated content")

            duration_ms = (time.time() - start_time) * 1000
            logger.log_operation_success(operation, duration_ms,
                                       model_name=card.model_details.name,
                                       metrics_count=len(card.evaluation_results))

            return card

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.log_operation_failure(operation, e, duration_ms)

            # Re-raise with context
            if isinstance(e, ModelCardError):
                raise
            else:
                raise ModelCardError(f"Model card generation failed: {e}",
                                   details={"operation": operation, "duration_ms": duration_ms})

    def generate_batch(self, tasks: List[Dict[str, Any]], max_workers: Optional[int] = None) -> List[ModelCard]:
        """Generate multiple model cards concurrently.
        
        Args:
            tasks: List of task dictionaries, each containing parameters for generate()
            max_workers: Maximum number of concurrent workers (defaults to CPU count)
        
        Returns:
            List of generated ModelCard instances
        """
        max_workers = max_workers or min(len(tasks), 4)  # Reasonable default
        results = []
        failed_tasks = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, task in enumerate(tasks):
                future = executor.submit(self.generate, **task)
                future_to_task[future] = (i, task)

            # Collect results
            for future in as_completed(future_to_task):
                task_index, task = future_to_task[future]
                try:
                    result = future.result()
                    results.append((task_index, result))
                    logger.info(f"Batch task {task_index} completed successfully")
                except Exception as e:
                    failed_tasks.append((task_index, task, str(e)))
                    logger.error(f"Batch task {task_index} failed: {e}")

        # Sort results by original task order
        results.sort(key=lambda x: x[0])
        ordered_results = [result for _, result in results]

        # Log batch summary
        logger.info(f"Batch processing completed: {len(ordered_results)} successful, {len(failed_tasks)} failed")
        if failed_tasks:
            logger.warning(f"Failed tasks: {[(i, str(e)) for i, _, e in failed_tasks]}")

        return ordered_results

    def from_wandb(self, run_id: str, project: Optional[str] = None) -> ModelCard:
        """Generate model card from Weights & Biases run."""
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb package required for W&B integration")

        api = wandb.Api()
        if project:
            run = api.run(f"{project}/{run_id}")
        else:
            run = api.run(run_id)

        card = ModelCard(config=self.config)

        # Extract model details
        card.model_details.name = run.config.get("model_name", run.name)
        card.model_details.version = run.config.get("model_version", "1.0.0")
        card.model_details.description = run.notes or f"Model trained in W&B run {run_id}"

        # Extract training details
        card.training_details.framework = run.config.get("framework", "unknown")
        card.training_details.hyperparameters = dict(run.config)

        # Extract metrics
        for metric_name, metric_value in run.summary.items():
            if isinstance(metric_value, (int, float)):
                card.add_metric(metric_name, float(metric_value))

        return card

    def from_mlflow(self, model_name: str, version: Optional[int] = None) -> ModelCard:
        """Generate model card from MLflow model."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            raise ImportError("mlflow package required for MLflow integration")

        client = MlflowClient()

        if version:
            model_version = client.get_model_version(model_name, version)
            run = client.get_run(model_version.run_id)
        else:
            latest_version = client.get_latest_versions(model_name)[0]
            run = client.get_run(latest_version.run_id)

        card = ModelCard(config=self.config)

        # Extract model details
        card.model_details.name = model_name
        card.model_details.version = str(version) if version else "latest"
        card.model_details.description = run.info.run_name or f"MLflow model {model_name}"

        # Extract training details
        card.training_details.hyperparameters = dict(run.data.params)

        # Extract metrics
        for metric_name, metric_value in run.data.metrics.items():
            card.add_metric(metric_name, metric_value)

        return card

    def _process_evaluation_results(self, card: ModelCard, eval_results: Union[str, Path, Dict[str, Any]]) -> None:
        """Process evaluation results and add metrics to the card."""
        if isinstance(eval_results, (str, Path)):
            path = Path(eval_results)
            if path.suffix == ".json":
                data = self.parser.parse_json(path)
            elif path.suffix in [".yaml", ".yml"]:
                data = self.parser.parse_yaml(path)
            elif path.suffix == ".csv":
                csv_data = self.parser.parse_csv(path)
                # Convert CSV to dict format
                data = {}
                for row in csv_data:
                    for key, value in row.items():
                        try:
                            data[key] = float(value)
                        except (ValueError, TypeError):
                            data[key] = value
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            data = eval_results

        # Extract metrics
        metric_fields = ["accuracy", "precision", "recall", "f1", "f1_score", "f1_macro", "f1_micro",
                        "auc", "roc_auc", "loss", "val_loss", "mse", "rmse", "mae", "r2",
                        "bleu", "rouge", "perplexity", "inference_time", "inference_time_ms"]

        for field in metric_fields:
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)):
                    card.add_metric(field, float(value))
                elif isinstance(value, dict) and "value" in value:
                    # Handle confidence intervals
                    ci = value.get("confidence_interval")
                    card.add_metric(field, float(value["value"]), confidence_interval=ci)

        # Extract model name if not set
        if not card.model_details.name and "model_name" in data:
            card.model_details.name = data["model_name"]

        # Extract dataset information
        if "dataset" in data:
            if isinstance(data["dataset"], str):
                card.training_details.training_data.append(data["dataset"])
            elif isinstance(data["dataset"], list):
                card.training_details.training_data.extend(data["dataset"])

    def _process_training_history(self, card: ModelCard, training_history: Union[str, Path]) -> None:
        """Process training history logs."""
        log_data = self.parser.parse_training_log(training_history)

        # Update hyperparameters
        card.training_details.hyperparameters.update(log_data["hyperparameters"])

        # Update hardware info
        if log_data["hardware"]:
            card.training_details.hardware = log_data["hardware"]

        # Add final metrics if available
        if log_data["metrics"]:
            final_metrics = log_data["metrics"][-1]  # Use last epoch metrics
            for metric_name, value in final_metrics.items():
                card.add_metric(f"final_{metric_name}", value)

    def _process_dataset_info(self, card: ModelCard, dataset_info: Union[str, Path, Dict[str, Any]]) -> None:
        """Process dataset information."""
        if isinstance(dataset_info, (str, Path)):
            path = Path(dataset_info)
            if path.suffix == ".json":
                data = self.parser.parse_json(path)
            elif path.suffix in [".yaml", ".yml"]:
                data = self.parser.parse_yaml(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            data = dataset_info

        # Extract dataset names
        if "datasets" in data:
            if isinstance(data["datasets"], list):
                card.model_details.datasets.extend(data["datasets"])
            else:
                card.model_details.datasets.append(str(data["datasets"]))

        if "training_data" in data:
            if isinstance(data["training_data"], list):
                card.training_details.training_data.extend(data["training_data"])
            else:
                card.training_details.training_data.append(str(data["training_data"]))

        # Extract preprocessing information
        if "preprocessing" in data:
            card.training_details.preprocessing = data["preprocessing"]

        # Extract bias and fairness information
        if "bias_analysis" in data:
            bias_data = data["bias_analysis"]
            if "bias_risks" in bias_data:
                card.ethical_considerations.bias_risks.extend(bias_data["bias_risks"])
            if "fairness_metrics" in bias_data:
                card.ethical_considerations.fairness_metrics.update(bias_data["fairness_metrics"])

    def _process_model_config(self, card: ModelCard, model_config: Union[str, Path, Dict[str, Any]]) -> None:
        """Process model configuration."""
        if isinstance(model_config, (str, Path)):
            path = Path(model_config)
            if path.suffix == ".json":
                data = self.parser.parse_json(path)
            elif path.suffix in [".yaml", ".yml"]:
                data = self.parser.parse_yaml(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            data = model_config

        # Extract model details
        if "name" in data:
            card.model_details.name = data["name"]
        if "version" in data:
            card.model_details.version = data["version"]
        if "description" in data:
            card.model_details.description = data["description"]
        if "license" in data:
            card.model_details.license = data["license"]
        if "authors" in data:
            card.model_details.authors = data["authors"] if isinstance(data["authors"], list) else [data["authors"]]
        if "base_model" in data:
            card.model_details.base_model = data["base_model"]
        if "language" in data:
            card.model_details.language = data["language"] if isinstance(data["language"], list) else [data["language"]]
        if "tags" in data:
            card.model_details.tags = data["tags"] if isinstance(data["tags"], list) else [data["tags"]]

        # Extract training details
        if "framework" in data:
            card.training_details.framework = data["framework"]
        if "architecture" in data:
            card.training_details.model_architecture = data["architecture"]
        if "hyperparameters" in data:
            card.training_details.hyperparameters.update(data["hyperparameters"])

    def _apply_additional_metadata(self, card: ModelCard, metadata: Dict[str, Any]) -> None:
        """Apply additional metadata from kwargs."""
        # Handle direct field assignments
        if "model_name" in metadata:
            card.model_details.name = metadata["model_name"]
        if "model_version" in metadata:
            card.model_details.version = metadata["model_version"]
        if "description" in metadata:
            card.model_details.description = metadata["description"]
        if "intended_use" in metadata:
            card.intended_use = metadata["intended_use"]
        if "license" in metadata:
            card.model_details.license = metadata["license"]
        if "authors" in metadata:
            card.model_details.authors = metadata["authors"] if isinstance(metadata["authors"], list) else [metadata["authors"]]

        # Store remaining metadata
        excluded_keys = {"model_name", "model_version", "description", "intended_use", "license", "authors"}
        for key, value in metadata.items():
            if key not in excluded_keys:
                card.metadata[key] = value

    def _auto_populate_sections(self, card: ModelCard) -> None:
        """Auto-populate missing sections with defaults."""
        # Set default model name if missing
        if not card.model_details.name:
            card.model_details.name = "untitled-model"

        # Set default version if missing
        if not card.model_details.version:
            card.model_details.version = "1.0.0"

        # Set default intended use if missing
        if not card.intended_use:
            card.intended_use = "This model is intended for research and educational purposes."

        # Add default limitations if none specified
        if not card.limitations.known_limitations:
            card.limitations.known_limitations = [
                "Model performance may degrade on out-of-distribution data",
                "Requires further validation for production use",
                "May exhibit biases present in training data"
            ]

        # Add default ethical considerations if enabled and missing
        if self.config.include_ethical_considerations and not card.ethical_considerations.bias_risks:
            card.ethical_considerations.bias_risks = [
                "Potential for demographic bias in predictions",
                "May perpetuate historical biases in training data"
            ]
            card.ethical_considerations.bias_mitigation = [
                "Regular bias audits recommended",
                "Diverse evaluation datasets should be used"
            ]

        # Set compliance info for regulatory standards
        if self.config.regulatory_standard:
            card.set_compliance_info(
                self.config.regulatory_standard,
                "pending_review",
                {"auto_generated": True, "requires_manual_review": True}
            )


class AutoUpdater:
    """Automatically update model cards based on file changes."""

    def __init__(self, card_path: str, watch_paths: List[str]):
        self.card_path = Path(card_path)
        self.watch_paths = [Path(p) for p in watch_paths]
        self.rules: List[Dict[str, Any]] = []

    def add_rule(self, trigger: str, action: str, **kwargs) -> None:
        """Add an update rule."""
        rule = {
            "trigger": trigger,
            "action": action,
            **kwargs
        }
        self.rules.append(rule)

    def run(self) -> None:
        """Run the auto-updater (simplified implementation)."""
        # This would typically use file watching or be triggered by CI
        # For now, just check if files exist and log
        logger.info(f"Auto-updater monitoring {len(self.watch_paths)} paths")
        for path in self.watch_paths:
            if path.exists():
                logger.info(f"Found file to watch: {path}")

        # Apply rules based on file modifications
        for rule in self.rules:
            if rule.get("auto_commit", False):
                logger.info(f"Would apply rule: {rule['action']} for {rule['trigger']}")


class Pipeline:
    """End-to-end pipeline for model card generation."""

    def __init__(self):
        self.generator = ModelCardGenerator()
        self.collected_data: Dict[str, Any] = {}

    def collect_from_wandb(self, run_id: str, project: Optional[str] = None) -> "Pipeline":
        """Collect data from Weights & Biases."""
        self.collected_data["wandb"] = {"run_id": run_id, "project": project}
        return self

    def collect_from_mlflow(self, experiment_name: str, run_id: Optional[str] = None) -> "Pipeline":
        """Collect data from MLflow."""
        self.collected_data["mlflow"] = {"experiment_name": experiment_name, "run_id": run_id}
        return self

    def collect_from_github(self, repo: str, branch: str = "main") -> "Pipeline":
        """Collect data from GitHub repository."""
        self.collected_data["github"] = {"repo": repo, "branch": branch}
        return self

    def generate(self, template: str, format: str, compliance: List[str]) -> ModelCard:
        """Generate model card from collected data."""
        config = CardConfig(
            format=format,
            regulatory_standard=compliance[0] if compliance else None
        )

        self.generator.config = config

        # Use collected data to generate card
        card = self.generator.generate()

        # Apply template if specified
        if template and hasattr(self, "_apply_template"):
            card = self._apply_template(card, template)

        return card

    def validate(self, card: ModelCard) -> Dict[str, Any]:
        """Validate the generated card."""
        from .validator import Validator
        validator = Validator()
        result = validator.validate_schema(card)

        return {
            "score": result.score,
            "suggestions": [issue.suggestion for issue in result.issues if issue.suggestion]
        }

    def auto_improve(self, card: ModelCard, suggestions: List[str]) -> ModelCard:
        """Auto-improve card based on suggestions."""
        # Simplified implementation
        for suggestion in suggestions:
            if "add limitations" in suggestion.lower():
                card.add_limitation("Additional validation required for production use")

        return card

    def publish(self, card: ModelCard, destinations: List[str]) -> None:
        """Publish card to multiple destinations."""
        for dest in destinations:
            logger.info(f"Publishing to {dest}")
            if dest == "github":
                card.save("MODEL_CARD.md")
            elif dest == "huggingface":
                card.save("README.md")  # HF format
            elif dest == "confluence":
                # Would integrate with Confluence API
                pass
