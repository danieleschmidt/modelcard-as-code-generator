"""MLflow integration for model card generation."""

import logging
from datetime import datetime
from typing import List, Optional

from ..core.models import CardConfig, ModelCard

logger = logging.getLogger(__name__)


class MLflowIntegration:
    """Integration with MLflow for automatic model card generation."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize MLflow integration."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            self.mlflow = mlflow
            self.MlflowClient = MlflowClient
        except ImportError:
            raise ImportError(
                "mlflow package is required for MLflow integration. "
                "Install with: pip install mlflow"
            )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()

    def from_run(self, run_id: str, config: Optional[CardConfig] = None) -> ModelCard:
        """Generate model card from an MLflow run."""
        try:
            run = self.client.get_run(run_id)
            logger.info(f"Loaded MLflow run: {run.info.run_name} ({run_id})")

            card = ModelCard(config or CardConfig())

            # Extract model details
            self._extract_model_details(card, run)

            # Extract training details
            self._extract_training_details(card, run)

            # Extract metrics
            self._extract_metrics(card, run)

            # Add MLflow metadata
            card.metadata.update({
                "mlflow_run_id": run.info.run_id,
                "mlflow_run_name": run.info.run_name,
                "mlflow_experiment_id": run.info.experiment_id,
                "mlflow_status": run.info.status,
                "mlflow_start_time": run.info.start_time,
                "mlflow_end_time": run.info.end_time,
                "mlflow_tracking_uri": self.mlflow.get_tracking_uri()
            })

            logger.info(f"Generated model card from MLflow run {run_id}")
            return card

        except Exception as e:
            logger.error(f"Failed to generate model card from MLflow run {run_id}: {e}")
            raise

    def from_model(self, model_name: str, version: Optional[int] = None, config: Optional[CardConfig] = None) -> ModelCard:
        """Generate model card from an MLflow registered model."""
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                latest_versions = self.client.get_latest_versions(model_name)
                if not latest_versions:
                    raise ValueError(f"No versions found for model {model_name}")
                model_version = latest_versions[0]

            # Get the run that created this model version
            run = self.client.get_run(model_version.run_id)

            # Generate card from run
            card = self.from_run(model_version.run_id, config)

            # Add model registry metadata
            card.model_details.name = model_name
            card.model_details.version = str(model_version.version)
            card.model_details.description = model_version.description or card.model_details.description

            card.metadata.update({
                "mlflow_model_name": model_name,
                "mlflow_model_version": model_version.version,
                "mlflow_model_stage": model_version.current_stage,
                "mlflow_model_source": model_version.source,
                "mlflow_model_status": model_version.status
            })

            logger.info(f"Generated model card from MLflow model {model_name}:{model_version.version}")
            return card

        except Exception as e:
            logger.error(f"Failed to generate model card from MLflow model {model_name}: {e}")
            raise

    def from_experiment(self, experiment_id: str, config: Optional[CardConfig] = None) -> List[ModelCard]:
        """Generate model cards from all runs in an MLflow experiment."""
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                run_view_type=self.mlflow.entities.ViewType.ACTIVE_ONLY
            )

            cards = []
            for run in runs:
                try:
                    card = self.from_run(run.info.run_id, config)
                    cards.append(card)
                except Exception as e:
                    logger.warning(f"Failed to process run {run.info.run_id}: {e}")
                    continue

            logger.info(f"Generated {len(cards)} model cards from experiment {experiment_id}")
            return cards

        except Exception as e:
            logger.error(f"Failed to process MLflow experiment {experiment_id}: {e}")
            raise

    def _extract_model_details(self, card: ModelCard, run) -> None:
        """Extract model details from MLflow run."""
        run_info = run.info
        params = run.data.params

        # Set model name
        card.model_details.name = params.get("model_name", run_info.run_name or run_info.run_id[:8])

        # Set version
        card.model_details.version = params.get("model_version", "1.0.0")

        # Set description
        card.model_details.description = params.get("model_description",
                                                   f"Model from MLflow run {run_info.run_name}")

        # Extract model architecture
        if "model_type" in params:
            card.training_details.model_architecture = params["model_type"]
        elif "architecture" in params:
            card.training_details.model_architecture = params["architecture"]

        # Extract base model
        if "base_model" in params:
            card.model_details.base_model = params["base_model"]

        # Extract license
        if "license" in params:
            card.model_details.license = params["license"]

        # Extract tags from run tags
        if hasattr(run_info, "tags") and run_info.tags:
            card.model_details.tags = list(run_info.tags.keys())

    def _extract_training_details(self, card: ModelCard, run) -> None:
        """Extract training details from MLflow run."""
        params = run.data.params

        # Framework detection
        framework_indicators = {
            "sklearn": "scikit-learn",
            "tensorflow": "TensorFlow",
            "keras": "Keras/TensorFlow",
            "torch": "PyTorch",
            "pytorch": "PyTorch",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM"
        }

        for key, framework in framework_indicators.items():
            if any(key in param_key.lower() for param_key in params.keys()):
                card.training_details.framework = framework
                break

        # Extract hyperparameters
        hyperparams = {}
        hyperparam_keys = [
            "learning_rate", "lr", "batch_size", "epochs", "n_estimators",
            "max_depth", "min_samples_split", "regularization",
            "optimizer", "weight_decay", "dropout", "hidden_size"
        ]

        for key, value in params.items():
            if any(hp_key in key.lower() for hp_key in hyperparam_keys):
                try:
                    # Try to convert to numeric if possible
                    if "." in value:
                        hyperparams[key] = float(value)
                    else:
                        hyperparams[key] = int(value)
                except ValueError:
                    hyperparams[key] = value

        card.training_details.hyperparameters = hyperparams

        # Extract dataset info
        dataset_params = ["dataset", "data_path", "train_data", "training_data"]
        for param in dataset_params:
            if param in params:
                card.training_details.training_data.append(params[param])

        # Training time
        if run.info.end_time and run.info.start_time:
            duration_ms = run.info.end_time - run.info.start_time
            duration_hours = duration_ms / (1000 * 60 * 60)
            card.training_details.training_time = f"{duration_hours:.2f} hours"

    def _extract_metrics(self, card: ModelCard, run) -> None:
        """Extract evaluation metrics from MLflow run."""
        metrics = run.data.metrics

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                card.add_metric(metric_name, float(metric_value))

    def register_model_card(self, card: ModelCard, model_name: str, description: Optional[str] = None) -> str:
        """Register model card as an MLflow artifact."""
        try:
            # Start MLflow run if not already active
            if not self.mlflow.active_run():
                self.mlflow.start_run()

            # Log model card as artifact
            card_content = card.render("markdown")

            import os
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                card_path = os.path.join(temp_dir, "MODEL_CARD.md")

                with open(card_path, "w") as f:
                    f.write(card_content)

                # Log artifact
                self.mlflow.log_artifact(card_path, "model_card")

                # Log model card metadata as parameters
                self.mlflow.log_param("model_card_format", card.config.format.value)
                self.mlflow.log_param("model_card_generated_at", datetime.now().isoformat())

                # Log key metrics as MLflow metrics
                for metric in card.evaluation_results:
                    self.mlflow.log_metric(f"card_{metric.name}", metric.value)

            run_id = self.mlflow.active_run().info.run_id
            logger.info(f"Registered model card for {model_name} in run {run_id}")

            return run_id

        except Exception as e:
            logger.error(f"Failed to register model card for {model_name}: {e}")
            raise
        finally:
            if self.mlflow.active_run():
                self.mlflow.end_run()

    def update_model_description(self, model_name: str, version: int, card: ModelCard) -> None:
        """Update registered model description with model card summary."""
        try:
            # Create summary from model card
            summary = f"# {card.model_details.name}\n\n"
            if card.model_details.description:
                summary += f"{card.model_details.description}\n\n"

            if card.evaluation_results:
                summary += "## Performance Metrics\n"
                for metric in card.evaluation_results[:5]:  # Top 5 metrics
                    summary += f"- {metric.name}: {metric.value:.4f}\n"

            # Update model version description
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=summary
            )

            logger.info(f"Updated description for model {model_name} version {version}")

        except Exception as e:
            logger.error(f"Failed to update model description: {e}")
            raise
