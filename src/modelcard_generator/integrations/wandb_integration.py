"""Weights & Biases integration for model card generation."""

import asyncio
import concurrent.futures
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

from ..core.cache import CacheManager
from ..core.models import CardConfig, ModelCard

logger = logging.getLogger(__name__)


def rate_limit(max_calls: int = 60, time_window: int = 60):
    """Rate limiting decorator for API calls."""
    def decorator(func):
        calls = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls outside the time window
            calls[:] = [call_time for call_time in calls if now - call_time < time_window]

            if len(calls) >= max_calls:
                sleep_time = time_window - (now - calls[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    calls[:] = []

            calls.append(now)
            return func(*args, **kwargs)

        return wrapper
    return decorator


class WandbIntegration:
    """Enhanced W&B integration with performance optimizations and scaling features."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_ttl: int = 3600,
                 max_workers: int = 4,
                 enable_compression: bool = True):
        """Initialize W&B integration with performance features.
        
        Args:
            api_key: W&B API key
            cache_ttl: Cache time-to-live in seconds
            max_workers: Max concurrent workers for parallel processing
            enable_compression: Enable response compression
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb package is required for W&B integration. "
                "Install with: pip install wandb"
            )

        self.api_key = api_key
        if api_key:
            wandb.login(key=api_key)

        # Initialize API with optimizations
        self.api = wandb.Api(timeout=30)

        # Performance optimizations
        self.cache = CacheManager(ttl=cache_ttl, max_size=1000)
        self.max_workers = max_workers
        self.enable_compression = enable_compression

        # Request statistics
        self.stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0
        }

    @rate_limit(max_calls=50, time_window=60)
    def from_run(self, run_path: str, config: Optional[CardConfig] = None) -> ModelCard:
        """Generate model card from a W&B run.
        
        Args:
            run_path: W&B run path in format "entity/project/run_id" or "project/run_id"
            config: Card configuration
            
        Returns:
            Generated model card
        """
        start_time = time.time()
        cache_key = f"wandb_run:{run_path}:{hash(str(config))}"

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for run {run_path}")
            return cached_result

        self.stats["cache_misses"] += 1
        self.stats["api_calls"] += 1

        try:
            # Load run with retry logic
            run = self._load_run_with_retry(run_path)
            logger.info(f"Loaded W&B run: {run.name} ({run.id})")

            # Create model card
            card = ModelCard(config or CardConfig())

            # Extract information concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._extract_model_details, card, run): "model_details",
                    executor.submit(self._extract_training_details, card, run): "training_details",
                    executor.submit(self._extract_metrics, card, run): "metrics",
                    executor.submit(self._extract_artifacts, card, run): "artifacts"
                }

                for future in concurrent.futures.as_completed(futures):
                    task_name = futures[future]
                    try:
                        future.result()
                        logger.debug(f"Completed {task_name} extraction")
                    except Exception as e:
                        logger.warning(f"Failed to extract {task_name}: {e}")

            # Add W&B specific metadata
            card.metadata.update({
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "wandb_project": run.project,
                "wandb_entity": run.entity,
                "wandb_url": run.url,
                "wandb_state": run.state,
                "wandb_created_at": run.created_at.isoformat() if run.created_at else None,
                "wandb_finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "extraction_time": time.time() - start_time
            })

            # Cache the result
            self.cache.set(cache_key, card)

            execution_time = time.time() - start_time
            self.stats["total_time"] += execution_time

            logger.info(f"Generated model card from W&B run {run.name} in {execution_time:.2f}s")
            return card

        except Exception as e:
            logger.error(f"Failed to generate model card from W&B run {run_path}: {e}")
            raise

    def _load_run_with_retry(self, run_path: str, max_retries: int = 3) -> Any:
        """Load W&B run with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return self.api.run(run_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                wait_time = (2 ** attempt) + (0.1 * attempt)
                logger.warning(f"Failed to load run {run_path}, attempt {attempt + 1}/{max_retries}. "
                             f"Retrying in {wait_time:.1f}s: {e}")
                time.sleep(wait_time)

    async def from_project_async(self,
                                project_path: str,
                                filters: Optional[Dict[str, Any]] = None,
                                max_runs: Optional[int] = None) -> List[ModelCard]:
        """Generate model cards from W&B project runs asynchronously.
        
        Args:
            project_path: W&B project path
            filters: Optional filters for runs
            max_runs: Maximum number of runs to process
            
        Returns:
            List of generated model cards
        """
        start_time = time.time()

        try:
            runs = list(self.api.runs(project_path, filters=filters))
            if max_runs:
                runs = runs[:max_runs]

            logger.info(f"Processing {len(runs)} runs from project {project_path}")

            # Process runs in batches to avoid overwhelming the API
            batch_size = min(self.max_workers, 10)
            cards = []

            for i in range(0, len(runs), batch_size):
                batch_runs = runs[i:i + batch_size]

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(self.from_run, f"{project_path}/{run.id}"): run
                        for run in batch_runs
                    }

                    for future in concurrent.futures.as_completed(futures):
                        run = futures[future]
                        try:
                            card = future.result(timeout=60)
                            cards.append(card)
                            logger.debug(f"Processed run {run.id}")
                        except Exception as e:
                            logger.warning(f"Failed to process run {run.id}: {e}")
                            continue

                # Add small delay between batches to be respectful to API
                if i + batch_size < len(runs):
                    await asyncio.sleep(0.5)

            execution_time = time.time() - start_time
            logger.info(f"Generated {len(cards)} model cards from project {project_path} "
                       f"in {execution_time:.2f}s")
            return cards

        except Exception as e:
            logger.error(f"Failed to process W&B project {project_path}: {e}")
            raise

    def from_project(self, project_path: str, filters: Optional[Dict[str, Any]] = None,
                    max_runs: Optional[int] = None) -> List[ModelCard]:
        """Generate model cards from all runs in a W&B project.
        
        Args:
            project_path: W&B project path in format "entity/project" or "project"
            filters: Optional filters for runs
            
        Returns:
            List of generated model cards
        """
        """Generate model cards from W&B project (synchronous version).
        
        Args:
            project_path: W&B project path in format "entity/project" or "project"
            filters: Optional filters for runs
            max_runs: Maximum number of runs to process
            
        Returns:
            List of generated model cards
        """
        # Use asyncio to run the async version
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.from_project_async(project_path, filters, max_runs)
            )
        finally:
            loop.close()

    def from_model(self, model_path: str, version: Optional[str] = None) -> ModelCard:
        """Generate model card from a W&B model.
        
        Args:
            model_path: W&B model path in format "entity/project/model_name"
            version: Model version (defaults to latest)
            
        Returns:
            Generated model card
        """
        try:
            # Get model artifact
            if version:
                artifact_path = f"{model_path}:{version}"
            else:
                artifact_path = f"{model_path}:latest"

            artifact = self.api.artifact(artifact_path)

            # Get linked runs
            runs = artifact.logged_by()
            if not runs:
                raise ValueError(f"No runs found for model {model_path}")

            # Use the most recent run
            latest_run = max(runs, key=lambda r: r.created_at)

            # Generate card from run
            card = self.from_run(f"{latest_run.entity}/{latest_run.project}/{latest_run.id}")

            # Add model-specific metadata
            card.metadata.update({
                "wandb_model_name": artifact.name,
                "wandb_model_version": artifact.version,
                "wandb_model_size": artifact.size,
                "wandb_model_digest": artifact.digest,
                "wandb_model_created_at": artifact.created_at
            })

            logger.info(f"Generated model card from W&B model {model_path}:{artifact.version}")
            return card

        except Exception as e:
            logger.error(f"Failed to generate model card from W&B model {model_path}: {e}")
            raise

    def _extract_model_details(self, card: ModelCard, run) -> None:
        """Extract model details from W&B run."""
        # Set model name
        card.model_details.name = run.config.get("model_name", run.name)

        # Set version
        card.model_details.version = run.config.get("model_version", run.id[:8])

        # Set description
        if run.notes:
            card.model_details.description = run.notes
        else:
            card.model_details.description = f"Model trained in W&B run {run.name}"

        # Set authors (from run user)
        if hasattr(run, "user") and run.user:
            card.model_details.authors = [run.user.name or run.user.username]

        # Extract additional model info from config
        if "model_type" in run.config:
            card.training_details.model_architecture = run.config["model_type"]

        if "base_model" in run.config:
            card.model_details.base_model = run.config["base_model"]

        # Extract tags
        if run.tags:
            card.model_details.tags = run.tags

    def _extract_training_details(self, card: ModelCard, run) -> None:
        """Extract training details from W&B run."""
        config = run.config

        # Framework detection
        if "framework" in config:
            card.training_details.framework = config["framework"]
        elif any(key.startswith(("torch", "pytorch")) for key in config.keys()):
            card.training_details.framework = "PyTorch"
        elif any(key.startswith(("tf", "tensorflow")) for key in config.keys()):
            card.training_details.framework = "TensorFlow"
        elif any(key.startswith(("sklearn", "scikit")) for key in config.keys()):
            card.training_details.framework = "scikit-learn"

        # Extract hyperparameters
        hyperparams = {}
        common_hyperparam_keys = [
            "learning_rate", "lr", "batch_size", "epochs", "num_epochs",
            "optimizer", "weight_decay", "dropout", "hidden_size",
            "num_layers", "max_length", "warmup_steps"
        ]

        for key, value in config.items():
            if key in common_hyperparam_keys:
                hyperparams[key] = value
            elif key.startswith(("train_", "training_")):
                hyperparams[key] = value

        card.training_details.hyperparameters = hyperparams

        # Extract dataset information
        dataset_keys = ["dataset", "dataset_name", "data_path", "train_file"]
        for key in dataset_keys:
            if key in config:
                dataset_value = config[key]
                if isinstance(dataset_value, str):
                    card.training_details.training_data.append(dataset_value)
                elif isinstance(dataset_value, list):
                    card.training_details.training_data.extend(dataset_value)

        # Extract hardware info from system metrics
        if hasattr(run, "system_metrics") and run.system_metrics:
            gpu_info = []
            for key, value in run.system_metrics.items():
                if "gpu" in key.lower():
                    gpu_info.append(f"{key}: {value}")

            if gpu_info:
                card.training_details.hardware = "; ".join(gpu_info)

        # Training time from run duration
        if run.finished_at and run.created_at:
            duration = run.finished_at - run.created_at
            card.training_details.training_time = str(duration)

    def _extract_metrics(self, card: ModelCard, run) -> None:
        """Extract evaluation metrics from W&B run."""
        summary = run.summary

        # Common metric names to extract
        metric_patterns = [
            "accuracy", "acc", "precision", "recall", "f1", "f1_score",
            "auc", "roc_auc", "loss", "val_loss", "test_loss",
            "bleu", "rouge", "meteor", "perplexity",
            "mse", "rmse", "mae", "r2", "r2_score"
        ]

        for key, value in summary.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue

            # Check if it's a metric we want to include
            key_lower = key.lower()
            is_metric = any(pattern in key_lower for pattern in metric_patterns)

            # Include validation/test metrics and final training metrics
            is_validation = any(prefix in key_lower for prefix in ["val_", "test_", "eval_", "final_"])

            if is_metric or is_validation:
                card.add_metric(key, float(value))

        # Extract best metrics if available
        for key, value in summary.items():
            if key.startswith("best_") and isinstance(value, (int, float)):
                metric_name = key.replace("best_", "")
                card.add_metric(f"best_{metric_name}", float(value))

    def _extract_artifacts(self, card: ModelCard, run) -> None:
        """Extract artifact information from W&B run."""
        try:
            artifacts = run.logged_artifacts()

            model_artifacts = []
            dataset_artifacts = []

            for artifact in artifacts:
                if artifact.type == "model":
                    model_artifacts.append({
                        "name": artifact.name,
                        "version": artifact.version,
                        "size": artifact.size,
                        "digest": artifact.digest
                    })
                elif artifact.type == "dataset":
                    dataset_artifacts.append(artifact.name)

            # Update card with artifact info
            if model_artifacts:
                card.metadata["wandb_model_artifacts"] = model_artifacts

            if dataset_artifacts:
                card.training_details.training_data.extend(dataset_artifacts)
                card.metadata["wandb_dataset_artifacts"] = dataset_artifacts

        except Exception as e:
            logger.warning(f"Failed to extract artifacts from run {run.id}: {e}")

    def upload_model_card(self, card: ModelCard, project: str, entity: Optional[str] = None) -> str:
        """Upload model card as an artifact to W&B.
        
        Args:
            card: Model card to upload
            project: W&B project name
            entity: W&B entity (optional)
            
        Returns:
            Artifact name
        """
        try:
            # Initialize run if not already active
            if not self.wandb.run:
                self.wandb.init(project=project, entity=entity, job_type="model_card_upload")

            # Create artifact
            artifact_name = f"{card.model_details.name}-model-card"
            artifact = self.wandb.Artifact(
                name=artifact_name,
                type="model_card",
                description=f"Model card for {card.model_details.name}",
                metadata={
                    "model_name": card.model_details.name,
                    "model_version": card.model_details.version,
                    "generated_at": datetime.now().isoformat(),
                    "format": card.config.format.value
                }
            )

            # Add model card files
            card_content = card.render("markdown")
            with artifact.new_file("MODEL_CARD.md", mode="w") as f:
                f.write(card_content)

            # Add JSON version
            json_content = card.render("json")
            with artifact.new_file("model_card.json", mode="w") as f:
                f.write(json_content)

            # Log artifact
            self.wandb.log_artifact(artifact)

            logger.info(f"Uploaded model card artifact: {artifact_name}")
            return artifact_name

        except Exception as e:
            logger.error(f"Failed to upload model card to W&B: {e}")
            raise
        finally:
            if self.wandb.run:
                self.wandb.finish()

    def sync_model_card(self, card: ModelCard, run_path: str) -> None:
        """Sync model card back to a W&B run.
        
        Args:
            card: Model card to sync
            run_path: W&B run path to sync to
        """
        try:
            run = self.api.run(run_path)

            # Update run notes with model card summary
            summary = f"# {card.model_details.name}\n\n"
            summary += f"{card.model_details.description}\n\n"
            summary += "## Key Metrics\n"

            for metric in card.evaluation_results[:5]:  # Top 5 metrics
                summary += f"- **{metric.name}**: {metric.value:.4f}\n"

            run.notes = summary
            run.update()

            # Add model card tags
            new_tags = run.tags + ["model-card-generated", f"version-{card.model_details.version}"]
            run.tags = list(set(new_tags))
            run.update()

            logger.info(f"Synced model card to W&B run {run_path}")

        except Exception as e:
            logger.error(f"Failed to sync model card to W&B run {run_path}: {e}")
            raise
