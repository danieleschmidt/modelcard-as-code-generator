"""Enhanced Weights & Biases integration with performance optimizations and scaling features."""

import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import time
from functools import wraps, lru_cache
import gzip

from ..core.models import ModelCard, CardConfig
from ..core.generator import ModelCardGenerator
from ..core.cache import CacheManager
from ..core.exceptions import ModelCardError


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


class EnhancedWandbIntegration:
    """Enhanced W&B integration with performance optimizations and scaling features."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 cache_ttl: int = 3600,
                 max_workers: int = 4,
                 enable_compression: bool = True,
                 enable_monitoring: bool = False):
        """Initialize enhanced W&B integration.
        
        Args:
            api_key: W&B API key
            cache_ttl: Cache time-to-live in seconds
            max_workers: Max concurrent workers for parallel processing
            enable_compression: Enable response compression
            enable_monitoring: Enable performance monitoring
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
        self.enable_monitoring = enable_monitoring
        
        # Request statistics
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    @rate_limit(max_calls=50, time_window=60)
    def from_run(self, run_path: str, config: Optional[CardConfig] = None) -> ModelCard:
        """Generate model card from a W&B run with enhanced performance.
        
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
            self.stats['cache_hits'] += 1
            logger.debug(f"Cache hit for run {run_path}")
            return cached_result
        
        self.stats['cache_misses'] += 1
        self.stats['api_calls'] += 1
        
        try:
            # Load run with retry logic
            run = self._load_run_with_retry(run_path)
            logger.info(f"Loaded W&B run: {run.name} ({run.id})")
            
            # Create model card
            card = ModelCard(config or CardConfig())
            
            # Extract information concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._extract_model_details, card, run): 'model_details',
                    executor.submit(self._extract_training_details, card, run): 'training_details', 
                    executor.submit(self._extract_metrics, card, run): 'metrics',
                    executor.submit(self._extract_artifacts, card, run): 'artifacts'
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
            self.stats['total_time'] += execution_time
            
            logger.info(f"Generated model card from W&B run {run.name} in {execution_time:.2f}s")
            return card
            
        except Exception as e:
            self.stats['errors'] += 1
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
        """Generate model cards from W&B project runs asynchronously."""
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
        """Generate model cards from W&B project (synchronous version)."""
        # Use asyncio to run the async version
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self.from_project_async(project_path, filters, max_runs)
            )
        finally:
            loop.close()
    
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
        if hasattr(run, 'user') and run.user:
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
        """Extract training details from W&B run with enhanced parsing."""
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
        
        # Extract hyperparameters with better filtering
        hyperparams = {}
        common_hyperparam_keys = [
            "learning_rate", "lr", "batch_size", "epochs", "num_epochs",
            "optimizer", "weight_decay", "dropout", "hidden_size",
            "num_layers", "max_length", "warmup_steps", "seed",
            "model_name", "model_type", "num_attention_heads", "intermediate_size"
        ]
        
        for key, value in config.items():
            # Skip wandb internal parameters and very long strings
            if key.startswith("_wandb_") or (isinstance(value, str) and len(value) > 500):
                continue
                
            if (key in common_hyperparam_keys or 
                key.startswith(("train_", "training_", "model_")) or
                key.endswith(("_rate", "_size", "_steps", "_layers"))):
                
                # Convert certain values to more readable formats
                if isinstance(value, float) and key in ["learning_rate", "lr", "weight_decay"]:
                    hyperparams[key] = f"{value:.2e}" if value < 0.001 else f"{value:.4f}"
                else:
                    hyperparams[key] = value
        
        # Limit hyperparameters to prevent excessive metadata
        if len(hyperparams) > 20:
            important_keys = ["learning_rate", "lr", "batch_size", "epochs", "num_epochs", "optimizer"]
            filtered_hyperparams = {k: v for k, v in hyperparams.items() if k in important_keys}
            remaining_keys = [k for k in hyperparams.keys() if k not in important_keys]
            filtered_hyperparams.update({k: hyperparams[k] for k in remaining_keys[:14]})  # Top 14 remaining
            hyperparams = filtered_hyperparams
            hyperparams["_truncated"] = f"Showing {len(hyperparams)} of {len(config)} config items"
        
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
        
        # Extract hardware info from system metrics with better parsing
        if hasattr(run, 'system_metrics') and run.system_metrics:
            gpu_info = []
            cpu_info = []
            memory_info = []
            
            for key, value in run.system_metrics.items():
                key_lower = key.lower()
                if "gpu" in key_lower:
                    gpu_info.append(f"{key}: {value}")
                elif "cpu" in key_lower:
                    cpu_info.append(f"{key}: {value}")
                elif "memory" in key_lower or "ram" in key_lower:
                    memory_info.append(f"{key}: {value}")
            
            hardware_parts = []
            if gpu_info:
                hardware_parts.append(f"GPU: {'; '.join(gpu_info[:3])}")  # Limit to 3 GPU metrics
            if cpu_info:
                hardware_parts.append(f"CPU: {'; '.join(cpu_info[:2])}")  # Limit to 2 CPU metrics  
            if memory_info:
                hardware_parts.append(f"Memory: {'; '.join(memory_info[:2])}")  # Limit to 2 memory metrics
            
            if hardware_parts:
                card.training_details.hardware = " | ".join(hardware_parts)
        
        # Training time from run duration
        if run.finished_at and run.created_at:
            duration = run.finished_at - run.created_at
            card.training_details.training_time = str(duration)
    
    def _extract_metrics(self, card: ModelCard, run) -> None:
        """Extract evaluation metrics from W&B run with enhanced filtering."""
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
            
            # Skip metrics that are too large or small (likely outliers)
            if isinstance(value, float) and (abs(value) > 1e6 or (abs(value) < 1e-10 and value != 0)):
                continue
            
            if is_metric or is_validation:
                try:
                    card.add_metric(key, float(value))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipped non-numeric metric {key}: {e}")
        
        # Extract best metrics if available
        for key, value in summary.items():
            if key.startswith("best_") and isinstance(value, (int, float)):
                try:
                    # Skip metrics that are too large or small
                    if isinstance(value, float) and (abs(value) > 1e6 or (abs(value) < 1e-10 and value != 0)):
                        continue
                    
                    metric_name = key.replace("best_", "")
                    card.add_metric(f"best_{metric_name}", float(value))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipped invalid best metric {key}: {e}")
    
    def _extract_artifacts(self, card: ModelCard, run) -> None:
        """Extract artifact information from W&B run with size limits."""
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
            
            # Add artifact info to card metadata with size limits
            if model_artifacts:
                # Limit artifacts to prevent excessive metadata
                card.metadata["wandb_model_artifacts"] = model_artifacts[:10]
                if len(model_artifacts) > 10:
                    card.metadata["wandb_total_model_artifacts"] = len(model_artifacts)
            
            if dataset_artifacts:
                # Limit dataset artifacts
                limited_datasets = dataset_artifacts[:5]
                card.training_details.training_data.extend(limited_datasets)
                card.metadata["wandb_dataset_artifacts"] = limited_datasets
                if len(dataset_artifacts) > 5:
                    card.metadata["wandb_total_dataset_artifacts"] = len(dataset_artifacts)
            
        except Exception as e:
            logger.warning(f"Failed to extract artifacts from run {run.id}: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the integration."""
        stats = self.stats.copy()
        stats.update({
            'cache_hit_ratio': stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses']),
            'avg_api_time': stats['total_time'] / max(1, stats['api_calls']),
            'cache_size': len(self.cache._cache) if hasattr(self.cache, '_cache') else 0,
            'error_rate': stats['errors'] / max(1, stats['api_calls'])
        })
        return stats
    
    def clear_cache(self) -> None:
        """Clear the integration cache."""
        self.cache.clear()
        logger.info("Cleared W&B integration cache")
    
    def optimize_for_batch_processing(self) -> None:
        """Optimize settings for batch processing of multiple runs."""
        self.max_workers = min(8, self.max_workers * 2)
        logger.info(f"Optimized for batch processing: {self.max_workers} workers")
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on W&B integration."""
        results = {
            'api_connection': False,
            'authentication': False,
            'rate_limiting': True,  # Always true as it's implemented
            'cache_working': False
        }
        
        try:
            # Test API connection
            self.api.viewer
            results['api_connection'] = True
            results['authentication'] = True
        except Exception as e:
            logger.warning(f"W&B API health check failed: {e}")
        
        try:
            # Test cache
            test_key = "health_check_test"
            test_value = "test"
            self.cache.set(test_key, test_value)
            retrieved = self.cache.get(test_key)
            results['cache_working'] = retrieved == test_value
            self.cache.delete(test_key)
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
        
        return results


class WandbBatchProcessor:
    """Specialized class for high-performance batch processing of W&B data."""
    
    def __init__(self, integration: EnhancedWandbIntegration):
        self.integration = integration
        self.processing_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'start_time': None,
            'batches_completed': 0
        }
    
    async def process_runs_parallel(self, 
                                  run_paths: List[str],
                                  batch_size: int = 20,
                                  max_concurrent: int = 5) -> List[ModelCard]:
        """Process multiple runs in parallel with batching."""
        self.processing_stats['start_time'] = time.time()
        all_cards = []
        
        # Create semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch_runs):
            async with semaphore:
                batch_cards = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.integration.max_workers) as executor:
                    futures = {
                        executor.submit(self.integration.from_run, run_path): run_path
                        for run_path in batch_runs
                    }
                    
                    for future in concurrent.futures.as_completed(futures):
                        run_path = futures[future]
                        try:
                            card = future.result(timeout=120)  # 2 minute timeout per run
                            batch_cards.append(card)
                            self.processing_stats['total_processed'] += 1
                        except Exception as e:
                            logger.error(f"Failed to process run {run_path}: {e}")
                            self.processing_stats['total_failed'] += 1
                
                self.processing_stats['batches_completed'] += 1
                logger.info(f"Completed batch {self.processing_stats['batches_completed']}: "
                           f"{len(batch_cards)}/{len(batch_runs)} successful")
                
                return batch_cards
        
        # Create batches
        batches = [run_paths[i:i + batch_size] for i in range(0, len(run_paths), batch_size)]
        logger.info(f"Processing {len(run_paths)} runs in {len(batches)} batches")
        
        # Process all batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
            else:
                all_cards.extend(result)
        
        total_time = time.time() - self.processing_stats['start_time']
        logger.info(f"Batch processing completed: {self.processing_stats['total_processed']} processed, "
                   f"{self.processing_stats['total_failed']} failed in {total_time:.2f}s")
        
        return all_cards
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current batch processing statistics."""
        stats = self.processing_stats.copy()
        if stats['start_time']:
            stats['elapsed_time'] = time.time() - stats['start_time']
            if stats['total_processed'] > 0:
                stats['avg_time_per_run'] = stats['elapsed_time'] / stats['total_processed']
        return stats