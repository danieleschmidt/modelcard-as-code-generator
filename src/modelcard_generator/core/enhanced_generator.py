"""Enhanced model card generator with robustness features."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .generator import ModelCardGenerator as BaseModelCardGenerator, DataSourceParser
from .models import ModelCard, CardConfig
from .exceptions import ModelCardError, DataSourceError, ResourceError
from .logging_config import get_logger
from .config import get_config
from .security import sanitizer, scan_for_vulnerabilities
from .rate_limiter import api_rate_limiter, file_operation_rate_limiter
from .circuit_breaker import registry as circuit_registry, API_CIRCUIT_CONFIG, FILE_CIRCUIT_CONFIG
from .retry import api_retry, file_retry
from .performance_monitor import performance_monitor, performance_tracker

logger = get_logger(__name__)


class EnhancedDataSourceParser(DataSourceParser):
    """Enhanced data source parser with reliability features."""
    
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    @performance_monitor("parse_json_enhanced")
    @file_retry
    async def parse_json_async(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse JSON with async support and enhanced error handling."""
        # Use circuit breaker for file operations
        return await circuit_registry.call_with_breaker(
            "file_operations", 
            self._parse_json_with_rate_limit,
            file_path,
            config=FILE_CIRCUIT_CONFIG
        )
    
    async def _parse_json_with_rate_limit(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse JSON with rate limiting."""
        # Apply rate limiting
        if not await file_operation_rate_limiter.wait_for_permission(str(file_path)):
            raise ResourceError("rate_limit", "File operation rate limit exceeded")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.parse_json, file_path)
    
    @performance_monitor("parse_yaml_enhanced")
    @file_retry
    async def parse_yaml_async(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse YAML with async support and enhanced error handling."""
        return await circuit_registry.call_with_breaker(
            "file_operations",
            self._parse_yaml_with_rate_limit,
            file_path,
            config=FILE_CIRCUIT_CONFIG
        )
    
    async def _parse_yaml_with_rate_limit(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse YAML with rate limiting."""
        if not await file_operation_rate_limiter.wait_for_permission(str(file_path)):
            raise ResourceError("rate_limit", "File operation rate limit exceeded")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.parse_yaml, file_path)
    
    @performance_monitor("parse_csv_enhanced")
    @file_retry
    async def parse_csv_async(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parse CSV with async support and enhanced error handling."""
        return await circuit_registry.call_with_breaker(
            "file_operations",
            self._parse_csv_with_rate_limit,
            file_path,
            config=FILE_CIRCUIT_CONFIG
        )
    
    async def _parse_csv_with_rate_limit(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parse CSV with rate limiting."""
        if not await file_operation_rate_limiter.wait_for_permission(str(file_path)):
            raise ResourceError("rate_limit", "File operation rate limit exceeded")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.parse_csv, file_path)


class EnhancedModelCardGenerator(BaseModelCardGenerator):
    """Enhanced model card generator with advanced reliability and performance features."""
    
    def __init__(self, config: Optional[CardConfig] = None):
        super().__init__(config)
        self.enhanced_parser = EnhancedDataSourceParser()
        self.generation_stats = {
            "total_generated": 0,
            "total_failures": 0,
            "avg_generation_time": 0.0
        }
    
    @performance_monitor("generate_enhanced", include_metadata=True)
    async def generate_async(
        self,
        eval_results: Optional[Union[str, Path, Dict[str, Any]]] = None,
        training_history: Optional[Union[str, Path]] = None,
        dataset_info: Optional[Union[str, Path, Dict[str, Any]]] = None,
        model_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        **kwargs
    ) -> ModelCard:
        """Generate model card asynchronously with enhanced error handling."""
        
        start_time = time.time()
        operation = "async_model_card_generation"
        
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
            
            # Process inputs concurrently with enhanced error handling
            tasks = []
            
            if eval_results:
                tasks.append(self._process_evaluation_results_async(card, eval_results))
            if training_history:
                tasks.append(self._process_training_history_async(card, training_history))
            if dataset_info:
                tasks.append(self._process_dataset_info_async(card, dataset_info))
            if model_config:
                tasks.append(self._process_model_config_async(card, model_config))
            
            # Execute all tasks concurrently with error isolation
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Processing task {i} failed: {result}")
                        # Continue with partial data rather than failing completely
            
            # Apply additional metadata
            if kwargs:
                self._apply_additional_metadata(card, kwargs)
            
            # Auto-populate missing information if enabled
            if self.config.auto_populate:
                self._auto_populate_sections(card)
            
            # Enhanced validation with circuit breaker
            if self.app_config.auto_validate:
                await self._validate_with_circuit_breaker(card)
            
            # Security scan with rate limiting
            if self.app_config.auto_scan_security:
                await self._security_scan_with_rate_limit(card)
            
            duration_ms = (time.time() - start_time) * 1000
            self.generation_stats["total_generated"] += 1
            self.generation_stats["avg_generation_time"] = (
                (self.generation_stats["avg_generation_time"] * (self.generation_stats["total_generated"] - 1) + duration_ms) / 
                self.generation_stats["total_generated"]
            )
            
            logger.log_operation_success(operation, duration_ms, 
                                       model_name=card.model_details.name,
                                       metrics_count=len(card.evaluation_results))
            
            return card
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.generation_stats["total_failures"] += 1
            logger.log_operation_failure(operation, e, duration_ms)
            
            if isinstance(e, ModelCardError):
                raise
            else:
                raise ModelCardError(f"Enhanced model card generation failed: {e}", 
                                   details={"operation": operation, "duration_ms": duration_ms})
    
    async def _process_evaluation_results_async(self, card: ModelCard, eval_results: Union[str, Path, Dict[str, Any]]) -> None:
        """Process evaluation results asynchronously."""
        if isinstance(eval_results, (str, Path)):
            path = Path(eval_results)
            if path.suffix == '.json':
                data = await self.enhanced_parser.parse_json_async(path)
            elif path.suffix in ['.yaml', '.yml']:
                data = await self.enhanced_parser.parse_yaml_async(path)
            elif path.suffix == '.csv':
                csv_data = await self.enhanced_parser.parse_csv_async(path)
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
        
        # Extract metrics (same logic as base class)
        metric_fields = ['accuracy', 'precision', 'recall', 'f1', 'f1_score', 'f1_macro', 'f1_micro', 
                        'auc', 'roc_auc', 'loss', 'val_loss', 'mse', 'rmse', 'mae', 'r2',
                        'bleu', 'rouge', 'perplexity', 'inference_time', 'inference_time_ms']
        
        for field in metric_fields:
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)):
                    card.add_metric(field, float(value))
                elif isinstance(value, dict) and 'value' in value:
                    ci = value.get('confidence_interval')
                    card.add_metric(field, float(value['value']), confidence_interval=ci)
        
        # Extract additional information
        if not card.model_details.name and 'model_name' in data:
            card.model_details.name = data['model_name']
        
        if 'dataset' in data:
            if isinstance(data['dataset'], str):
                card.training_details.training_data.append(data['dataset'])
            elif isinstance(data['dataset'], list):
                card.training_details.training_data.extend(data['dataset'])
    
    async def _process_training_history_async(self, card: ModelCard, training_history: Union[str, Path]) -> None:
        """Process training history asynchronously."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        log_data = await loop.run_in_executor(None, self.parser.parse_training_log, training_history)
        
        # Update card with training data
        card.training_details.hyperparameters.update(log_data["hyperparameters"])
        
        if log_data["hardware"]:
            card.training_details.hardware = log_data["hardware"]
        
        if log_data["metrics"]:
            final_metrics = log_data["metrics"][-1]
            for metric_name, value in final_metrics.items():
                card.add_metric(f"final_{metric_name}", value)
    
    async def _process_dataset_info_async(self, card: ModelCard, dataset_info: Union[str, Path, Dict[str, Any]]) -> None:
        """Process dataset information asynchronously."""
        if isinstance(dataset_info, (str, Path)):
            path = Path(dataset_info)
            if path.suffix == '.json':
                data = await self.enhanced_parser.parse_json_async(path)
            elif path.suffix in ['.yaml', '.yml']:
                data = await self.enhanced_parser.parse_yaml_async(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            data = dataset_info
        
        # Process dataset information (same logic as base class)
        if 'datasets' in data:
            if isinstance(data['datasets'], list):
                card.model_details.datasets.extend(data['datasets'])
            else:
                card.model_details.datasets.append(str(data['datasets']))
        
        if 'training_data' in data:
            if isinstance(data['training_data'], list):
                card.training_details.training_data.extend(data['training_data'])
            else:
                card.training_details.training_data.append(str(data['training_data']))
        
        if 'preprocessing' in data:
            card.training_details.preprocessing = data['preprocessing']
        
        if 'bias_analysis' in data:
            bias_data = data['bias_analysis']
            if 'bias_risks' in bias_data:
                card.ethical_considerations.bias_risks.extend(bias_data['bias_risks'])
            if 'fairness_metrics' in bias_data:
                card.ethical_considerations.fairness_metrics.update(bias_data['fairness_metrics'])
    
    async def _process_model_config_async(self, card: ModelCard, model_config: Union[str, Path, Dict[str, Any]]) -> None:
        """Process model configuration asynchronously."""
        if isinstance(model_config, (str, Path)):
            path = Path(model_config)
            if path.suffix == '.json':
                data = await self.enhanced_parser.parse_json_async(path)
            elif path.suffix in ['.yaml', '.yml']:
                data = await self.enhanced_parser.parse_yaml_async(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            data = model_config
        
        # Process configuration (same logic as base class)
        config_fields = {
            'name': 'name',
            'version': 'version', 
            'description': 'description',
            'license': 'license',
            'authors': 'authors',
            'base_model': 'base_model',
            'language': 'language',
            'tags': 'tags'
        }
        
        for config_key, card_attr in config_fields.items():
            if config_key in data:
                value = data[config_key]
                if config_key in ['authors', 'language', 'tags'] and not isinstance(value, list):
                    value = [value]
                setattr(card.model_details, card_attr, value)
        
        # Training details
        training_fields = {
            'framework': 'framework',
            'architecture': 'model_architecture',
            'hyperparameters': 'hyperparameters'
        }
        
        for config_key, training_attr in training_fields.items():
            if config_key in data:
                if config_key == 'hyperparameters':
                    card.training_details.hyperparameters.update(data[config_key])
                else:
                    setattr(card.training_details, training_attr, data[config_key])
    
    async def _validate_with_circuit_breaker(self, card: ModelCard) -> None:
        """Validate card with circuit breaker protection."""
        try:
            from .validator import Validator
            validator = Validator()
            
            async def validation_task():
                result = validator.validate_schema(card, self.config.format.value)
                logger.log_validation_result(result.is_valid, result.score, result.issues)
                
                if not result.is_valid and self.app_config.validation.enforce_compliance:
                    raise ValidationError("Model card validation failed", result.issues)
                return result
            
            await circuit_registry.call_with_breaker(
                "validation",
                validation_task,
                config=API_CIRCUIT_CONFIG
            )
            
        except Exception as e:
            logger.error(f"Validation with circuit breaker failed: {e}")
            # Don't fail generation due to validation issues unless enforced
            if self.app_config.validation.enforce_compliance:
                raise
    
    async def _security_scan_with_rate_limit(self, card: ModelCard) -> None:
        """Security scan with rate limiting."""
        try:
            # Apply rate limiting
            if not await api_rate_limiter.wait_for_permission("security_scan"):
                logger.warning("Security scan rate limited")
                return
            
            content = card.render()
            scan_result = scan_for_vulnerabilities(content)
            logger.log_security_check("content_scan", scan_result["passed"], scan_result)
            
            if not scan_result["passed"]:
                logger.warning("Security vulnerabilities detected in generated content")
        
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            # Don't fail generation due to security scan issues
    
    @performance_monitor("batch_generate_enhanced")
    async def generate_batch_async(self, tasks: List[Dict[str, Any]], max_concurrent: int = 3) -> List[ModelCard]:
        """Generate multiple model cards concurrently with enhanced reliability."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        failed_tasks = []
        
        async def generate_single(task_index: int, task: Dict[str, Any]):
            async with semaphore:
                try:
                    result = await self.generate_async(**task)
                    logger.info(f"Batch task {task_index} completed successfully")
                    return task_index, result
                except Exception as e:
                    logger.error(f"Batch task {task_index} failed: {e}")
                    failed_tasks.append((task_index, task, str(e)))
                    return task_index, None
        
        # Submit all tasks
        task_futures = [generate_single(i, task) for i, task in enumerate(tasks)]
        
        # Collect results
        completed_results = await asyncio.gather(*task_futures, return_exceptions=True)
        
        # Sort and filter results
        successful_results = []
        for result in completed_results:
            if isinstance(result, Exception):
                logger.error(f"Batch task failed with exception: {result}")
                continue
            
            task_index, model_card = result
            if model_card is not None:
                successful_results.append((task_index, model_card))
        
        # Sort by original order and extract model cards
        successful_results.sort(key=lambda x: x[0])
        results = [model_card for _, model_card in successful_results]
        
        # Log batch summary
        success_count = len(results)
        failure_count = len(failed_tasks)
        total_count = len(tasks)
        
        logger.info(f"Batch generation completed", 
                   total=total_count, successful=success_count, failed=failure_count)
        
        if failed_tasks:
            logger.warning(f"Failed batch tasks: {[(i, str(e)) for i, _, e in failed_tasks]}")
        
        return results
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            **self.generation_stats,
            "success_rate": (
                (self.generation_stats["total_generated"] - self.generation_stats["total_failures"]) / 
                max(1, self.generation_stats["total_generated"])
            )
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "generation_stats": self.get_generation_statistics(),
            "performance_stats": {
                "generate_enhanced": performance_tracker.get_operation_stats("generate_enhanced"),
                "parse_json_enhanced": performance_tracker.get_operation_stats("parse_json_enhanced"),
                "parse_yaml_enhanced": performance_tracker.get_operation_stats("parse_yaml_enhanced"),
                "parse_csv_enhanced": performance_tracker.get_operation_stats("parse_csv_enhanced"),
                "batch_generate_enhanced": performance_tracker.get_operation_stats("batch_generate_enhanced")
            },
            "circuit_breaker_status": circuit_registry.get_all_status(),
            "rate_limiter_stats": {
                "api_rate_limiter": api_rate_limiter.rate_limiter.get_stats(),
                "file_operation_rate_limiter": file_operation_rate_limiter.rate_limiter.get_stats()
            }
        }