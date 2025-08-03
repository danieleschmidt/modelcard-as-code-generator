"""
Core model card generation engine.

This module provides the main ModelCardGenerator class that orchestrates
the entire model card generation process from data collection to final output.
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

from .config import CardConfig, GenerationContext
from .model_card import ModelCard, ModelDetails, MetricValue, Dataset
from ..collectors.base import DataCollector
from ..collectors.evaluation import EvaluationCollector
from ..collectors.training import TrainingLogCollector
from ..collectors.config import ConfigCollector
from ..templates.registry import TemplateRegistry
from ..validators.validator import Validator
from ..security.scanner import SecretScanner
from ..monitoring.metrics import get_metrics_collector


class ModelCardGenerator:
    """
    Main model card generation engine.
    
    Orchestrates the entire process of collecting data from various sources,
    applying templates, validating content, and producing the final model card.
    """
    
    def __init__(self, config: Optional[CardConfig] = None):
        """Initialize the generator with configuration."""
        self.config = config or CardConfig()
        self.config.validate()
        
        self.logger = logging.getLogger(__name__)
        self.template_registry = TemplateRegistry()
        self.validator = Validator()
        self.secret_scanner = SecretScanner()
        self.metrics = get_metrics_collector()
        
        # Initialize collectors
        self.collectors = {
            'evaluation': EvaluationCollector(),
            'training': TrainingLogCollector(),
            'config': ConfigCollector(),
        }
        
        self.logger.info(f"Initialized ModelCardGenerator with format: {self.config.format}")
    
    def generate(
        self,
        eval_results: Optional[Union[str, Path, Dict]] = None,
        training_history: Optional[Union[str, Path]] = None,
        dataset_info: Optional[Union[str, Path, Dict]] = None,
        model_config: Optional[Union[str, Path, Dict]] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        **kwargs
    ) -> ModelCard:
        """
        Generate a model card from provided sources.
        
        Args:
            eval_results: Evaluation results (file path or data)
            training_history: Training logs/history (file path)
            dataset_info: Dataset information (file path or data)
            model_config: Model configuration (file path or data)
            model_name: Name of the model
            model_version: Version of the model
            **kwargs: Additional data sources
            
        Returns:
            Generated ModelCard instance
        """
        start_time = datetime.now()
        context = GenerationContext(
            model_name=model_name,
            model_version=model_version,
            generation_timestamp=start_time.isoformat(),
            config_used=self.config
        )
        
        try:
            self.logger.info("Starting model card generation")
            
            # Step 1: Collect data from all sources
            collected_data = self._collect_data(
                eval_results=eval_results,
                training_history=training_history,
                dataset_info=dataset_info,
                model_config=model_config,
                context=context,
                **kwargs
            )
            
            # Step 2: Create initial model card structure
            model_card = self._create_model_card(collected_data, context)
            
            # Step 3: Apply template and populate sections
            self._apply_template(model_card, collected_data, context)
            
            # Step 4: Security scanning
            if self.config.scan_for_secrets:
                self._scan_security(model_card, context)
            
            # Step 5: Validate the generated card
            if self.config.validate_schema or self.config.validate_content:
                self._validate_card(model_card, context)
            
            # Step 6: Record metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_card_generation(
                format_name=self.config.format,
                duration=duration,
                success=not context.has_errors()
            )
            
            if context.has_errors():
                raise ValueError(f"Model card generation failed: {'; '.join(context.errors)}")
            
            self.logger.info(f"Successfully generated model card in {duration:.2f}s")
            return model_card
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_card_generation(
                format_name=self.config.format,
                duration=duration,
                success=False
            )
            self.logger.error(f"Model card generation failed: {e}")
            raise
    
    def _collect_data(
        self,
        eval_results: Optional[Union[str, Path, Dict]],
        training_history: Optional[Union[str, Path]],
        dataset_info: Optional[Union[str, Path, Dict]],
        model_config: Optional[Union[str, Path, Dict]],
        context: GenerationContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Collect data from all provided sources."""
        collected_data = {}
        
        # Collect evaluation results
        if eval_results is not None:
            try:
                eval_data = self.collectors['evaluation'].collect(eval_results)
                collected_data['evaluation'] = eval_data
                context.source_files['evaluation'] = str(eval_results) if isinstance(eval_results, (str, Path)) else 'provided_data'
            except Exception as e:
                context.add_error(f"Failed to collect evaluation data: {e}")
        
        # Collect training history
        if training_history is not None:
            try:
                training_data = self.collectors['training'].collect(training_history)
                collected_data['training'] = training_data
                context.source_files['training'] = str(training_history)
            except Exception as e:
                context.add_error(f"Failed to collect training data: {e}")
        
        # Collect dataset information
        if dataset_info is not None:
            try:
                dataset_data = self._normalize_dataset_info(dataset_info)
                collected_data['datasets'] = dataset_data
                context.source_files['datasets'] = str(dataset_info) if isinstance(dataset_info, (str, Path)) else 'provided_data'
            except Exception as e:
                context.add_error(f"Failed to collect dataset data: {e}")
        
        # Collect model configuration
        if model_config is not None:
            try:
                config_data = self.collectors['config'].collect(model_config)
                collected_data['config'] = config_data
                context.source_files['config'] = str(model_config) if isinstance(model_config, (str, Path)) else 'provided_data'
            except Exception as e:
                context.add_error(f"Failed to collect config data: {e}")
        
        # Collect additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                collected_data[key] = value
                context.source_files[key] = 'provided_data'
        
        return collected_data
    
    def _normalize_dataset_info(self, dataset_info: Union[str, Path, Dict]) -> Dict[str, Any]:
        """Normalize dataset information to standard format."""
        if isinstance(dataset_info, dict):
            return dataset_info
        
        # Load from file
        dataset_path = Path(dataset_info)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset info file not found: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.suffix.lower() == '.json':
                return json.load(f)
            elif dataset_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported dataset info format: {dataset_path.suffix}")
    
    def _create_model_card(self, collected_data: Dict[str, Any], context: GenerationContext) -> ModelCard:
        """Create initial model card structure."""
        model_card = ModelCard()
        
        # Set model details
        model_card.model_details.name = context.model_name or "Unnamed Model"
        model_card.model_details.version = context.model_version or "1.0"
        
        # Extract basic information from config if available
        if 'config' in collected_data:
            config_data = collected_data['config']
            model_card.model_details.description = config_data.get('description')
            model_card.model_details.architecture = config_data.get('architecture')
            model_card.model_details.parameters = config_data.get('parameters')
            model_card.model_details.license = config_data.get('license')
            model_card.model_details.languages = config_data.get('languages', [])
            model_card.model_details.tags = config_data.get('tags', [])
        
        # Add metrics from evaluation data
        if 'evaluation' in collected_data:
            eval_data = collected_data['evaluation']
            metrics = eval_data.get('metrics', {})
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    model_card.add_metric(MetricValue(
                        name=metric_name,
                        value=metric_value.get('value', metric_value),
                        unit=metric_value.get('unit'),
                        description=metric_value.get('description'),
                        dataset=metric_value.get('dataset', 'test'),
                        threshold=metric_value.get('threshold'),
                        confidence_interval=metric_value.get('confidence_interval')
                    ))
                else:
                    model_card.add_metric(MetricValue(
                        name=metric_name,
                        value=metric_value,
                        dataset='test'
                    ))
        
        # Add dataset information
        if 'datasets' in collected_data:
            dataset_data = collected_data['datasets']
            
            # Training datasets
            for ds in dataset_data.get('training', []):
                model_card.add_dataset(Dataset(
                    name=ds.get('name', 'Unknown'),
                    description=ds.get('description'),
                    version=ds.get('version'),
                    url=ds.get('url'),
                    size=ds.get('size'),
                    license=ds.get('license'),
                    preprocessing=ds.get('preprocessing')
                ), dataset_type='training')
            
            # Evaluation datasets
            for ds in dataset_data.get('evaluation', []):
                model_card.add_dataset(Dataset(
                    name=ds.get('name', 'Unknown'),
                    description=ds.get('description'),
                    version=ds.get('version'),
                    url=ds.get('url'),
                    size=ds.get('size'),
                    license=ds.get('license'),
                    preprocessing=ds.get('preprocessing')
                ), dataset_type='evaluation')
        
        return model_card
    
    def _apply_template(self, model_card: ModelCard, collected_data: Dict[str, Any], context: GenerationContext) -> None:
        """Apply template-specific formatting and content."""
        template = self.template_registry.get_template(self.config.format)
        
        if template is None:
            context.add_warning(f"Template not found for format: {self.config.format}")
            return
        
        try:
            # Apply template-specific enhancements
            template.enhance_model_card(model_card, collected_data, self.config)
            
        except Exception as e:
            context.add_error(f"Template application failed: {e}")
    
    def _scan_security(self, model_card: ModelCard, context: GenerationContext) -> None:
        """Scan model card for security issues."""
        try:
            card_data = model_card.to_dict()
            findings = self.secret_scanner.scan_model_card_data(card_data)
            
            # Handle findings based on severity
            critical_findings = [f for f in findings if f.severity == 'critical']
            high_findings = [f for f in findings if f.severity == 'high']
            
            if critical_findings:
                for finding in critical_findings:
                    context.add_error(f"Security issue: {finding.message}")
            
            if high_findings:
                for finding in high_findings:
                    context.add_warning(f"Security warning: {finding.message}")
            
            # Redact sensitive information if configured
            if self.config.redact_sensitive_info and (critical_findings or high_findings):
                self._redact_sensitive_info(model_card, findings)
                
        except Exception as e:
            context.add_warning(f"Security scan failed: {e}")
    
    def _redact_sensitive_info(self, model_card: ModelCard, findings: List) -> None:
        """Redact sensitive information from model card."""
        # This is a simplified implementation - in practice, you'd want
        # more sophisticated redaction logic
        for finding in findings:
            if finding.severity in ['critical', 'high']:
                # Replace with placeholder
                self.logger.warning(f"Redacted sensitive information: {finding.type}")
    
    def _validate_card(self, model_card: ModelCard, context: GenerationContext) -> None:
        """Validate the generated model card."""
        try:
            validation_result = self.validator.validate(model_card, self.config.format)
            
            if not validation_result.is_valid:
                for error in validation_result.errors:
                    context.add_error(f"Validation error: {error}")
            
            for warning in validation_result.warnings:
                context.add_warning(f"Validation warning: {warning}")
            
            # Check completeness score
            completeness = model_card.get_completeness_score()
            if completeness < self.config.min_completeness_score:
                context.add_warning(
                    f"Completeness score {completeness:.2f} below threshold {self.config.min_completeness_score}"
                )
            
        except Exception as e:
            context.add_error(f"Validation failed: {e}")
    
    def render(self, model_card: ModelCard, format: str = "markdown") -> str:
        """Render model card to specified format."""
        template = self.template_registry.get_template(self.config.format)
        
        if template is None:
            raise ValueError(f"Template not found for format: {self.config.format}")
        
        return template.render(model_card, format)
    
    def export(self, model_card: ModelCard, output_path: Path, format: Optional[str] = None) -> None:
        """Export model card to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format is None:
            format = self.config.output_format
        
        if format == "markdown":
            content = self.render(model_card, "markdown")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format == "json":
            model_card.save_json(output_path)
        elif format == "yaml":
            model_card.save_yaml(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported model card to: {output_path}")
    
    def validate_sources(self, **sources) -> Dict[str, bool]:
        """Validate that all provided sources are accessible and valid."""
        results = {}
        
        for source_name, source_value in sources.items():
            if source_value is None:
                results[source_name] = True  # None is valid (optional)
                continue
            
            try:
                if isinstance(source_value, (str, Path)):
                    # Check if file exists
                    file_path = Path(source_value)
                    results[source_name] = file_path.exists()
                else:
                    # Assume it's data
                    results[source_name] = True
            except Exception:
                results[source_name] = False
        
        return results