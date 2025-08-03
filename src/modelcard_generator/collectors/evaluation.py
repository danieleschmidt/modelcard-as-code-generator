"""
Evaluation results collector.

Collects and normalizes evaluation metrics and results from various formats
including JSON, CSV, and common ML evaluation outputs.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import logging

from .base import DataCollector


class EvaluationCollector(DataCollector):
    """Collector for evaluation results and metrics."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.json', '.csv', '.yaml', '.yml', '.txt', '.log']
    
    def collect(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect evaluation data from source.
        
        Supports:
        - JSON files with metrics
        - CSV files with evaluation results
        - Text/log files with parsed metrics
        - Direct dictionary data
        """
        if isinstance(source, dict):
            return self._normalize_evaluation_data(source)
        
        file_path = self._normalize_path(source)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {file_path}")
        
        if not self.validate_source(file_path):
            raise ValueError(f"Unsupported evaluation file format: {file_path.suffix}")
        
        # Load data based on file type
        if file_path.suffix.lower() in ['.json']:
            data = self._load_json_file(file_path)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            data = self._load_yaml_file(file_path)
        elif file_path.suffix.lower() == '.csv':
            data = self._load_csv_evaluation(file_path)
        elif file_path.suffix.lower() in ['.txt', '.log']:
            data = self._parse_text_evaluation(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self._normalize_evaluation_data(data)
    
    def validate_source(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Validate evaluation source."""
        if isinstance(source, dict):
            return True  # Direct data is always valid
        
        file_path = self._normalize_path(source)
        
        # Check if file exists and has supported extension
        return (
            file_path.exists() and 
            file_path.suffix.lower() in self.supported_formats
        )
    
    def _normalize_evaluation_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize evaluation data to standard format."""
        normalized = {
            'metrics': {},
            'metadata': {},
            'datasets': [],
            'model_info': {}
        }
        
        # Handle different input formats
        if 'metrics' in data:
            # Already normalized format
            normalized['metrics'] = data['metrics']
        elif 'results' in data:
            # Common evaluation format
            normalized['metrics'] = self._extract_metrics_from_results(data['results'])
        elif 'evaluation' in data:
            # Nested evaluation format
            eval_data = data['evaluation']
            if isinstance(eval_data, dict):
                normalized['metrics'] = self._extract_metrics_from_dict(eval_data)
        else:
            # Assume top-level metrics
            normalized['metrics'] = self._extract_metrics_from_dict(data)
        
        # Extract metadata
        for key in ['dataset', 'model', 'timestamp', 'version', 'config']:
            if key in data:
                normalized['metadata'][key] = data[key]
        
        # Extract dataset information
        if 'dataset' in data:
            dataset_info = data['dataset']
            if isinstance(dataset_info, dict):
                normalized['datasets'] = [dataset_info]
            elif isinstance(dataset_info, list):
                normalized['datasets'] = dataset_info
        
        # Extract model information
        for key in ['model_name', 'model_version', 'architecture']:
            if key in data:
                normalized['model_info'][key] = data[key]
        
        return normalized
    
    def _extract_metrics_from_results(self, results: Union[Dict, List]) -> Dict[str, Any]:
        """Extract metrics from results section."""
        metrics = {}
        
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
                elif isinstance(value, dict) and 'value' in value:
                    metrics[key] = value
        elif isinstance(results, list):
            # Handle list of result dictionaries
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            metrics[f"{key}_{i}"] = value
        
        return metrics
    
    def _extract_metrics_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numeric metrics from dictionary."""
        metrics = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            elif isinstance(value, dict):
                # Check if it's a metric with metadata
                if 'value' in value:
                    metrics[key] = value
                elif 'score' in value:
                    metrics[key] = {'value': value['score'], **{k: v for k, v in value.items() if k != 'score'}}
        
        return metrics
    
    def _load_csv_evaluation(self, file_path: Path) -> Dict[str, Any]:
        """Load evaluation data from CSV file."""
        csv_data = self._load_csv_file(file_path)
        
        # Try to detect CSV format
        if not csv_data:
            return {}
        
        first_row = csv_data[0]
        
        # Format 1: metric_name, value columns
        if 'metric_name' in first_row and 'value' in first_row:
            metrics = {}
            for row in csv_data:
                metrics[row['metric_name']] = self._parse_numeric_value(row['value'])
            return {'metrics': metrics}
        
        # Format 2: Each column is a metric
        elif all(self._is_numeric_column(first_row[col]) for col in first_row if col not in ['dataset', 'model', 'timestamp']):
            metrics = {}
            for col in first_row:
                if col not in ['dataset', 'model', 'timestamp']:
                    # Average values if multiple rows
                    values = [self._parse_numeric_value(row[col]) for row in csv_data if self._is_numeric_value(row[col])]
                    if values:
                        metrics[col] = sum(values) / len(values)
            return {'metrics': metrics}
        
        # Format 3: Single row with metrics as columns
        else:
            metrics = {}
            for col, value in first_row.items():
                if self._is_numeric_value(value):
                    metrics[col] = self._parse_numeric_value(value)
            return {'metrics': metrics}
    
    def _parse_text_evaluation(self, file_path: Path) -> Dict[str, Any]:
        """Parse evaluation metrics from text/log file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics = {}
        
        # Common patterns for metrics in logs
        patterns = [
            # Pattern: "accuracy: 0.95"
            r'(\w+):\s*([\d.]+)',
            # Pattern: "Accuracy = 0.95"
            r'(\w+)\s*=\s*([\d.]+)',
            # Pattern: "Test accuracy 0.95"
            r'(?:test\s+)?(\w+)\s+([\d.]+)',
            # Pattern: "Final accuracy: 95.2%"
            r'(?:final\s+)?(\w+):\s*([\d.]+)%?',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for metric_name, value in matches:
                # Clean metric name
                metric_name = metric_name.lower().strip()
                if metric_name and self._is_valid_metric_name(metric_name):
                    try:
                        metrics[metric_name] = float(value)
                    except ValueError:
                        continue
        
        # Look for JSON blocks in logs
        json_blocks = re.findall(r'{[^{}]*}', content)
        for block in json_blocks:
            try:
                json_data = json.loads(block)
                if isinstance(json_data, dict):
                    for key, value in json_data.items():
                        if isinstance(value, (int, float)) and self._is_valid_metric_name(key):
                            metrics[key] = value
            except json.JSONDecodeError:
                continue
        
        return {'metrics': metrics}
    
    def _is_numeric_value(self, value: Any) -> bool:
        """Check if value is numeric."""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False
    
    def _parse_numeric_value(self, value: Any) -> float:
        """Parse value as numeric."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove percentage sign if present
            value = value.strip('%')
            return float(value)
        raise ValueError(f"Cannot parse as numeric: {value}")
    
    def _is_numeric_column(self, value: Any) -> bool:
        """Check if column contains numeric data."""
        return self._is_numeric_value(value)
    
    def _is_valid_metric_name(self, name: str) -> bool:
        """Check if name is a valid metric name."""
        # Filter out common non-metric words
        invalid_names = {
            'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'epoch', 'step', 'batch', 'time', 'date', 'id', 'name', 'file',
            'path', 'url', 'config', 'param', 'arg', 'flag'
        }
        
        name = name.lower().strip()
        return (
            len(name) > 1 and
            name not in invalid_names and
            not name.isdigit() and
            any(c.isalpha() for c in name)
        )


class HuggingFaceEvaluationCollector(EvaluationCollector):
    """Specialized collector for Hugging Face evaluation results."""
    
    def _normalize_evaluation_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize HF-specific evaluation format."""
        normalized = super()._normalize_evaluation_data(data)
        
        # Handle HF-specific format
        if 'eval' in data:
            eval_data = data['eval']
            if isinstance(eval_data, dict):
                for key, value in eval_data.items():
                    if key.startswith('eval_'):
                        metric_name = key[5:]  # Remove 'eval_' prefix
                        normalized['metrics'][metric_name] = value
        
        return normalized


class MLflowEvaluationCollector(EvaluationCollector):
    """Specialized collector for MLflow evaluation results."""
    
    def _normalize_evaluation_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize MLflow-specific evaluation format."""
        normalized = super()._normalize_evaluation_data(data)
        
        # Handle MLflow metrics format
        if 'metrics' in data and isinstance(data['metrics'], list):
            metrics_dict = {}
            for metric in data['metrics']:
                if isinstance(metric, dict) and 'key' in metric and 'value' in metric:
                    metrics_dict[metric['key']] = metric['value']
            normalized['metrics'] = metrics_dict
        
        return normalized


class WandBEvaluationCollector(EvaluationCollector):
    """Specialized collector for Weights & Biases evaluation results."""
    
    def _normalize_evaluation_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize W&B-specific evaluation format."""
        normalized = super()._normalize_evaluation_data(data)
        
        # Handle W&B summary format
        if 'summary' in data:
            summary = data['summary']
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if isinstance(value, (int, float)):
                        normalized['metrics'][key] = value
        
        return normalized