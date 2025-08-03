"""
Training log collector.

Collects and normalizes training information from logs, history files,
and training configuration data.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import logging
from datetime import datetime

from .base import DataCollector


class TrainingLogCollector(DataCollector):
    """Collector for training logs and history."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.log', '.txt', '.json', '.csv', '.yaml', '.yml']
    
    def collect(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect training data from source.
        
        Supports:
        - Log files with training metrics
        - JSON training history
        - CSV training logs
        - Direct dictionary data
        """
        if isinstance(source, dict):
            return self._normalize_training_data(source)
        
        file_path = self._normalize_path(source)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Training file not found: {file_path}")
        
        if not self.validate_source(file_path):
            raise ValueError(f"Unsupported training file format: {file_path.suffix}")
        
        # Load data based on file type
        if file_path.suffix.lower() in ['.json']:
            data = self._load_json_file(file_path)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            data = self._load_yaml_file(file_path)
        elif file_path.suffix.lower() == '.csv':
            data = self._load_csv_training(file_path)
        elif file_path.suffix.lower() in ['.log', '.txt']:
            data = self._parse_training_log(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self._normalize_training_data(data)
    
    def validate_source(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Validate training source."""
        if isinstance(source, dict):
            return True
        
        file_path = self._normalize_path(source)
        return (
            file_path.exists() and 
            file_path.suffix.lower() in self.supported_formats
        )
    
    def _normalize_training_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize training data to standard format."""
        normalized = {
            'epochs': [],
            'final_metrics': {},
            'hyperparameters': {},
            'training_time': None,
            'model_info': {},
            'metadata': {}
        }
        
        # Handle different training data formats
        if 'history' in data:
            # Keras-style history
            normalized['epochs'] = self._extract_epoch_data(data['history'])
            normalized['final_metrics'] = self._get_final_metrics(normalized['epochs'])
        elif 'epochs' in data:
            # Direct epoch data
            normalized['epochs'] = data['epochs']
            normalized['final_metrics'] = self._get_final_metrics(normalized['epochs'])
        elif 'logs' in data:
            # Log-style data
            normalized['epochs'] = self._extract_logs_data(data['logs'])
            normalized['final_metrics'] = self._get_final_metrics(normalized['epochs'])
        else:
            # Try to extract from top-level
            if self._has_epoch_structure(data):
                normalized['epochs'] = [data]
                normalized['final_metrics'] = self._extract_metrics_from_dict(data)
            else:
                # Assume final metrics
                normalized['final_metrics'] = self._extract_metrics_from_dict(data)
        
        # Extract hyperparameters
        for key in ['hyperparameters', 'params', 'config', 'args']:
            if key in data:
                normalized['hyperparameters'].update(data[key])
        
        # Extract training time
        if 'training_time' in data:
            normalized['training_time'] = data['training_time']
        elif 'duration' in data:
            normalized['training_time'] = data['duration']
        
        # Extract model information
        for key in ['model_name', 'architecture', 'parameters']:
            if key in data:
                normalized['model_info'][key] = data[key]
        
        # Extract metadata
        for key in ['timestamp', 'version', 'framework', 'device']:
            if key in data:
                normalized['metadata'][key] = data[key]
        
        return normalized
    
    def _extract_epoch_data(self, history: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract epoch-by-epoch data from history."""
        if not history:
            return []
        
        # Get all metric names
        metric_names = list(history.keys())
        if not metric_names:
            return []
        
        # Determine number of epochs
        num_epochs = len(history[metric_names[0]]) if metric_names else 0
        
        epochs = []
        for epoch_idx in range(num_epochs):
            epoch_data = {'epoch': epoch_idx + 1}
            
            for metric_name in metric_names:
                if epoch_idx < len(history[metric_name]):
                    epoch_data[metric_name] = history[metric_name][epoch_idx]
            
            epochs.append(epoch_data)
        
        return epochs
    
    def _extract_logs_data(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract epoch data from logs list."""
        epochs = []
        
        for i, log_entry in enumerate(logs):
            epoch_data = {'epoch': i + 1}
            epoch_data.update(self._extract_metrics_from_dict(log_entry))
            epochs.append(epoch_data)
        
        return epochs
    
    def _has_epoch_structure(self, data: Dict[str, Any]) -> bool:
        """Check if data has epoch-like structure."""
        epoch_indicators = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'learning_rate']
        return any(indicator in data for indicator in epoch_indicators)
    
    def _extract_metrics_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numeric metrics from dictionary."""
        metrics = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            elif isinstance(value, str) and self._is_numeric_string(value):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass
        
        return metrics
    
    def _get_final_metrics(self, epochs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get final metrics from last epoch."""
        if not epochs:
            return {}
        
        last_epoch = epochs[-1]
        final_metrics = {}
        
        for key, value in last_epoch.items():
            if key != 'epoch' and isinstance(value, (int, float)):
                final_metrics[key] = value
        
        return final_metrics
    
    def _load_csv_training(self, file_path: Path) -> Dict[str, Any]:
        """Load training data from CSV file."""
        csv_data = self._load_csv_file(file_path)
        
        if not csv_data:
            return {}
        
        # Assume each row is an epoch
        epochs = []
        for i, row in enumerate(csv_data):
            epoch_data = {'epoch': i + 1}
            for key, value in row.items():
                if self._is_numeric_string(value):
                    try:
                        epoch_data[key] = float(value)
                    except ValueError:
                        epoch_data[key] = value
                else:
                    epoch_data[key] = value
            epochs.append(epoch_data)
        
        return {'epochs': epochs}
    
    def _parse_training_log(self, file_path: Path) -> Dict[str, Any]:
        """Parse training information from log file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        training_data = {
            'epochs': [],
            'hyperparameters': {},
            'metadata': {}
        }
        
        # Parse epoch information
        epoch_patterns = [
            # Pattern: "Epoch 1/10 - loss: 0.5 - accuracy: 0.8"
            r'Epoch\s+(\d+)(?:/\d+)?\s*[-:](.+)',
            # Pattern: "epoch: 1, loss: 0.5, accuracy: 0.8"
            r'epoch:\s*(\d+)[,\s]+(.+)',
            # Pattern: "[Epoch 1] loss=0.5 accuracy=0.8"
            r'\[Epoch\s+(\d+)\](.+)',
        ]
        
        for pattern in epoch_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for epoch_num, metrics_str in matches:
                epoch_data = {'epoch': int(epoch_num)}
                
                # Extract metrics from the metrics string
                metrics = self._parse_metrics_string(metrics_str)
                epoch_data.update(metrics)
                
                training_data['epochs'].append(epoch_data)
        
        # Parse hyperparameters
        hyperparam_patterns = [
            # Pattern: "learning_rate: 0.001"
            r'(\w+):\s*([\d.e-]+)',
            # Pattern: "lr=0.001"
            r'(\w+)=([\d.e-]+)',
        ]
        
        for pattern in hyperparam_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for param_name, value in matches:
                param_name = param_name.lower()
                if self._is_hyperparameter_name(param_name):
                    try:
                        training_data['hyperparameters'][param_name] = float(value)
                    except ValueError:
                        training_data['hyperparameters'][param_name] = value
        
        # Parse training time
        time_patterns = [
            r'(?:training\s+)?(?:time|duration):\s*([\d.]+)\s*(?:s|sec|seconds?|m|min|minutes?|h|hr|hours?)',
            r'(?:took|elapsed):\s*([\d.]+)\s*(?:s|sec|seconds?|m|min|minutes?|h|hr|hours?)',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    training_data['training_time'] = float(match.group(1))
                    break
                except ValueError:
                    pass
        
        # Look for JSON blocks in logs
        json_blocks = re.findall(r'{[^{}]*}', content)
        for block in json_blocks:
            try:
                json_data = json.loads(block)
                if isinstance(json_data, dict):
                    # Check if it looks like training data
                    if any(key in json_data for key in ['epoch', 'loss', 'accuracy', 'lr']):
                        epoch_data = {'epoch': len(training_data['epochs']) + 1}
                        epoch_data.update(self._extract_metrics_from_dict(json_data))
                        training_data['epochs'].append(epoch_data)
            except json.JSONDecodeError:
                continue
        
        return training_data
    
    def _parse_metrics_string(self, metrics_str: str) -> Dict[str, float]:
        """Parse metrics from a string like 'loss: 0.5 - accuracy: 0.8'."""
        metrics = {}
        
        # Common patterns for metrics
        patterns = [
            r'(\w+):\s*([\d.e-]+)',  # loss: 0.5
            r'(\w+)=([\d.e-]+)',     # loss=0.5
            r'(\w+)\s+([\d.e-]+)',   # loss 0.5
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, metrics_str, re.IGNORECASE)
            for metric_name, value in matches:
                metric_name = metric_name.lower().strip()
                if self._is_valid_metric_name(metric_name):
                    try:
                        metrics[metric_name] = float(value)
                    except ValueError:
                        pass
        
        return metrics
    
    def _is_hyperparameter_name(self, name: str) -> bool:
        """Check if name is likely a hyperparameter."""
        hyperparam_names = {
            'lr', 'learning_rate', 'batch_size', 'epochs', 'weight_decay',
            'momentum', 'beta1', 'beta2', 'epsilon', 'dropout', 'alpha',
            'gamma', 'hidden_size', 'num_layers', 'embedding_dim'
        }
        
        return name.lower() in hyperparam_names
    
    def _is_valid_metric_name(self, name: str) -> bool:
        """Check if name is a valid metric name."""
        metric_names = {
            'loss', 'accuracy', 'acc', 'val_loss', 'val_accuracy', 'val_acc',
            'precision', 'recall', 'f1', 'auc', 'mse', 'mae', 'rmse',
            'learning_rate', 'lr', 'grad_norm', 'perplexity'
        }
        
        name = name.lower().strip()
        return (
            name in metric_names or
            name.startswith('val_') or
            name.startswith('test_') or
            name.startswith('train_')
        )
    
    def _is_numeric_string(self, value: str) -> bool:
        """Check if string represents a number."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


class TensorBoardLogCollector(TrainingLogCollector):
    """Specialized collector for TensorBoard logs."""
    
    def collect(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Collect from TensorBoard event files."""
        # This would require tensorboard or tensorflow to read event files
        # For now, we'll handle JSON exports of TensorBoard data
        return super().collect(source)


class KerasHistoryCollector(TrainingLogCollector):
    """Specialized collector for Keras training history."""
    
    def _normalize_training_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Keras-specific training format."""
        normalized = super()._normalize_training_data(data)
        
        # Handle Keras model.fit() history format
        if 'history' in data and isinstance(data['history'], dict):
            history = data['history']
            epochs = self._extract_epoch_data(history)
            normalized['epochs'] = epochs
            normalized['final_metrics'] = self._get_final_metrics(epochs)
        
        return normalized


class PyTorchLogCollector(TrainingLogCollector):
    """Specialized collector for PyTorch training logs."""
    
    def _normalize_training_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize PyTorch-specific training format."""
        normalized = super()._normalize_training_data(data)
        
        # Handle PyTorch Lightning logs
        if 'trainer' in data and 'logger' in data['trainer']:
            logger_data = data['trainer']['logger']
            if 'metrics' in logger_data:
                normalized['final_metrics'].update(logger_data['metrics'])
        
        return normalized