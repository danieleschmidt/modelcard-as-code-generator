"""
Configuration collector.

Collects and normalizes model configuration, hyperparameters,
and metadata from various configuration formats.
"""

import json
import yaml
import toml
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import logging
import re

from .base import DataCollector


class ConfigCollector(DataCollector):
    """Collector for model configuration and metadata."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.py']
    
    def collect(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collect configuration data from source.
        
        Supports:
        - JSON configuration files
        - YAML configuration files
        - TOML configuration files
        - Python configuration files (basic parsing)
        - Direct dictionary data
        """
        if isinstance(source, dict):
            return self._normalize_config_data(source)
        
        file_path = self._normalize_path(source)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        if not self.validate_source(file_path):
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
        
        # Load data based on file type
        if file_path.suffix.lower() in ['.json']:
            data = self._load_json_file(file_path)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            data = self._load_yaml_file(file_path)
        elif file_path.suffix.lower() in ['.toml']:
            data = self._load_toml_file(file_path)
        elif file_path.suffix.lower() in ['.cfg', '.ini']:
            data = self._load_ini_file(file_path)
        elif file_path.suffix.lower() in ['.py']:
            data = self._parse_python_config(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self._normalize_config_data(data)
    
    def validate_source(self, source: Union[str, Path, Dict[str, Any]]) -> bool:
        """Validate configuration source."""
        if isinstance(source, dict):
            return True
        
        file_path = self._normalize_path(source)
        return (
            file_path.exists() and 
            file_path.suffix.lower() in self.supported_formats
        )
    
    def _normalize_config_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration data to standard format."""
        normalized = {
            'model': {},
            'training': {},
            'data': {},
            'hardware': {},
            'framework': {},
            'metadata': {}
        }
        
        # Categorize configuration fields
        for key, value in data.items():
            category = self._categorize_config_field(key)
            if category in normalized:
                normalized[category][key] = value
            else:
                # Put unrecognized fields in metadata
                normalized['metadata'][key] = value
        
        # Extract specific model information
        self._extract_model_info(data, normalized)
        self._extract_training_info(data, normalized)
        self._extract_framework_info(data, normalized)
        
        return normalized
    
    def _categorize_config_field(self, key: str) -> str:
        """Categorize configuration field into appropriate section."""
        key_lower = key.lower()
        
        # Model-related fields
        model_keywords = [
            'model', 'architecture', 'network', 'layers', 'hidden', 'embedding',
            'vocab', 'max_length', 'num_classes', 'dropout', 'activation'
        ]
        if any(keyword in key_lower for keyword in model_keywords):
            return 'model'
        
        # Training-related fields
        training_keywords = [
            'train', 'epoch', 'batch', 'learning', 'lr', 'optimizer', 'loss',
            'metric', 'scheduler', 'gradient', 'weight_decay', 'momentum'
        ]
        if any(keyword in key_lower for keyword in training_keywords):
            return 'training'
        
        # Data-related fields
        data_keywords = [
            'data', 'dataset', 'input', 'output', 'preprocess', 'augment',
            'tokenizer', 'vocab', 'sequence'
        ]
        if any(keyword in key_lower for keyword in data_keywords):
            return 'data'
        
        # Hardware-related fields
        hardware_keywords = [
            'gpu', 'cpu', 'device', 'memory', 'precision', 'distributed',
            'parallel', 'workers'
        ]
        if any(keyword in key_lower for keyword in hardware_keywords):
            return 'hardware'
        
        # Framework-related fields
        framework_keywords = [
            'framework', 'library', 'version', 'backend', 'engine'
        ]
        if any(keyword in key_lower for keyword in framework_keywords):
            return 'framework'
        
        return 'metadata'
    
    def _extract_model_info(self, data: Dict[str, Any], normalized: Dict[str, Any]) -> None:
        """Extract model-specific information."""
        # Common model fields
        model_fields = [
            'model_name', 'model_type', 'architecture', 'num_parameters',
            'hidden_size', 'num_layers', 'num_attention_heads', 'vocab_size',
            'max_position_embeddings', 'num_classes', 'dropout_rate'
        ]
        
        for field in model_fields:
            if field in data:
                normalized['model'][field] = data[field]
        
        # Extract from nested model config
        if 'model' in data and isinstance(data['model'], dict):
            normalized['model'].update(data['model'])
        
        if 'config' in data and isinstance(data['config'], dict):
            model_config = data['config']
            for key, value in model_config.items():
                if any(keyword in key.lower() for keyword in ['model', 'arch', 'network']):
                    normalized['model'][key] = value
    
    def _extract_training_info(self, data: Dict[str, Any], normalized: Dict[str, Any]) -> None:
        """Extract training-specific information."""
        # Common training fields
        training_fields = [
            'learning_rate', 'batch_size', 'num_epochs', 'optimizer',
            'loss_function', 'scheduler', 'weight_decay', 'momentum',
            'gradient_clipping', 'warmup_steps', 'eval_steps'
        ]
        
        for field in training_fields:
            if field in data:
                normalized['training'][field] = data[field]
        
        # Extract from nested training config
        if 'training' in data and isinstance(data['training'], dict):
            normalized['training'].update(data['training'])
        
        if 'hyperparameters' in data and isinstance(data['hyperparameters'], dict):
            normalized['training'].update(data['hyperparameters'])
    
    def _extract_framework_info(self, data: Dict[str, Any], normalized: Dict[str, Any]) -> None:
        """Extract framework and environment information."""
        framework_fields = [
            'framework', 'framework_version', 'python_version',
            'cuda_version', 'transformers_version', 'torch_version',
            'tensorflow_version'
        ]
        
        for field in framework_fields:
            if field in data:
                normalized['framework'][field] = data[field]
        
        # Extract from nested environment info
        if 'environment' in data and isinstance(data['environment'], dict):
            normalized['framework'].update(data['environment'])
    
    def _load_toml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from TOML file."""
        try:
            import toml
            with open(file_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except ImportError:
            raise ImportError("toml package required to load TOML files")
    
    def _load_ini_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from INI/CFG file."""
        import configparser
        
        config = configparser.ConfigParser()
        config.read(file_path, encoding='utf-8')
        
        # Convert to nested dictionary
        data = {}
        for section_name in config.sections():
            section = config[section_name]
            data[section_name] = {}
            
            for key, value in section.items():
                # Try to parse as different types
                data[section_name][key] = self._parse_ini_value(value)
        
        return data
    
    def _parse_ini_value(self, value: str) -> Any:
        """Parse INI value to appropriate type."""
        # Boolean values
        if value.lower() in ['true', 'yes', '1', 'on']:
            return True
        elif value.lower() in ['false', 'no', '0', 'off']:
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # List values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # String value
        return value
    
    def _parse_python_config(self, file_path: Path) -> Dict[str, Any]:
        """Parse Python configuration file (basic variable extraction)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        config = {}
        
        # Extract simple variable assignments
        # Pattern: VARIABLE = value
        patterns = [
            r'^(\w+)\s*=\s*(["\'])(.*?)\2\s*$',  # String assignments
            r'^(\w+)\s*=\s*(\d+\.?\d*)\s*$',     # Numeric assignments
            r'^(\w+)\s*=\s*(True|False|None)\s*$', # Boolean/None assignments
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    if len(match.groups()) == 3:  # String pattern
                        var_value = match.group(3)
                    else:  # Other patterns
                        var_value = match.group(2)
                        # Convert to appropriate type
                        if var_value.lower() == 'true':
                            var_value = True
                        elif var_value.lower() == 'false':
                            var_value = False
                        elif var_value.lower() == 'none':
                            var_value = None
                        else:
                            try:
                                var_value = float(var_value) if '.' in var_value else int(var_value)
                            except ValueError:
                                pass
                    
                    config[var_name] = var_value
                    break
        
        # Look for dictionary definitions
        dict_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
        dict_matches = re.findall(dict_pattern, content, re.MULTILINE | re.DOTALL)
        
        for var_name, dict_content in dict_matches:
            try:
                # Simple dictionary parsing
                dict_items = {}
                for item in dict_content.split(','):
                    item = item.strip()
                    if ':' in item:
                        key, value = item.split(':', 1)
                        key = key.strip().strip('\'"')
                        value = value.strip().strip('\'"')
                        dict_items[key] = value
                
                config[var_name] = dict_items
            except Exception:
                # If parsing fails, store as string
                config[var_name] = dict_content.strip()
        
        return config


class HuggingFaceConfigCollector(ConfigCollector):
    """Specialized collector for Hugging Face model configurations."""
    
    def _normalize_config_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Hugging Face config format."""
        normalized = super()._normalize_config_data(data)
        
        # Handle Hugging Face specific fields
        hf_model_fields = [
            'model_type', 'architectures', 'hidden_size', 'num_hidden_layers',
            'num_attention_heads', 'intermediate_size', 'vocab_size',
            'max_position_embeddings', 'type_vocab_size', 'initializer_range',
            'layer_norm_eps', 'hidden_dropout_prob', 'attention_probs_dropout_prob'
        ]
        
        for field in hf_model_fields:
            if field in data:
                normalized['model'][field] = data[field]
        
        return normalized


class TransformersConfigCollector(ConfigCollector):
    """Specialized collector for transformer model configurations."""
    
    def _normalize_config_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize transformer-specific config format."""
        normalized = super()._normalize_config_data(data)
        
        # Handle transformer-specific fields
        transformer_fields = [
            'd_model', 'n_heads', 'n_layers', 'dff', 'dropout_rate',
            'activation', 'positional_encoding', 'max_seq_length'
        ]
        
        for field in transformer_fields:
            if field in data:
                normalized['model'][field] = data[field]
        
        return normalized