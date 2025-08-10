"""Configuration management for model card generator."""

import os
import json

try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

from .exceptions import ConfigurationError, ValidationError
from .security import sanitizer
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None
    structured: bool = False
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_scanning: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.md', '.txt', '.json', '.yaml', '.yml', '.csv', '.tsv', 
        '.xml', '.html', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.svg'
    ])
    blocked_extensions: List[str] = field(default_factory=lambda: [
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', 
        '.jar', '.app', '.deb', '.rpm', '.dmg', '.pkg', '.sh', '.ps1'
    ])
    scan_content: bool = True
    allow_external_urls: bool = False


@dataclass
class ValidationConfig:
    """Validation configuration."""
    min_completeness_score: float = 0.8
    required_sections: List[str] = field(default_factory=lambda: [
        "model_details", "intended_use", "evaluation_results"
    ])
    enforce_compliance: bool = False
    compliance_standards: List[str] = field(default_factory=lambda: ["huggingface"])


@dataclass
class IntegrationConfig:
    """Integration configuration."""
    wandb_enabled: bool = False
    wandb_api_key: Optional[str] = None
    mlflow_enabled: bool = False
    mlflow_tracking_uri: Optional[str] = None
    huggingface_hub_enabled: bool = False
    huggingface_token: Optional[str] = None
    dvc_enabled: bool = False


@dataclass
class DriftConfig:
    """Drift detection configuration."""
    enabled: bool = True
    default_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.02,
        "precision": 0.02, 
        "recall": 0.02,
        "f1": 0.02,
        "loss": 0.05
    })
    save_snapshots: bool = True
    snapshot_dir: str = ".modelcard/snapshots"
    max_snapshots: int = 50


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    cache_dir: str = ".modelcard/cache"
    ttl_seconds: int = 3600  # 1 hour
    max_size_mb: int = 500


@dataclass
class ModelCardConfig:
    """Complete model card generator configuration."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # General settings
    output_dir: str = "model_cards"
    default_format: str = "huggingface"
    auto_validate: bool = True
    auto_scan_security: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate logging level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level.upper() not in valid_levels:
            raise ConfigurationError("logging.level", f"Invalid level: {self.logging.level}")
        
        # Validate security settings
        if self.security.max_file_size <= 0:
            raise ConfigurationError("security.max_file_size", "Must be positive")
        
        # Validate validation settings
        if not 0 <= self.validation.min_completeness_score <= 1:
            raise ConfigurationError("validation.min_completeness_score", "Must be between 0 and 1")
        
        # Validate drift thresholds
        for metric, threshold in self.drift.default_thresholds.items():
            if threshold <= 0:
                raise ConfigurationError(f"drift.default_thresholds.{metric}", "Must be positive")
        
        # Validate format
        valid_formats = ["huggingface", "google", "eu_cra"]
        if self.default_format not in valid_formats:
            raise ConfigurationError("default_format", f"Invalid format: {self.default_format}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCardConfig':
        """Create from dictionary."""
        # Handle nested dataclasses
        if 'logging' in data and isinstance(data['logging'], dict):
            data['logging'] = LoggingConfig(**data['logging'])
        if 'security' in data and isinstance(data['security'], dict):
            data['security'] = SecurityConfig(**data['security'])
        if 'validation' in data and isinstance(data['validation'], dict):
            data['validation'] = ValidationConfig(**data['validation'])
        if 'integrations' in data and isinstance(data['integrations'], dict):
            data['integrations'] = IntegrationConfig(**data['integrations'])
        if 'drift' in data and isinstance(data['drift'], dict):
            data['drift'] = DriftConfig(**data['drift'])
        if 'cache' in data and isinstance(data['cache'], dict):
            data['cache'] = CacheConfig(**data['cache'])
        
        return cls(**data)


class ConfigManager:
    """Configuration manager for model card generator."""
    
    DEFAULT_CONFIG_PATHS = [
        "modelcard.config.json",
        "modelcard.config.yaml",
        "modelcard.config.yml",
        ".modelcard/config.json",
        ".modelcard/config.yaml",
        ".modelcard/config.yml",
        "~/.modelcard/config.json",
        "~/.modelcard/config.yaml",
        "~/.modelcard/config.yml",
    ]
    
    ENV_PREFIX = "MODELCARD_"
    
    def __init__(self):
        self._config: Optional[ModelCardConfig] = None
        self._config_path: Optional[str] = None
        self._env_overrides: Dict[str, Any] = {}
    
    @property
    def config(self) -> ModelCardConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self, config_path: Optional[str] = None) -> ModelCardConfig:
        """Load configuration from file and environment."""
        logger.info("Loading configuration")
        
        # Start with default config
        config_data = {}
        
        # Load from file
        if config_path:
            config_data = self._load_config_file(config_path)
            self._config_path = config_path
        else:
            # Try default paths
            for path in self.DEFAULT_CONFIG_PATHS:
                expanded_path = Path(path).expanduser().resolve()
                if expanded_path.exists():
                    config_data = self._load_config_file(str(expanded_path))
                    self._config_path = str(expanded_path)
                    logger.info(f"Loaded configuration from {expanded_path}")
                    break
        
        # Apply environment overrides
        env_overrides = self._load_env_overrides()
        config_data = self._merge_config(config_data, env_overrides)
        
        # Create and validate config
        try:
            config = ModelCardConfig.from_dict(config_data)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ConfigurationError("config_validation", str(e))
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(config_path).expanduser().resolve()
        
        if not path.exists():
            raise ConfigurationError("config_file", f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    if yaml is None:
                        raise ConfigurationError("yaml_support", "YAML support not available - install PyYAML")
                    data = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ConfigurationError("config_format", f"Unsupported format: {path.suffix}")
            
            # Sanitize loaded data
            if isinstance(data, dict):
                data = sanitizer.validate_json(data)
            
            return data or {}
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError("config_parse", f"Failed to parse {path}: {e}")
        except Exception as e:
            raise ConfigurationError("config_load", f"Failed to load {path}: {e}")
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                config_key = key[len(self.ENV_PREFIX):].lower()
                
                # Handle nested keys (e.g., MODELCARD_LOGGING_LEVEL)
                key_parts = config_key.split('_')
                
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value)
                
                # Set nested value
                self._set_nested_value(overrides, key_parts, converted_value)
        
        return overrides
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON values
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String value
        return sanitizer.sanitize_string(value)
    
    def _set_nested_value(self, config: Dict[str, Any], key_parts: List[str], value: Any) -> None:
        """Set nested configuration value."""
        current = config
        
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[key_parts[-1]] = value
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config_path: Optional[str] = None, config: Optional[ModelCardConfig] = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        if config_path is None:
            config_path = self._config_path or "modelcard.config.yaml"
        
        path = Path(config_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = config.to_dict()
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    if yaml is None:
                        raise ConfigurationError("yaml_support", "YAML support not available - install PyYAML")
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise ConfigurationError("config_format", f"Unsupported format: {path.suffix}")
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            raise ConfigurationError("config_save", f"Failed to save configuration: {e}")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get configuration setting by key path."""
        key_parts = key.split('.')
        current = self.config.to_dict()
        
        try:
            for part in key_parts:
                current = current[part]
            return current
        except KeyError:
            return default
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set configuration setting by key path."""
        if self._config is None:
            self._config = self.load_config()
        
        config_dict = self._config.to_dict()
        key_parts = key.split('.')
        
        current = config_dict
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[key_parts[-1]] = value
        
        # Recreate config object
        self._config = ModelCardConfig.from_dict(config_dict)
    
    @contextmanager
    def temporary_config(self, **overrides):
        """Context manager for temporary configuration changes."""
        original_config = self._config
        
        if original_config:
            # Apply overrides
            config_dict = original_config.to_dict()
            for key, value in overrides.items():
                self._set_nested_value(config_dict, key.split('.'), value)
            self._config = ModelCardConfig.from_dict(config_dict)
        
        try:
            yield self.config
        finally:
            self._config = original_config
    
    def validate_config(self, config: Optional[ModelCardConfig] = None) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        if config is None:
            config = self.config
        
        issues = []
        warnings = []
        
        try:
            config._validate()
        except ConfigurationError as e:
            issues.append(f"Configuration error in {e.config_key}: {e.message}")
        
        # Check file permissions and paths
        paths_to_check = [
            config.logging.file,
            config.drift.snapshot_dir,
            config.cache.cache_dir,
            config.output_dir
        ]
        
        for path in paths_to_check:
            if path:
                try:
                    path_obj = Path(path).expanduser().resolve()
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    warnings.append(f"Path access issue for {path}: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


# Global configuration manager
config_manager = ConfigManager()


def get_config() -> ModelCardConfig:
    """Get current configuration."""
    return config_manager.config


def load_config(config_path: Optional[str] = None) -> ModelCardConfig:
    """Load configuration from file."""
    return config_manager.load_config(config_path)


def get_setting(key: str, default: Any = None) -> Any:
    """Get configuration setting."""
    return config_manager.get_setting(key, default)


def set_setting(key: str, value: Any) -> None:
    """Set configuration setting."""
    config_manager.set_setting(key, value)