"""Simple configuration for model card generator."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SecurityConfig:
    """Security configuration."""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    scan_content: bool = False  # Simplified: no external scanner needed


@dataclass
class ValidationConfig:
    """Validation configuration."""
    min_completeness_score: float = 0.8
    required_sections: List[str] = field(default_factory=lambda: [
        "model_details", "intended_use", "evaluation_results"
    ])
    enforce_compliance: bool = False


@dataclass
class DriftConfig:
    """Drift detection configuration."""
    enabled: bool = True
    default_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.02,
        "precision": 0.02,
        "recall": 0.02,
        "f1": 0.02,
        "loss": 0.05,
    })
    snapshot_dir: str = ".modelcard/snapshots"


@dataclass
class ModelCardConfig:
    """Top-level configuration."""
    security: SecurityConfig = field(default_factory=SecurityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    output_dir: str = "model_cards"
    default_format: str = "huggingface"
    auto_validate: bool = True
    auto_scan_security: bool = False


_config = ModelCardConfig()


def get_config() -> ModelCardConfig:
    return _config


def load_config(config_path: Optional[str] = None) -> ModelCardConfig:
    """Load config from YAML/JSON file if provided, else return defaults."""
    global _config
    if config_path is None:
        return _config

    import json
    from pathlib import Path
    path = Path(config_path)
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)

    # Simple flat merge of top-level keys
    cfg = ModelCardConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    _config = cfg
    return _config
