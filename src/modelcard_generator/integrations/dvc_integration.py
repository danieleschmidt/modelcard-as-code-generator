"""DVC integration for model card generation."""

try:
    import yaml
except ImportError:
    yaml = None
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.models import ModelCard, CardConfig
from ..core.generator import ModelCardGenerator


logger = logging.getLogger(__name__)


class DVCIntegration:
    """Integration with DVC for automatic model card generation."""
    
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize DVC integration."""
        try:
            import dvc.api
            self.dvc = dvc.api
        except ImportError:
            raise ImportError(
                "dvc package is required for DVC integration. "
                "Install with: pip install dvc"
            )
        
        self.repo_path = repo_path or "."
    
    def from_pipeline(self, dvc_file: str = "dvc.yaml", stage: Optional[str] = None, 
                     config: Optional[CardConfig] = None) -> ModelCard:
        """Generate model card from DVC pipeline."""
        try:
            dvc_path = Path(self.repo_path) / dvc_file
            
            if not dvc_path.exists():
                raise FileNotFoundError(f"DVC file not found: {dvc_path}")
            
            with open(dvc_path, 'r') as f:
                pipeline_data = yaml.safe_load(f)
            
            card = ModelCard(config or CardConfig())
            
            # Extract information from pipeline stages
            stages = pipeline_data.get("stages", {})
            
            if stage and stage in stages:
                # Process specific stage
                self._extract_stage_info(card, stage, stages[stage])
            else:
                # Process all stages
                for stage_name, stage_data in stages.items():
                    self._extract_stage_info(card, stage_name, stage_data)
            
            # Extract parameters if available
            self._extract_params(card)
            
            # Extract metrics if available
            self._extract_dvc_metrics(card)
            
            # Add DVC metadata
            card.metadata.update({
                "dvc_pipeline": dvc_file,
                "dvc_repo_path": str(self.repo_path),
                "dvc_stages": list(stages.keys())
            })
            
            logger.info(f"Generated model card from DVC pipeline {dvc_file}")
            return card
            
        except Exception as e:
            logger.error(f"Failed to generate model card from DVC pipeline: {e}")
            raise
    
    def from_model_file(self, model_path: str, config: Optional[CardConfig] = None) -> ModelCard:
        """Generate model card from DVC-tracked model file."""
        try:
            # Get model info from DVC
            model_info = self.dvc.get_url(model_path, repo=self.repo_path)
            
            card = ModelCard(config or CardConfig())
            
            # Extract model details
            model_name = Path(model_path).stem
            card.model_details.name = model_name
            card.model_details.description = f"Model tracked by DVC at {model_path}"
            
            # Try to extract associated metadata
            metadata_paths = [
                f"{model_path}.yaml",
                f"{model_path}.json", 
                f"{Path(model_path).parent}/metadata.yaml",
                f"{Path(model_path).parent}/metadata.json"
            ]
            
            for metadata_path in metadata_paths:
                try:
                    metadata = self._load_metadata_file(metadata_path)
                    if metadata:
                        self._apply_metadata(card, metadata)
                        break
                except:
                    continue
            
            # Add DVC tracking info
            card.metadata.update({
                "dvc_model_path": model_path,
                "dvc_model_url": model_info,
                "dvc_tracked": True
            })
            
            logger.info(f"Generated model card for DVC model {model_path}")
            return card
            
        except Exception as e:
            logger.error(f"Failed to generate model card from DVC model {model_path}: {e}")
            raise
    
    def from_experiment(self, experiment_name: Optional[str] = None, 
                       config: Optional[CardConfig] = None) -> ModelCard:
        """Generate model card from DVC experiment."""
        try:
            # Get experiment data
            experiments_path = Path(self.repo_path) / "dvc.lock"
            
            if experiments_path.exists():
                with open(experiments_path, 'r') as f:
                    lock_data = yaml.safe_load(f)
            else:
                lock_data = {}
            
            card = ModelCard(config or CardConfig())
            
            # Extract experiment information
            if "stages" in lock_data:
                for stage_name, stage_data in lock_data["stages"].items():
                    if experiment_name and experiment_name not in stage_name:
                        continue
                    
                    self._extract_lock_stage_info(card, stage_name, stage_data)
            
            # Extract metrics and parameters
            self._extract_params(card)
            self._extract_dvc_metrics(card)
            
            card.metadata.update({
                "dvc_experiment": experiment_name or "default",
                "dvc_lock_file": str(experiments_path)
            })
            
            logger.info(f"Generated model card from DVC experiment {experiment_name}")
            return card
            
        except Exception as e:
            logger.error(f"Failed to generate model card from DVC experiment: {e}")
            raise
    
    def _extract_stage_info(self, card: ModelCard, stage_name: str, stage_data: Dict[str, Any]) -> None:
        """Extract information from a DVC pipeline stage."""
        # Extract command information
        if "cmd" in stage_data:
            cmd = stage_data["cmd"]
            if isinstance(cmd, list):
                cmd = " ".join(cmd)
            
            # Try to infer framework from command
            if "python" in cmd:
                if "train" in cmd:
                    card.add_section("Training Command", f"```bash\n{cmd}\n```")
                elif "evaluate" in cmd:
                    card.add_section("Evaluation Command", f"```bash\n{cmd}\n```")
            
            # Detect framework from command
            if any(fw in cmd for fw in ["torch", "pytorch"]):
                card.training_details.framework = "PyTorch"
            elif any(fw in cmd for fw in ["tensorflow", "tf", "keras"]):
                card.training_details.framework = "TensorFlow"
            elif "sklearn" in cmd:
                card.training_details.framework = "scikit-learn"
        
        # Extract dependencies (input data/models)
        if "deps" in stage_data:
            deps = stage_data["deps"]
            for dep in deps:
                if dep.endswith((".csv", ".json", ".parquet", ".txt")):
                    card.training_details.training_data.append(dep)
                elif dep.endswith((".py", ".ipynb")):
                    # Training script
                    pass
        
        # Extract outputs (models, metrics)
        if "outs" in stage_data:
            outs = stage_data["outs"]
            for out in outs:
                if out.endswith((".pkl", ".pt", ".h5", ".onnx", ".joblib")):
                    # Model file
                    card.metadata[f"{stage_name}_model_output"] = out
        
        # Extract parameters reference
        if "params" in stage_data:
            params_ref = stage_data["params"]
            if isinstance(params_ref, list):
                for param_file in params_ref:
                    self._extract_params_from_file(card, param_file)
        
        # Extract metrics reference
        if "metrics" in stage_data:
            metrics_ref = stage_data["metrics"]
            if isinstance(metrics_ref, list):
                for metric_file in metrics_ref:
                    self._extract_metrics_from_file(card, metric_file)
    
    def _extract_lock_stage_info(self, card: ModelCard, stage_name: str, stage_data: Dict[str, Any]) -> None:
        """Extract information from DVC lock file stage."""
        # Extract dependency checksums for reproducibility
        if "deps" in stage_data:
            deps_info = []
            for dep in stage_data["deps"]:
                if isinstance(dep, dict) and "path" in dep:
                    deps_info.append(f"{dep['path']} (md5: {dep.get('md5', 'unknown')})")
            
            if deps_info:
                card.add_section("Data Dependencies", "\n".join(f"- {dep}" for dep in deps_info))
        
        # Extract output checksums
        if "outs" in stage_data:
            outs_info = []
            for out in stage_data["outs"]:
                if isinstance(out, dict) and "path" in out:
                    outs_info.append(f"{out['path']} (md5: {out.get('md5', 'unknown')})")
            
            if outs_info:
                card.add_section("Model Outputs", "\n".join(f"- {out}" for out in outs_info))
    
    def _extract_params(self, card: ModelCard) -> None:
        """Extract parameters from params.yaml."""
        params_path = Path(self.repo_path) / "params.yaml"
        if params_path.exists():
            self._extract_params_from_file(card, str(params_path))
    
    def _extract_params_from_file(self, card: ModelCard, params_file: str) -> None:
        """Extract parameters from a specific file."""
        try:
            params_path = Path(self.repo_path) / params_file
            
            if params_path.exists():
                with open(params_path, 'r') as f:
                    params = yaml.safe_load(f)
                
                if isinstance(params, dict):
                    # Extract training hyperparameters
                    train_params = params.get("train", params.get("training", {}))
                    if train_params:
                        card.training_details.hyperparameters.update(train_params)
                    
                    # Extract model parameters
                    model_params = params.get("model", {})
                    if model_params:
                        if "architecture" in model_params:
                            card.training_details.model_architecture = model_params["architecture"]
                        if "framework" in model_params:
                            card.training_details.framework = model_params["framework"]
                    
                    # Extract general parameters
                    for key, value in params.items():
                        if key not in ["train", "training", "model"] and not isinstance(value, dict):
                            card.training_details.hyperparameters[key] = value
            
        except Exception as e:
            logger.warning(f"Failed to extract parameters from {params_file}: {e}")
    
    def _extract_dvc_metrics(self, card: ModelCard) -> None:
        """Extract metrics from DVC metrics files."""
        metrics_path = Path(self.repo_path) / "metrics.json"
        if metrics_path.exists():
            self._extract_metrics_from_file(card, str(metrics_path))
        
        # Also check for other common metric files
        metric_files = [
            "metrics.yaml",
            "eval/metrics.json", 
            "evaluation/metrics.json",
            "results/metrics.json"
        ]
        
        for metric_file in metric_files:
            full_path = Path(self.repo_path) / metric_file
            if full_path.exists():
                self._extract_metrics_from_file(card, str(full_path))
    
    def _extract_metrics_from_file(self, card: ModelCard, metrics_file: str) -> None:
        """Extract metrics from a specific file."""
        try:
            metrics_path = Path(self.repo_path) / metrics_file
            
            if metrics_path.exists():
                if metrics_path.suffix == '.json':
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                else:  # yaml
                    with open(metrics_path, 'r') as f:
                        metrics = yaml.safe_load(f)
                
                self._flatten_and_add_metrics(card, metrics)
            
        except Exception as e:
            logger.warning(f"Failed to extract metrics from {metrics_file}: {e}")
    
    def _flatten_and_add_metrics(self, card: ModelCard, metrics: Dict[str, Any], prefix: str = "") -> None:
        """Recursively flatten and add metrics to card."""
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively handle nested metrics
                self._flatten_and_add_metrics(card, value, full_key)
            elif isinstance(value, (int, float)):
                card.add_metric(full_key, float(value))
    
    def _load_metadata_file(self, metadata_path: str) -> Optional[Dict[str, Any]]:
        """Load metadata from file."""
        try:
            full_path = Path(self.repo_path) / metadata_path
            
            if not full_path.exists():
                return None
            
            if full_path.suffix == '.json':
                with open(full_path, 'r') as f:
                    return json.load(f)
            else:  # yaml
                with open(full_path, 'r') as f:
                    return yaml.safe_load(f)
        
        except Exception:
            return None
    
    def _apply_metadata(self, card: ModelCard, metadata: Dict[str, Any]) -> None:
        """Apply metadata to model card."""
        # Apply model details
        if "name" in metadata:
            card.model_details.name = metadata["name"]
        if "version" in metadata:
            card.model_details.version = metadata["version"]
        if "description" in metadata:
            card.model_details.description = metadata["description"]
        if "authors" in metadata:
            card.model_details.authors = metadata["authors"]
        if "license" in metadata:
            card.model_details.license = metadata["license"]
        
        # Apply training details
        if "framework" in metadata:
            card.training_details.framework = metadata["framework"]
        if "architecture" in metadata:
            card.training_details.model_architecture = metadata["architecture"]
        
        # Apply metrics
        if "metrics" in metadata:
            self._flatten_and_add_metrics(card, metadata["metrics"])
    
    def save_model_card_to_dvc(self, card: ModelCard, output_path: str = "MODEL_CARD.md") -> None:
        """Save model card and track it with DVC."""
        try:
            # Save model card
            full_path = Path(self.repo_path) / output_path
            card.save(str(full_path))
            
            # Add to DVC tracking
            import subprocess
            
            result = subprocess.run(
                ["dvc", "add", str(full_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Model card saved and tracked by DVC: {output_path}")
            else:
                logger.warning(f"Failed to add model card to DVC: {result.stderr}")
            
        except Exception as e:
            logger.error(f"Failed to save model card to DVC: {e}")
            raise