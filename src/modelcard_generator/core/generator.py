"""Core model card generation engine."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

from .exceptions import DataSourceError, ModelCardError
from .logging_config import get_logger
from .models import CardConfig, ModelCard

logger = get_logger(__name__)


class DataSourceParser:
    """Parse various data sources for model card generation."""

    def parse_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse a JSON file into a dict."""
        path = Path(file_path)
        if not path.exists():
            raise DataSourceError(str(path), "File not found")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise DataSourceError(str(path), f"Invalid JSON: {e}")

    def parse_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse a YAML file into a dict."""
        if yaml is None:
            raise DataSourceError(str(file_path), "PyYAML not installed — run: pip install pyyaml")
        path = Path(file_path)
        if not path.exists():
            raise DataSourceError(str(path), "File not found")
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise DataSourceError(str(path), f"Invalid YAML: {e}")

    def parse_csv(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parse a CSV file into a list of row dicts."""
        path = Path(file_path)
        if not path.exists():
            raise DataSourceError(str(path), "File not found")
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                if i >= 10_000:
                    break
                rows.append(dict(row))
        return rows

    def parse(self, file_path: Union[str, Path]) -> Any:
        """Parse any supported file type."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self.parse_json(path)
        elif suffix in (".yaml", ".yml"):
            return self.parse_yaml(path)
        elif suffix in (".csv", ".tsv"):
            return self.parse_csv(path)
        else:
            raise DataSourceError(str(path), f"Unsupported extension: {suffix}")


class ModelCardGenerator:
    """Generate model cards from YAML/JSON config and optional eval results."""

    def __init__(self, config: Optional[CardConfig] = None):
        self.config = config or CardConfig()
        self.parser = DataSourceParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_config(self, config_path: Union[str, Path]) -> ModelCard:
        """
        Primary entry point: load a YAML/JSON model config file and produce
        a fully populated ModelCard.

        The config file should follow the schema in examples/bert_sentiment.yaml.
        """
        path = Path(config_path)
        data = self.parser.parse(path)
        return self._build_from_dict(data)

    def generate(
        self,
        eval_results: Optional[Union[str, Path, Dict[str, Any]]] = None,
        training_history: Optional[Union[str, Path]] = None,
        dataset_info: Optional[Union[str, Path, Dict[str, Any]]] = None,
        model_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelCard:
        """
        Compose a ModelCard from individual source files/dicts.

        Keyword arguments are merged as top-level overrides (model_name,
        intended_use, authors, license, etc.).
        """
        card = ModelCard(config=self.config)

        if model_config:
            self._process_model_config(card, model_config)
        if eval_results:
            self._process_evaluation_results(card, eval_results)
        if training_history:
            self._process_training_history(card, training_history)
        if dataset_info:
            self._process_dataset_info(card, dataset_info)

        self._apply_overrides(card, kwargs)

        if self.config.auto_populate:
            self._auto_populate(card)

        return card

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_from_dict(self, data: Dict[str, Any]) -> ModelCard:
        """Build a ModelCard from a full config dict."""
        card = ModelCard(config=self.config)

        # --- model_details ---
        md = data.get("model_details", {})
        card.model_details.name = md.get("name", data.get("name", ""))
        card.model_details.version = md.get("version", data.get("version", ""))
        card.model_details.description = md.get("description", data.get("description", ""))
        card.model_details.license = md.get("license", data.get("license", ""))
        card.model_details.base_model = md.get("base_model", data.get("base_model", ""))
        card.model_details.library_name = md.get("library_name", data.get("library_name", ""))

        authors = md.get("authors", data.get("authors", []))
        card.model_details.authors = authors if isinstance(authors, list) else [authors]

        lang = md.get("language", data.get("language", []))
        card.model_details.language = lang if isinstance(lang, list) else [lang]

        tags = md.get("tags", data.get("tags", []))
        card.model_details.tags = tags if isinstance(tags, list) else [tags]

        datasets = md.get("datasets", data.get("datasets", []))
        card.model_details.datasets = datasets if isinstance(datasets, list) else [datasets]

        # --- intended_use ---
        card.intended_use = data.get("intended_use", "")

        # --- training_details ---
        td = data.get("training_details", {})
        card.training_details.framework = td.get("framework", "")
        card.training_details.model_architecture = td.get("model_architecture", td.get("architecture", ""))
        train_data = td.get("training_data", [])
        card.training_details.training_data = train_data if isinstance(train_data, list) else [train_data]
        card.training_details.preprocessing = td.get("preprocessing", "")
        card.training_details.hardware = td.get("hardware", "")
        card.training_details.training_time = td.get("training_time", "")
        card.training_details.hyperparameters = td.get("hyperparameters", {})

        # --- evaluation_results ---
        for entry in data.get("evaluation_results", []):
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("metric")
                value = entry.get("value") or entry.get("score")
                if name is not None and value is not None:
                    card.add_metric(
                        name,
                        float(value),
                        confidence_interval=entry.get("confidence_interval"),
                        dataset=entry.get("dataset"),
                        slice_name=entry.get("slice"),
                    )

        # --- ethical_considerations ---
        ec = data.get("ethical_considerations", {})
        card.ethical_considerations.bias_risks = ec.get("bias_risks", [])
        card.ethical_considerations.bias_mitigation = ec.get("bias_mitigation", [])
        card.ethical_considerations.fairness_constraints = ec.get("fairness_constraints", [])
        card.ethical_considerations.sensitive_attributes = ec.get("sensitive_attributes", [])
        card.ethical_considerations.fairness_metrics = ec.get("fairness_metrics", {})

        # --- limitations ---
        lim = data.get("limitations", {})
        if isinstance(lim, list):
            card.limitations.known_limitations = lim
        else:
            card.limitations.known_limitations = lim.get("known_limitations", [])
            card.limitations.out_of_scope_uses = lim.get("out_of_scope_uses", [])
            card.limitations.sensitive_use_cases = lim.get("sensitive_use_cases", [])
            card.limitations.recommendations = lim.get("recommendations", [])

        # --- custom sections ---
        for section_name, content in data.get("custom_sections", {}).items():
            card.add_section(section_name, content)

        if self.config.auto_populate:
            self._auto_populate(card)

        return card

    def _process_evaluation_results(
        self, card: ModelCard, eval_results: Union[str, Path, Dict[str, Any]]
    ) -> None:
        if isinstance(eval_results, (str, Path)):
            data = self.parser.parse(eval_results)
        else:
            data = eval_results

        # Support both list-of-dicts and flat key:value formats
        if isinstance(data, list):
            for entry in data:
                name = entry.get("name") or entry.get("metric")
                value = entry.get("value") or entry.get("score")
                if name and value is not None:
                    card.add_metric(name, float(value))
        elif isinstance(data, dict):
            metric_keys = {
                "accuracy", "precision", "recall", "f1", "f1_score", "f1_macro", "f1_micro",
                "auc", "roc_auc", "loss", "val_loss", "mse", "rmse", "mae", "r2",
                "bleu", "rouge", "perplexity",
            }
            # Check for eval_results sub-key
            if "eval_results" in data:
                self._process_evaluation_results(card, data["eval_results"])
                return
            for key, val in data.items():
                if isinstance(val, (int, float)):
                    card.add_metric(key, float(val))
                elif isinstance(val, dict) and "value" in val:
                    card.add_metric(key, float(val["value"]),
                                    confidence_interval=val.get("confidence_interval"))
            if not card.model_details.name and "model_name" in data:
                card.model_details.name = data["model_name"]

    def _process_training_history(
        self, card: ModelCard, training_history: Union[str, Path]
    ) -> None:
        """Very lightweight training-log parser."""
        path = Path(training_history)
        if not path.exists():
            return
        hyperparams: Dict[str, Any] = {}
        last_metrics: Dict[str, float] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "learning_rate:" in line or "batch_size:" in line or "epochs:" in line:
                try:
                    k, v = line.split(":", 1)
                    val_s = v.strip()
                    hyperparams[k.strip()] = float(val_s) if "." in val_s else int(val_s)
                except (ValueError, KeyError):
                    pass
            for metric in ("loss", "accuracy", "f1", "precision", "recall"):
                if f"{metric}:" in line:
                    try:
                        val = float(line.split(f"{metric}:")[1].split(",")[0].strip())
                        last_metrics[metric] = val
                    except (ValueError, IndexError):
                        pass
        card.training_details.hyperparameters.update(hyperparams)
        for k, v in last_metrics.items():
            card.add_metric(f"final_{k}", v)

    def _process_dataset_info(
        self, card: ModelCard, dataset_info: Union[str, Path, Dict[str, Any]]
    ) -> None:
        if isinstance(dataset_info, (str, Path)):
            data = self.parser.parse(dataset_info)
        else:
            data = dataset_info

        datasets = data.get("datasets", [])
        if isinstance(datasets, list):
            card.model_details.datasets.extend(datasets)
        elif datasets:
            card.model_details.datasets.append(str(datasets))

        train_data = data.get("training_data", [])
        if isinstance(train_data, list):
            card.training_details.training_data.extend(train_data)
        elif train_data:
            card.training_details.training_data.append(str(train_data))

        if "preprocessing" in data:
            card.training_details.preprocessing = data["preprocessing"]

        if "bias_analysis" in data:
            ba = data["bias_analysis"]
            card.ethical_considerations.bias_risks.extend(ba.get("bias_risks", []))
            card.ethical_considerations.fairness_metrics.update(ba.get("fairness_metrics", {}))

    def _process_model_config(
        self, card: ModelCard, model_config: Union[str, Path, Dict[str, Any]]
    ) -> None:
        if isinstance(model_config, (str, Path)):
            data = self.parser.parse(model_config)
        else:
            data = dict(model_config)
        # Merge directly via _build_from_dict logic for model_details fields
        self._build_from_dict.__func__  # just use dict merge
        if "name" in data:
            card.model_details.name = data["name"]
        if "version" in data:
            card.model_details.version = data["version"]
        if "description" in data:
            card.model_details.description = data["description"]
        if "license" in data:
            card.model_details.license = data["license"]
        if "authors" in data:
            a = data["authors"]
            card.model_details.authors = a if isinstance(a, list) else [a]
        if "base_model" in data:
            card.model_details.base_model = data["base_model"]
        if "framework" in data:
            card.training_details.framework = data["framework"]
        if "architecture" in data:
            card.training_details.model_architecture = data["architecture"]
        if "hyperparameters" in data:
            card.training_details.hyperparameters.update(data["hyperparameters"])

    def _apply_overrides(self, card: ModelCard, overrides: Dict[str, Any]) -> None:
        mapping = {
            "model_name": lambda v: setattr(card.model_details, "name", v),
            "model_version": lambda v: setattr(card.model_details, "version", v),
            "description": lambda v: setattr(card.model_details, "description", v),
            "intended_use": lambda v: setattr(card, "intended_use", v),
            "license": lambda v: setattr(card.model_details, "license", v),
            "authors": lambda v: setattr(card.model_details, "authors",
                                         v if isinstance(v, list) else [v]),
        }
        for key, fn in mapping.items():
            if key in overrides:
                fn(overrides[key])
        for key, val in overrides.items():
            if key not in mapping:
                card.metadata[key] = val

    def _auto_populate(self, card: ModelCard) -> None:
        if not card.model_details.name:
            card.model_details.name = "untitled-model"
        if not card.model_details.version:
            card.model_details.version = "1.0.0"
        if not card.intended_use:
            card.intended_use = (
                "This model is intended for research and educational purposes. "
                "Please review limitations before deploying in production."
            )
        if not card.limitations.known_limitations:
            card.limitations.known_limitations = [
                "Performance may degrade on out-of-distribution data.",
                "Requires further validation for production deployment.",
                "May reflect biases present in training data.",
            ]
        if (
            self.config.include_ethical_considerations
            and not card.ethical_considerations.bias_risks
        ):
            card.ethical_considerations.bias_risks = [
                "Potential for demographic bias in predictions.",
                "May perpetuate biases in training data.",
            ]
            card.ethical_considerations.bias_mitigation = [
                "Conduct regular bias audits.",
                "Evaluate on diverse demographic slices before deployment.",
            ]
