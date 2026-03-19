"""Tests for modelcard-as-code-generator."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from modelcard_generator import (
    CardConfig,
    CardFormat,
    DriftDetector,
    ModelCard,
    ModelCardGenerator,
    Validator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BERT_CONFIG = {
    "model_details": {
        "name": "bert-test",
        "version": "1.0.0",
        "description": "Test BERT model.",
        "authors": ["Alice"],
        "license": "MIT",
        "base_model": "bert-base-uncased",
        "language": ["en"],
        "tags": ["text-classification"],
        "datasets": ["test-dataset"],
    },
    "intended_use": "Unit testing.",
    "training_details": {
        "framework": "PyTorch",
        "training_data": ["test-dataset"],
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
        },
    },
    "evaluation_results": [
        {"name": "accuracy", "value": 0.924, "dataset": "test-split"},
        {"name": "f1", "value": 0.918, "confidence_interval": [0.910, 0.926]},
        {"name": "precision", "value": 0.921},
        {"name": "recall", "value": 0.915},
    ],
    "ethical_considerations": {
        "bias_risks": ["Potential demographic bias."],
        "bias_mitigation": ["Regular audits."],
    },
    "limitations": {
        "known_limitations": ["English only.", "Short texts may underperform."],
        "out_of_scope_uses": ["Medical decisions."],
    },
}


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def yaml_config(tmp_dir) -> Path:
    path = tmp_dir / "model.yaml"
    path.write_text(yaml.dump(BERT_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def json_config(tmp_dir) -> Path:
    path = tmp_dir / "model.json"
    path.write_text(json.dumps(BERT_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def eval_json(tmp_dir) -> Path:
    """Eval results that differ significantly from BERT_CONFIG values (>2% drift)."""
    path = tmp_dir / "eval.json"
    path.write_text(json.dumps({
        "accuracy": 0.890,   # -3.4% from 0.924
        "f1": 0.885,         # -3.3% from 0.918
        "precision": 0.888,  # -3.3% from 0.921
        "recall": 0.882,     # -3.3% from 0.915
    }), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Core model tests
# ---------------------------------------------------------------------------

class TestModelCard:
    def test_create_empty(self):
        card = ModelCard()
        assert card.model_details.name == ""

    def test_add_metric(self):
        card = ModelCard()
        card.add_metric("accuracy", 0.95)
        assert len(card.evaluation_results) == 1
        assert card.evaluation_results[0].name == "accuracy"
        assert card.evaluation_results[0].value == 0.95

    def test_add_limitation(self):
        card = ModelCard()
        card.add_limitation("Limitation A")
        assert "Limitation A" in card.limitations.known_limitations

    def test_render_markdown_returns_string(self):
        card = ModelCard()
        card.model_details.name = "my-model"
        md = card.render("markdown")
        assert "# my-model" in md

    def test_render_html_returns_html(self):
        card = ModelCard()
        card.model_details.name = "html-test"
        html = card.render("html")
        assert "<html>" in html
        assert "html-test" in html

    def test_render_json_roundtrip(self):
        card = ModelCard()
        card.model_details.name = "json-test"
        card.model_details.version = "2.0"
        card.add_metric("accuracy", 0.9)
        j = card.render("json")
        data = json.loads(j)
        assert data["model_details"]["name"] == "json-test"
        assert data["evaluation_results"][0]["name"] == "accuracy"

    def test_render_unsupported_format(self):
        card = ModelCard()
        with pytest.raises(ValueError):
            card.render("xml")

    def test_save_and_load_markdown(self, tmp_dir):
        card = ModelCard()
        card.model_details.name = "save-test"
        card.model_details.version = "1.0"
        card.model_details.authors = ["Bob"]
        card.model_details.license = "MIT"
        card.add_metric("f1", 0.88)
        card.add_limitation("Only English.")
        card.intended_use = "Testing save/load."

        path = tmp_dir / "card.md"
        card.save(str(path))
        assert path.exists()

        loaded = ModelCard.load(str(path))
        assert loaded.model_details.name == "save-test"

    def test_metric_with_confidence_interval(self):
        card = ModelCard()
        card.add_metric("f1", 0.91, confidence_interval=[0.905, 0.915])
        md = card.render("markdown")
        assert "CI" in md

    def test_custom_section(self):
        card = ModelCard()
        card.model_details.name = "custom-test"
        card.add_section("Carbon Footprint", "1.2 kg CO₂e")
        md = card.render("markdown")
        assert "Carbon Footprint" in md
        assert "1.2 kg CO₂e" in md


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestModelCardGenerator:
    def test_from_yaml_config(self, yaml_config):
        gen = ModelCardGenerator()
        card = gen.from_config(yaml_config)
        assert card.model_details.name == "bert-test"
        assert card.model_details.version == "1.0.0"
        assert "Alice" in card.model_details.authors
        assert card.model_details.license == "MIT"

    def test_from_json_config(self, json_config):
        gen = ModelCardGenerator()
        card = gen.from_config(json_config)
        assert card.model_details.name == "bert-test"

    def test_eval_metrics_loaded(self, yaml_config):
        gen = ModelCardGenerator()
        card = gen.from_config(yaml_config)
        metric_names = [m.name for m in card.evaluation_results]
        assert "accuracy" in metric_names
        assert "f1" in metric_names

    def test_confidence_interval_preserved(self, yaml_config):
        gen = ModelCardGenerator()
        card = gen.from_config(yaml_config)
        f1_metric = next(m for m in card.evaluation_results if m.name == "f1")
        assert f1_metric.confidence_interval == [0.910, 0.926]

    def test_limitations_loaded(self, yaml_config):
        gen = ModelCardGenerator()
        card = gen.from_config(yaml_config)
        assert len(card.limitations.known_limitations) >= 2
        assert "English only." in card.limitations.known_limitations

    def test_ethical_considerations_loaded(self, yaml_config):
        gen = ModelCardGenerator()
        card = gen.from_config(yaml_config)
        assert len(card.ethical_considerations.bias_risks) >= 1

    def test_training_details_loaded(self, yaml_config):
        gen = ModelCardGenerator()
        card = gen.from_config(yaml_config)
        assert card.training_details.framework == "PyTorch"
        assert card.training_details.hyperparameters.get("epochs") == 3

    def test_auto_populate_fills_defaults(self, tmp_dir):
        minimal = {"model_details": {"name": "minimal-model"}}
        path = tmp_dir / "minimal.yaml"
        path.write_text(yaml.dump(minimal), encoding="utf-8")
        gen = ModelCardGenerator()
        card = gen.from_config(path)
        assert card.intended_use != ""
        assert len(card.limitations.known_limitations) > 0

    def test_no_auto_populate(self, tmp_dir):
        minimal = {"model_details": {"name": "no-fill"}}
        path = tmp_dir / "minimal.yaml"
        path.write_text(yaml.dump(minimal), encoding="utf-8")
        config = CardConfig(auto_populate=False)
        gen = ModelCardGenerator(config=config)
        card = gen.from_config(path)
        assert card.intended_use == ""
        assert len(card.limitations.known_limitations) == 0

    def test_generate_with_eval_file(self, tmp_dir, eval_json):
        gen = ModelCardGenerator()
        card = gen.generate(eval_results=eval_json, model_name="eval-only-model")
        assert card.model_details.name == "eval-only-model"
        metric_names = [m.name for m in card.evaluation_results]
        assert "accuracy" in metric_names

    def test_missing_config_file(self):
        gen = ModelCardGenerator()
        with pytest.raises(Exception):
            gen.from_config("/nonexistent/path/model.yaml")

    def test_card_format_google(self, yaml_config):
        config = CardConfig(format=CardFormat.GOOGLE)
        gen = ModelCardGenerator(config=config)
        card = gen.from_config(yaml_config)
        assert card.config.format == CardFormat.GOOGLE

    def test_full_markdown_output(self, yaml_config):
        gen = ModelCardGenerator()
        card = gen.from_config(yaml_config)
        md = card.render("markdown")
        # All major sections present
        assert "## Model Details" in md
        assert "## Intended Use" in md
        assert "## Training Details" in md
        assert "## Evaluation Results" in md
        assert "## Limitations" in md


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def _make_complete_card(self) -> ModelCard:
        card = ModelCard()
        card.model_details.name = "complete-model"
        card.model_details.version = "1.0"
        card.model_details.description = "A complete model for testing purposes."
        card.model_details.authors = ["Dev"]
        card.model_details.license = "MIT"
        card.model_details.language = ["en"]
        card.intended_use = "Testing validation."
        card.training_details.training_data = ["test-dataset"]
        card.add_metric("accuracy", 0.9)
        card.add_limitation("Some limitation.")
        return card

    def test_complete_card_passes(self):
        v = Validator()
        card = self._make_complete_card()
        result = v.validate_schema(card, "huggingface")
        assert result.is_valid

    def test_empty_card_has_errors(self):
        v = Validator()
        card = ModelCard()
        result = v.validate_schema(card, "huggingface")
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_missing_name_is_error(self):
        v = Validator()
        card = self._make_complete_card()
        card.model_details.name = ""
        result = v.validate_schema(card, "huggingface")
        assert not result.is_valid

    def test_missing_eval_results_is_error(self):
        v = Validator()
        card = self._make_complete_card()
        card.evaluation_results.clear()
        result = v.validate_schema(card, "huggingface")
        assert not result.is_valid

    def test_score_between_0_and_1(self):
        v = Validator()
        card = ModelCard()
        card.model_details.name = "partial"
        result = v.validate_schema(card, "huggingface")
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Drift detector tests
# ---------------------------------------------------------------------------

class TestDriftDetector:
    def _make_card_with_metrics(self, accuracy=0.94, f1=0.93) -> ModelCard:
        card = ModelCard()
        card.model_details.name = "drift-test"
        card.add_metric("accuracy", accuracy)
        card.add_metric("f1", f1)
        card.add_metric("precision", 0.92)
        card.add_metric("recall", 0.94)
        return card

    def test_no_drift_when_metrics_stable(self):
        old = self._make_card_with_metrics(accuracy=0.94, f1=0.93)
        new = self._make_card_with_metrics(accuracy=0.941, f1=0.929)  # within threshold
        detector = DriftDetector()
        report = detector.check(old, new_card=new)
        assert not report.has_drift

    def test_drift_detected_when_metric_drops(self):
        old = self._make_card_with_metrics(accuracy=0.94, f1=0.93)
        new = self._make_card_with_metrics(accuracy=0.91, f1=0.90)  # >2% drop
        detector = DriftDetector()
        report = detector.check(old, new_card=new)
        assert report.has_drift

    def test_drift_report_has_changes(self):
        old = self._make_card_with_metrics(accuracy=0.94)
        new = self._make_card_with_metrics(accuracy=0.90)
        detector = DriftDetector()
        report = detector.check(old, new_card=new)
        assert len(report.changes) > 0

    def test_drift_check_with_dict(self):
        card = self._make_card_with_metrics(accuracy=0.94, f1=0.93)
        new_eval = {"accuracy": 0.88, "f1": 0.86}
        detector = DriftDetector()
        report = detector.check(card, new_eval_results=new_eval)
        assert report.has_drift

    def test_drift_check_with_json_file(self, tmp_path):
        card = self._make_card_with_metrics(accuracy=0.94, f1=0.93)
        eval_file = tmp_path / "eval.json"
        eval_file.write_text(json.dumps({"accuracy": 0.88, "f1": 0.86}))
        detector = DriftDetector()
        report = detector.check(card, new_eval_results=str(eval_file))
        assert report.has_drift

    def test_significant_changes_list(self):
        old = self._make_card_with_metrics(accuracy=0.94, f1=0.93)
        new = self._make_card_with_metrics(accuracy=0.90, f1=0.89)
        detector = DriftDetector()
        report = detector.check(old, new_card=new)
        sig = report.significant_changes
        assert len(sig) > 0
        assert all(c.is_significant for c in sig)

    def test_custom_thresholds(self):
        old = self._make_card_with_metrics(accuracy=0.94)
        new = self._make_card_with_metrics(accuracy=0.93)  # 1% drop
        # Default threshold=0.02 → NOT significant
        detector_default = DriftDetector()
        report_default = detector_default.check(old, new_card=new)
        accuracy_change = next(c for c in report_default.changes if c.metric_name == "accuracy")
        assert not accuracy_change.is_significant

        # Custom threshold=0.005 → IS significant
        detector_strict = DriftDetector(default_thresholds={"accuracy": 0.005})
        report_strict = detector_strict.check(old, new_card=new)
        accuracy_change_strict = next(
            c for c in report_strict.changes if c.metric_name == "accuracy"
        )
        assert accuracy_change_strict.is_significant

    def test_no_source_raises(self):
        card = self._make_card_with_metrics()
        detector = DriftDetector()
        with pytest.raises(ValueError):
            detector.check(card)

    def test_snapshot_save_and_load(self, tmp_path):
        card = self._make_card_with_metrics()
        snap_file = tmp_path / "snapshots.json"
        detector = DriftDetector()
        detector.save_snapshot(card, snap_file)
        assert snap_file.exists()
        data = json.loads(snap_file.read_text())
        assert len(data) == 1
        assert data[0]["model_name"] == "drift-test"
        assert "accuracy" in data[0]["metrics"]


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_generate_cli(self, yaml_config, tmp_path):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        output_path = str(tmp_path / "card.md")
        result = runner.invoke(cli, ["generate", str(yaml_config), "-o", output_path])
        assert result.exit_code == 0, result.output
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert "bert-test" in content

    def test_generate_cli_stdout(self, yaml_config):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["generate", str(yaml_config)])
        assert result.exit_code == 0
        assert "bert-test" in result.output

    def test_generate_cli_with_html(self, yaml_config, tmp_path):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        md_path = str(tmp_path / "card.md")
        html_path = str(tmp_path / "card.html")
        result = runner.invoke(cli, ["generate", str(yaml_config), "-o", md_path, "--html", html_path])
        assert result.exit_code == 0
        assert Path(html_path).exists()
        content = Path(html_path).read_text()
        assert "<html>" in content

    def test_validate_cli_valid_card(self, yaml_config, tmp_path):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        card_path = str(tmp_path / "card.md")
        runner.invoke(cli, ["generate", str(yaml_config), "-o", card_path])

        result = runner.invoke(cli, ["validate", card_path])
        assert result.exit_code == 0

    def test_check_cli_drift_detected(self, yaml_config, tmp_path, eval_json):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        card_path = str(tmp_path / "card.md")
        runner.invoke(cli, ["generate", str(yaml_config), "-o", card_path])

        result = runner.invoke(cli, ["check", card_path, "--new-eval", str(eval_json)])
        # eval_json has metrics that differ >2% → drift → exit 1
        assert result.exit_code == 1
        assert "drift" in result.output.lower() or "🔴" in result.output

    def test_check_cli_no_drift(self, yaml_config, tmp_path):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        card_path = str(tmp_path / "card.md")
        runner.invoke(cli, ["generate", str(yaml_config), "-o", card_path])

        # Metrics nearly identical to the card's own values (within 2% threshold)
        stable_eval = tmp_path / "stable_eval.json"
        stable_eval.write_text(json.dumps({
            "accuracy": 0.923,
            "f1": 0.917,
            "precision": 0.920,
            "recall": 0.914,
        }))
        result = runner.invoke(cli, ["check", card_path, "--new-eval", str(stable_eval)])
        assert result.exit_code == 0

    def test_schema_cli(self):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["schema"])
        assert result.exit_code == 0
        assert "model_details" in result.output

    def test_check_cli_json_output(self, yaml_config, tmp_path, eval_json):
        from click.testing import CliRunner
        from modelcard_generator.cli.main import cli

        runner = CliRunner()
        card_path = str(tmp_path / "card.md")
        runner.invoke(cli, ["generate", str(yaml_config), "-o", card_path])

        result = runner.invoke(cli, ["check", card_path, "--new-eval", str(eval_json), "--json-output"])
        data = json.loads(result.output)
        assert "has_drift" in data
        assert "changes" in data
