"""CLI for modelcard-as-code-generator."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from ..core.drift_detector import DriftDetector
from ..core.generator import ModelCardGenerator
from ..core.models import CardConfig, CardFormat
from ..core.validator import Validator


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="2.0.0", prog_name="mcg")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
def cli(verbose: bool) -> None:
    """mcg — Model Card Generator.

    Generate, validate, and diff ML model cards from YAML/JSON configs.

    \b
    Examples:
      mcg generate mymodel.yaml -o MODEL_CARD.md
      mcg generate mymodel.yaml --html output.html
      mcg validate MODEL_CARD.md
      mcg check MODEL_CARD.md --new-eval eval_results.json
    """
    import logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(),
              help="Output markdown file (default: stdout).")
@click.option("--html", type=click.Path(),
              help="Also write an HTML version to this path.")
@click.option("--format", "card_format",
              type=click.Choice(["huggingface", "google", "eu_cra"]),
              default="huggingface", show_default=True,
              help="Model card standard to follow.")
@click.option("--no-auto-populate", is_flag=True, default=False,
              help="Disable auto-filling of missing fields.")
def generate(
    config_file: str,
    output: Optional[str],
    html: Optional[str],
    card_format: str,
    no_auto_populate: bool,
) -> None:
    """Generate a model card from CONFIG_FILE (YAML or JSON).

    \b
    CONFIG_FILE schema (YAML example):
      model_details:
        name: my-bert
        version: "1.0"
        authors: [Alice, Bob]
        license: Apache-2.0
        base_model: bert-base-uncased
        language: [en]
      intended_use: "Sentiment classification on product reviews."
      training_details:
        framework: PyTorch
        training_data: [amazon-reviews-v2]
        hyperparameters:
          learning_rate: 2e-5
          batch_size: 32
          epochs: 3
      evaluation_results:
        - name: accuracy
          value: 0.924
          dataset: amazon-reviews-test
        - name: f1
          value: 0.918
      limitations:
        known_limitations:
          - "Only evaluated on English text."
    """
    fmt_map = {
        "huggingface": CardFormat.HUGGINGFACE,
        "google": CardFormat.GOOGLE,
        "eu_cra": CardFormat.EU_CRA,
    }
    config = CardConfig(
        format=fmt_map[card_format],
        auto_populate=not no_auto_populate,
    )
    generator = ModelCardGenerator(config=config)

    try:
        card = generator.from_config(config_file)
    except Exception as e:
        click.echo(f"❌ Failed to generate model card: {e}", err=True)
        sys.exit(1)

    markdown = card.render("markdown")

    if output:
        Path(output).write_text(markdown, encoding="utf-8")
        click.echo(f"✅ Saved markdown: {output}")
    else:
        click.echo(markdown)

    if html:
        html_content = card.render("html")
        Path(html).write_text(html_content, encoding="utf-8")
        click.echo(f"✅ Saved HTML: {html}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("card_file", type=click.Path(exists=True))
@click.option("--format", "card_format",
              type=click.Choice(["huggingface", "google", "eu_cra"]),
              default="huggingface", show_default=True)
@click.option("--strict", is_flag=True, default=False,
              help="Exit 1 on warnings as well as errors.")
def validate(card_file: str, card_format: str, strict: bool) -> None:
    """Validate an existing model card for completeness.

    Checks required sections are present and well-formed.
    Exits 0 on success, 1 on validation failure.
    """
    from ..core.models import ModelCard
    try:
        card = ModelCard.load(card_file)
    except Exception as e:
        click.echo(f"❌ Could not load card: {e}", err=True)
        sys.exit(1)

    validator = Validator()
    result = validator.validate_schema(card, card_format)

    if result.is_valid:
        click.echo(f"✅ Valid  score={result.score:.2f}")
    else:
        click.echo(f"⚠️  Score: {result.score:.2f}")

    for issue in result.issues:
        icon = "❌" if issue.severity.value == "error" else "⚠️ "
        click.echo(f"  {icon} [{issue.severity.value}] {issue.message}")
        if issue.suggestion:
            click.echo(f"       → {issue.suggestion}")

    if result.missing_sections:
        click.echo(f"\nMissing sections: {', '.join(result.missing_sections)}")

    failed = not result.is_valid or (strict and result.warnings)
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# check  (drift / diff)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("card_file", type=click.Path(exists=True))
@click.option("--new-eval", type=click.Path(exists=True),
              help="New evaluation results file (JSON/YAML) to compare against.")
@click.option("--new-card", type=click.Path(exists=True),
              help="New model card file to compare against.")
@click.option("--threshold", type=float, default=None,
              help="Override default drift threshold (applied to all metrics).")
@click.option("--json-output", is_flag=True, default=False,
              help="Print results as JSON.")
def check(
    card_file: str,
    new_eval: Optional[str],
    new_card: Optional[str],
    threshold: Optional[float],
    json_output: bool,
) -> None:
    """Check an existing model card for metric drift.

    Compares the card's recorded metrics against new evaluation results
    or a new card, and reports any significant changes.

    \b
    Examples:
      mcg check MODEL_CARD.md --new-eval eval_results.json
      mcg check MODEL_CARD.md --new-card NEW_MODEL_CARD.md
    """
    from ..core.models import ModelCard

    if not new_eval and not new_card:
        click.echo("❌ Provide --new-eval or --new-card", err=True)
        sys.exit(1)

    try:
        card = ModelCard.load(card_file)
    except Exception as e:
        click.echo(f"❌ Could not load card: {e}", err=True)
        sys.exit(1)

    thresholds = None
    if threshold is not None:
        # apply to all known metrics
        thresholds = {m: threshold for m in [
            "accuracy", "precision", "recall", "f1", "f1_score",
            "auc", "roc_auc", "loss", "val_loss",
        ]}

    detector = DriftDetector(default_thresholds=thresholds)

    try:
        if new_card:
            other = ModelCard.load(new_card)
            report = detector.check(card, new_card=other, thresholds=thresholds)
        else:
            report = detector.check(card, new_eval_results=new_eval, thresholds=thresholds)
    except Exception as e:
        click.echo(f"❌ Drift check failed: {e}", err=True)
        sys.exit(1)

    if json_output:
        output = {
            "has_drift": report.has_drift,
            "changes": [
                {
                    "metric": c.metric_name,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "delta": c.delta,
                    "is_significant": c.is_significant,
                }
                for c in report.changes
            ],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        if not report.changes:
            click.echo("✅ No metrics to compare (card has no recorded metrics).")
            return

        if not report.has_drift:
            click.echo("✅ No significant drift detected.")
        else:
            click.echo("⚠️  Significant drift detected!\n")

        for change in report.changes:
            icon = "🔴" if change.is_significant else "✅"
            direction = "▲" if change.delta > 0 else "▼"
            click.echo(
                f"  {icon} {change.metric_name:20s} "
                f"{change.old_value:.4f} → {change.new_value:.4f} "
                f"({direction}{abs(change.delta):.4f})"
            )

    sys.exit(1 if report.has_drift else 0)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--format", "fmt", type=click.Choice(["yaml", "json"]),
              default="yaml", show_default=True)
def schema(fmt: str) -> None:
    """Print the config file schema / template to stdout."""
    template = """\
# Model Card config for mcg generate
# All fields except model_details.name are optional.

model_details:
  name: my-model-name           # required
  version: "1.0.0"
  description: "One-line description of this model."
  authors:
    - Your Name
  license: Apache-2.0
  base_model: bert-base-uncased # parent model, if fine-tuned
  library_name: transformers
  language:
    - en
  tags:
    - text-classification
  datasets:
    - my-dataset-v1

intended_use: >
  Describe what the model is for and who the target users are.
  Include primary intended use cases.

training_details:
  framework: PyTorch
  model_architecture: BERT
  training_data:
    - my-dataset-v1
  preprocessing: "Tokenised with bert-base-uncased tokeniser, max_length=512."
  hardware: "1x NVIDIA A100 40GB"
  training_time: "4 hours"
  hyperparameters:
    learning_rate: 2.0e-5
    batch_size: 32
    epochs: 3
    warmup_steps: 500

evaluation_results:
  - name: accuracy
    value: 0.924
    dataset: my-dataset-test
  - name: f1
    value: 0.918
    dataset: my-dataset-test
    confidence_interval: [0.910, 0.926]
  - name: precision
    value: 0.921
  - name: recall
    value: 0.915

ethical_considerations:
  bias_risks:
    - "Performance may vary across demographic groups."
  bias_mitigation:
    - "Evaluate on stratified demographic slices before deployment."
  sensitive_attributes:
    - age
    - gender
  fairness_metrics:
    demographic_parity: 0.95

limitations:
  known_limitations:
    - "Only evaluated on English text."
    - "Performance may degrade on informal/social media language."
  out_of_scope_uses:
    - "Not for use in high-stakes medical or legal decisions."
  recommendations:
    - "Run bias audits before production deployment."

# Optional custom sections (freeform markdown)
custom_sections:
  Carbon Footprint: |
    Training emitted approximately 0.5 kg CO₂e (estimated via CodeCarbon).
"""
    if fmt == "json":
        import yaml as _yaml
        data = _yaml.safe_load(template)
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(template)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
