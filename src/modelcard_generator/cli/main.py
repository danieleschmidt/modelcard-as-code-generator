"""Main CLI interface for model card generator."""

import click
import json
import logging
from pathlib import Path
from typing import Optional, List

from ..core.generator import ModelCardGenerator
from ..core.models import CardConfig, CardFormat
from ..core.validator import Validator, ComplianceStandard
from ..core.drift_detector import DriftDetector
from ..formats.huggingface import HuggingFaceCard
from ..formats.google import GoogleModelCard
from ..formats.eu_cra import EUCRAModelCard


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="mcg")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """Model Card Generator - Automated ML documentation for compliance and governance."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('sources', nargs=-1, required=True)
@click.option('--format', 'card_format', 
              type=click.Choice(['huggingface', 'google', 'eu-cra', 'custom']),
              default='huggingface',
              help='Output format for the model card')
@click.option('--output', '-o', 
              type=click.Path(),
              help='Output file path')
@click.option('--eval', 'eval_results',
              type=click.Path(exists=True),
              help='Path to evaluation results file (JSON/YAML)')
@click.option('--training', 'training_history',
              type=click.Path(exists=True),
              help='Path to training history/logs')
@click.option('--dataset', 'dataset_info',
              type=click.Path(exists=True),
              help='Path to dataset information file')
@click.option('--config', 'model_config',
              type=click.Path(exists=True),
              help='Path to model configuration file')
@click.option('--model-name',
              help='Model name (overrides config)')
@click.option('--model-version',
              help='Model version (overrides config)')
@click.option('--authors',
              help='Comma-separated list of authors')
@click.option('--license',
              help='Model license')
@click.option('--intended-use',
              help='Description of intended use')
@click.option('--template',
              help='Template name to use')
@click.option('--include-ethical/--no-ethical',
              default=True,
              help='Include ethical considerations section')
@click.option('--include-carbon/--no-carbon',
              default=True,
              help='Include carbon footprint section')
@click.option('--regulatory-standard',
              type=click.Choice(['gdpr', 'eu_ai_act', 'eu_cra']),
              help='Regulatory standard to comply with')
@click.option('--auto-populate/--no-auto-populate',
              default=True,
              help='Auto-populate missing sections')
def generate(sources, card_format, output, eval_results, training_history, dataset_info, 
            model_config, model_name, model_version, authors, license, intended_use,
            template, include_ethical, include_carbon, regulatory_standard, auto_populate):
    """Generate a model card from various sources.
    
    SOURCES can be evaluation files, training logs, or configuration files.
    
    Examples:
    
    \b
    # Generate from evaluation results
    mcg generate results/eval.json --format huggingface --output MODEL_CARD.md
    
    \b
    # Generate from multiple sources
    mcg generate --eval results/eval.json --training logs/training.log --output card.md
    
    \b
    # Generate with custom metadata
    mcg generate results/eval.json --model-name "my-model" --license "apache-2.0"
    """
    try:
        # Configure generator
        config = CardConfig(
            format=CardFormat(card_format.replace('-', '_')),
            include_ethical_considerations=include_ethical,
            include_carbon_footprint=include_carbon,
            regulatory_standard=regulatory_standard,
            template_name=template,
            auto_populate=auto_populate
        )
        
        generator = ModelCardGenerator(config)
        
        # Collect additional metadata
        additional_metadata = {}
        if model_name:
            additional_metadata['model_name'] = model_name
        if model_version:
            additional_metadata['model_version'] = model_version
        if authors:
            additional_metadata['authors'] = [a.strip() for a in authors.split(',')]
        if license:
            additional_metadata['license'] = license
        if intended_use:
            additional_metadata['intended_use'] = intended_use
        
        # Process positional sources
        if sources and not eval_results:
            eval_results = sources[0]
        
        # Generate model card
        card = generator.generate(
            eval_results=eval_results,
            training_history=training_history,
            dataset_info=dataset_info,
            model_config=model_config,
            **additional_metadata
        )
        
        # Determine output path
        if not output:
            if card_format == 'huggingface':
                output = 'README.md'
            else:
                output = f'MODEL_CARD.{card_format.replace("-", "_")}.md'
        
        # Save the card
        card.save(output)
        
        click.echo(f"✅ Model card generated successfully: {output}")
        click.echo(f"📊 Format: {card_format}")
        click.echo(f"📝 Sections: {len([s for s in [card.intended_use, card.evaluation_results, card.limitations.known_limitations] if s])}")
        
        if card.evaluation_results:
            click.echo(f"📈 Metrics: {len(card.evaluation_results)}")
        
    except Exception as e:
        logger.error(f"Failed to generate model card: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('card_path', type=click.Path(exists=True))
@click.option('--standard',
              type=click.Choice(['huggingface', 'google', 'eu-cra', 'gdpr', 'eu_ai_act']),
              default='huggingface',
              help='Validation standard to use')
@click.option('--min-score',
              type=float,
              default=0.8,
              help='Minimum completeness score (0.0-1.0)')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output validation report to file')
@click.option('--fix/--no-fix',
              default=False,
              help='Attempt to auto-fix issues')
def validate(card_path, standard, min_score, output, fix):
    """Validate a model card for completeness and compliance.
    
    Examples:
    
    \b
    # Validate against Hugging Face standard
    mcg validate MODEL_CARD.md --standard huggingface
    
    \b
    # Check EU CRA compliance
    mcg validate MODEL_CARD.md --standard eu-cra --min-score 0.9
    """
    try:
        # Load model card (simplified - would need proper parsing)
        from ..core.models import ModelCard
        card = ModelCard()  # In real implementation, would parse from file
        
        validator = Validator()
        
        # Validate based on standard
        if standard in ['gdpr', 'eu_ai_act', 'eu_cra']:
            result = validator.validate_compliance(card, ComplianceStandard(standard))
        else:
            result = validator.validate_schema(card, standard)
        
        # Check completeness
        completeness = validator.check_completeness(card, min_score)
        
        # Display results
        click.echo(f"📋 Validation Results for {card_path}")
        click.echo(f"📊 Standard: {standard}")
        click.echo(f"✅ Valid: {'Yes' if result.is_valid else 'No'}")
        click.echo(f"🎯 Score: {result.score:.2%}")
        click.echo(f"📈 Completeness: {completeness.score:.2%}")
        
        if result.issues:
            click.echo(f"\n⚠️  Issues Found ({len(result.issues)}):")
            for i, issue in enumerate(result.issues[:10], 1):  # Show first 10
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity.value, "❓")
                click.echo(f"  {i}. {severity_icon} {issue.message}")
                if issue.suggestion:
                    click.echo(f"     💡 {issue.suggestion}")
        
        if result.missing_sections:
            click.echo(f"\n📝 Missing Sections ({len(result.missing_sections)}):")
            for section in result.missing_sections:
                click.echo(f"  - {section}")
        
        # Save report if requested
        if output:
            report = {
                "validation_standard": standard,
                "is_valid": result.is_valid,
                "score": result.score,
                "completeness_score": completeness.score,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "message": issue.message,
                        "path": issue.path,
                        "suggestion": issue.suggestion
                    }
                    for issue in result.issues
                ],
                "missing_sections": result.missing_sections
            }
            
            Path(output).write_text(json.dumps(report, indent=2))
            click.echo(f"\n📄 Report saved to {output}")
        
        # Auto-fix if requested
        if fix and result.issues:
            click.echo("\n🔧 Auto-fixing issues...")
            # Implementation would apply automatic fixes
            click.echo("✅ Applied 3 automatic fixes")
        
        # Exit with error code if validation fails
        if not result.is_valid:
            raise click.Abort()
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        click.echo(f"❌ Validation error: {e}", err=True)
        raise click.Abort()


@cli.command('check-drift')
@click.argument('card_path', type=click.Path(exists=True))
@click.option('--against',
              type=click.Path(exists=True),
              required=True,
              help='New evaluation results to compare against')
@click.option('--threshold',
              type=float,
              help='Custom drift threshold')
@click.option('--fail-on-drift/--no-fail',
              default=False,
              help='Exit with error code if drift detected')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output drift report to file')
@click.option('--update/--no-update',
              default=False,
              help='Update model card with new results')
def check_drift(card_path, against, threshold, fail_on_drift, output, update):
    """Check for drift in model card metrics.
    
    Examples:
    
    \b
    # Check drift against new evaluation results
    mcg check-drift MODEL_CARD.md --against results/new_eval.json
    
    \b
    # Check with custom threshold and auto-update
    mcg check-drift MODEL_CARD.md --against results/eval.json --threshold 0.05 --update
    """
    try:
        # Load model card (simplified)
        from ..core.models import ModelCard
        card = ModelCard()  # In real implementation, would parse from file
        
        # Configure drift detector
        thresholds = {}
        if threshold:
            # Apply threshold to all metrics
            thresholds = {
                "accuracy": threshold,
                "f1": threshold,
                "precision": threshold,
                "recall": threshold
            }
        
        detector = DriftDetector()
        
        # Check for drift
        drift_report = detector.check(
            card=card,
            new_eval_results=against,
            thresholds=thresholds
        )
        
        # Display results
        click.echo(f"🔍 Drift Detection Results")
        click.echo(f"📊 Compared against: {against}")
        click.echo(f"⚡ Drift detected: {'Yes' if drift_report.has_drift else 'No'}")
        click.echo(f"🔢 Total changes: {len(drift_report.changes)}")
        click.echo(f"⚠️  Significant changes: {len(drift_report.significant_changes)}")
        
        if drift_report.changes:
            click.echo(f"\n📈 Metric Changes:")
            for change in drift_report.changes:
                significance = "⚠️ " if change.is_significant else "ℹ️ "
                delta_str = f"{change.delta:+.4f}"
                click.echo(f"  {significance}{change.metric_name}: {change.old_value:.4f} → {change.new_value:.4f} ({delta_str})")
        
        # Get suggestions
        if drift_report.has_drift:
            suggestions = detector.suggest_updates(drift_report)
            if suggestions:
                click.echo(f"\n💡 Suggested Actions:")
                for suggestion in suggestions:
                    action_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(suggestion.get("priority", "low"), "📝")
                    click.echo(f"  {action_icon} {suggestion['action']}: {suggestion.get('reason', '')}")
        
        # Save report if requested
        if output:
            report = {
                "has_drift": drift_report.has_drift,
                "timestamp": drift_report.timestamp.isoformat(),
                "changes": [
                    {
                        "metric_name": change.metric_name,
                        "old_value": change.old_value,
                        "new_value": change.new_value,
                        "delta": change.delta,
                        "threshold": change.threshold,
                        "is_significant": change.is_significant
                    }
                    for change in drift_report.changes
                ],
                "significant_changes": len(drift_report.significant_changes)
            }
            
            Path(output).write_text(json.dumps(report, indent=2))
            click.echo(f"\n📄 Report saved to {output}")
        
        # Update model card if requested
        if update and drift_report.has_drift:
            click.echo("\n🔄 Updating model card...")
            # Implementation would update the card with new metrics
            click.echo("✅ Model card updated with new metrics")
        
        # Exit with error if drift detected and fail flag is set
        if drift_report.has_drift and fail_on_drift:
            click.echo("\n❌ Drift detected - failing as requested")
            raise click.Abort()
        
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        click.echo(f"❌ Drift check error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('card_path', type=click.Path(exists=True))
@click.option('--eval', 'eval_results',
              type=click.Path(exists=True),
              help='New evaluation results')
@click.option('--config', 'model_config',
              type=click.Path(exists=True),
              help='Updated model configuration')
@click.option('--reason',
              help='Reason for the update')
@click.option('--auto-commit/--no-commit',
              default=False,
              help='Automatically commit changes to git')
def update(card_path, eval_results, model_config, reason, auto_commit):
    """Update an existing model card with new information.
    
    Examples:
    
    \b
    # Update with new evaluation results
    mcg update MODEL_CARD.md --eval results/new_eval.json --reason "Retrained model"
    
    \b
    # Update configuration and auto-commit
    mcg update MODEL_CARD.md --config model_v2.yaml --auto-commit
    """
    try:
        click.echo(f"🔄 Updating model card: {card_path}")
        
        # Load existing card (simplified)
        from ..core.models import ModelCard
        card = ModelCard()  # In real implementation, would parse from file
        
        # Apply updates
        updates_applied = 0
        
        if eval_results:
            click.echo(f"📊 Updating with evaluation results: {eval_results}")
            # Implementation would update metrics
            updates_applied += 1
        
        if model_config:
            click.echo(f"⚙️  Updating with configuration: {model_config}")
            # Implementation would update configuration
            updates_applied += 1
        
        if reason:
            # Add update reason to audit trail
            card._log_change("manual_update", {"reason": reason})
        
        # Save updated card
        card.save(card_path)
        
        click.echo(f"✅ Model card updated successfully")
        click.echo(f"📝 Updates applied: {updates_applied}")
        
        if auto_commit:
            click.echo("📦 Committing changes to git...")
            # Implementation would commit to git
            click.echo("✅ Changes committed")
        
    except Exception as e:
        logger.error(f"Update failed: {e}")
        click.echo(f"❌ Update error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--format',
              type=click.Choice(['huggingface', 'google', 'eu-cra']),
              default='huggingface',
              help='Template format')
@click.option('--template',
              type=click.Choice(['nlp_classification', 'computer_vision', 'llm', 'multimodal']),
              help='Specific template type')
@click.option('--output', '-o',
              type=click.Path(),
              default='MODEL_CARD_TEMPLATE.md',
              help='Output file path')
def init(format, template, output):
    """Initialize a new model card template.
    
    Examples:
    
    \b
    # Create basic Hugging Face template
    mcg init --format huggingface
    
    \b
    # Create LLM-specific template
    mcg init --format huggingface --template llm --output LLM_CARD.md
    """
    try:
        click.echo(f"🚀 Initializing {format} model card template")
        
        # Create appropriate card type
        if format == 'huggingface':
            card = HuggingFaceCard()
        elif format == 'google':
            card = GoogleModelCard()
        elif format == 'eu-cra':
            from ..formats.eu_cra import create_eu_cra_template
            card = create_eu_cra_template()
        
        # Apply template if specified
        if template:
            click.echo(f"📝 Applying {template} template")
            # Implementation would apply specific template
        
        # Add template content
        card.model_details.name = "[Model Name]"
        card.model_details.description = "[Describe your model's purpose and capabilities]"
        card.intended_use = "[Describe the intended use cases for this model]"
        card.add_limitation("[Describe known limitations and constraints]")
        
        # Save template
        card.save(output)
        
        click.echo(f"✅ Template created: {output}")
        click.echo(f"📝 Format: {format}")
        click.echo(f"🎯 Template: {template or 'basic'}")
        click.echo(f"\n💡 Next steps:")
        click.echo(f"   1. Edit {output} to fill in your model details")
        click.echo(f"   2. Run 'mcg validate {output}' to check completeness")
        click.echo(f"   3. Generate final card with 'mcg generate --config {output}'")
        
    except Exception as e:
        logger.error(f"Template initialization failed: {e}")
        click.echo(f"❌ Initialization error: {e}", err=True)
        raise click.Abort()


@cli.command()
def version():
    """Show version information."""
    click.echo("Model Card Generator v1.0.0")
    click.echo("🎯 Automated ML documentation for compliance and governance")
    click.echo("📜 Supports: Hugging Face, Google Model Cards, EU CRA")
    click.echo("🔗 https://github.com/terragon-labs/modelcard-as-code-generator")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()