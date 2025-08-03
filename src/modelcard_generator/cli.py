"""
Command-line interface for Model Card Generator.

Provides a comprehensive CLI for generating, validating, and managing
model cards with support for multiple formats and standards.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.generator import ModelCardGenerator
from .core.config import CardConfig
from .core.model_card import ModelCard
from .validators.validator import Validator
from .drift.detector import DriftDetector
from .compliance.checker import ComplianceChecker
from .security.scanner import scan_model_card_for_secrets


# Global console for rich output
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """Model Card as Code Generator - Automated model documentation."""
    setup_logging(verbose)
    
    # Load configuration
    if config:
        card_config = CardConfig.from_file(Path(config))
    else:
        card_config = CardConfig()
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = card_config
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--eval-results', '-e', type=click.Path(exists=True), help='Evaluation results file')
@click.option('--training-history', '-t', type=click.Path(exists=True), help='Training history file')
@click.option('--dataset-info', '-d', type=click.Path(exists=True), help='Dataset information file')
@click.option('--model-config', '-m', type=click.Path(exists=True), help='Model configuration file')
@click.option('--model-name', '-n', help='Model name')
@click.option('--model-version', help='Model version')
@click.option('--format', '-f', 
              type=click.Choice(['huggingface', 'google', 'eu_cra']), 
              default='huggingface', 
              help='Output format')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output file path')
@click.option('--output-format', 
              type=click.Choice(['markdown', 'json', 'yaml']), 
              default='markdown',
              help='Output file format')
@click.pass_context
def generate(
    ctx: click.Context,
    eval_results: Optional[str],
    training_history: Optional[str], 
    dataset_info: Optional[str],
    model_config: Optional[str],
    model_name: Optional[str],
    model_version: Optional[str],
    format: str,
    output: str,
    output_format: str
) -> None:
    """Generate a model card from provided sources."""
    
    config = ctx.obj['config']
    config.format = format
    config.output_format = output_format
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Initialize generator
        task = progress.add_task("Initializing generator...", total=None)
        generator = ModelCardGenerator(config)
        
        # Generate model card
        progress.update(task, description="Generating model card...")
        
        try:
            model_card = generator.generate(
                eval_results=eval_results,
                training_history=training_history,
                dataset_info=dataset_info,
                model_config=model_config,
                model_name=model_name,
                model_version=model_version
            )
            
            # Export model card
            progress.update(task, description="Exporting model card...")
            output_path = Path(output)
            generator.export(model_card, output_path, output_format)
            
            progress.update(task, description="Complete!", completed=True)
            
            # Display results
            console.print(f"✅ Model card generated successfully!", style="green")
            console.print(f"📄 Output: {output_path}")
            console.print(f"📊 Completeness: {model_card.get_completeness_score():.1%}")
            
            # Show metrics summary
            if model_card.metrics:
                console.print(f"📈 Metrics: {len(model_card.metrics)} included")
            
        except Exception as e:
            console.print(f"❌ Error generating model card: {str(e)}", style="red")
            sys.exit(1)


@cli.command()
@click.argument('card_path', type=click.Path(exists=True))
@click.option('--standard', '-s', 
              type=click.Choice(['huggingface', 'google', 'eu_cra']),
              default='huggingface',
              help='Validation standard')
@click.option('--output', '-o', type=click.Path(), help='Output validation report')
@click.pass_context
def validate(ctx: click.Context, card_path: str, standard: str, output: Optional[str]) -> None:
    """Validate a model card against a standard."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Loading model card...", total=None)
        
        try:
            # Load model card
            model_card = ModelCard.load(Path(card_path))
            
            # Validate
            progress.update(task, description="Validating model card...")
            validator = Validator()
            result = validator.validate(model_card, standard)
            
            progress.update(task, description="Complete!", completed=True)
            
            # Display results
            _display_validation_results(result, standard)
            
            # Save report if requested
            if output:
                _save_validation_report(result, Path(output))
                console.print(f"📄 Validation report saved to: {output}")
            
            # Exit with appropriate code
            if not result.is_valid:
                sys.exit(1)
                
        except Exception as e:
            console.print(f"❌ Error validating model card: {str(e)}", style="red")
            sys.exit(1)


@cli.command()
@click.argument('card_path', type=click.Path(exists=True))
@click.option('--new-data', '-n', type=click.Path(exists=True), required=True, 
              help='New evaluation data or model card')
@click.option('--threshold', '-t', multiple=True, 
              help='Custom thresholds (format: metric_name:threshold)')
@click.option('--fail-on-drift', is_flag=True, help='Exit with error if drift detected')
@click.option('--output', '-o', type=click.Path(), help='Output drift report')
@click.pass_context
def check_drift(
    ctx: click.Context, 
    card_path: str, 
    new_data: str,
    threshold: List[str],
    fail_on_drift: bool,
    output: Optional[str]
) -> None:
    """Check for model drift against new data."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Loading model card...", total=None)
        
        try:
            # Load current model card
            current_card = ModelCard.load(Path(card_path))
            
            # Parse custom thresholds
            custom_thresholds = {}
            for t in threshold:
                if ':' in t:
                    metric, thresh = t.split(':', 1)
                    custom_thresholds[metric] = float(thresh)
            
            # Check drift
            progress.update(task, description="Detecting drift...")
            detector = DriftDetector()
            drift_report = detector.check_drift(current_card, new_data, custom_thresholds)
            
            progress.update(task, description="Complete!", completed=True)
            
            # Display results
            _display_drift_results(drift_report)
            
            # Save report if requested
            if output:
                _save_drift_report(drift_report, Path(output))
                console.print(f"📄 Drift report saved to: {output}")
            
            # Exit with error if drift detected and flag set
            if fail_on_drift and drift_report.has_drift:
                console.print("❌ Drift detected - failing as requested", style="red")
                sys.exit(1)
                
        except Exception as e:
            console.print(f"❌ Error checking drift: {str(e)}", style="red")
            sys.exit(1)


@cli.command()
@click.argument('card_path', type=click.Path(exists=True))
@click.option('--standards', '-s', multiple=True,
              type=click.Choice(['gdpr', 'eu_ai_act', 'ccpa', 'iso_23053']),
              help='Compliance standards to check')
@click.option('--strict', is_flag=True, help='Use strict compliance checking')
@click.option('--output', '-o', type=click.Path(), help='Output compliance report')
@click.pass_context
def check_compliance(
    ctx: click.Context,
    card_path: str,
    standards: List[str],
    strict: bool,
    output: Optional[str]
) -> None:
    """Check compliance against regulatory standards."""
    
    # Default standards if none specified
    if not standards:
        standards = ['gdpr', 'eu_ai_act']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Loading model card...", total=None)
        
        try:
            # Load model card
            model_card = ModelCard.load(Path(card_path))
            
            # Check compliance
            progress.update(task, description="Checking compliance...")
            checker = ComplianceChecker()
            results = checker.check_multiple_standards(model_card, list(standards), strict)
            
            progress.update(task, description="Complete!", completed=True)
            
            # Display results
            _display_compliance_results(results)
            
            # Save report if requested
            if output:
                _save_compliance_report(results, Path(output))
                console.print(f"📄 Compliance report saved to: {output}")
            
            # Exit with error if any standard fails
            if any(not result.compliant for result in results.values()):
                sys.exit(1)
                
        except Exception as e:
            console.print(f"❌ Error checking compliance: {str(e)}", style="red")
            sys.exit(1)


@cli.command()
@click.argument('card_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output security report')
@click.pass_context
def scan_security(ctx: click.Context, card_path: str, output: Optional[str]) -> None:
    """Scan model card for security issues."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Loading model card...", total=None)
        
        try:
            # Load model card
            model_card = ModelCard.load(Path(card_path))
            
            # Scan for security issues
            progress.update(task, description="Scanning for security issues...")
            report = scan_model_card_for_secrets(model_card.to_dict())
            
            progress.update(task, description="Complete!", completed=True)
            
            # Display results
            _display_security_results(report)
            
            # Save report if requested
            if output:
                _save_security_report(report, Path(output))
                console.print(f"📄 Security report saved to: {output}")
            
            # Exit with error if critical issues found
            if report['status'] == 'critical':
                sys.exit(1)
                
        except Exception as e:
            console.print(f"❌ Error scanning security: {str(e)}", style="red")
            sys.exit(1)


@cli.command()
@click.argument('card_path', type=click.Path(exists=True))
@click.pass_context  
def info(ctx: click.Context, card_path: str) -> None:
    """Display information about a model card."""
    
    try:
        model_card = ModelCard.load(Path(card_path))
        summary = model_card.get_summary()
        
        # Create info table
        table = Table(title="Model Card Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Model Name", summary['model_name'])
        table.add_row("Model Version", summary.get('model_version', 'Not specified'))
        table.add_row("Metrics Count", str(summary['metrics_count']))
        table.add_row("Training Datasets", str(summary['training_datasets']))
        table.add_row("Evaluation Datasets", str(summary['evaluation_datasets']))
        table.add_row("Custom Sections", str(summary['custom_sections']))
        table.add_row("Completeness Score", f"{summary['completeness_score']:.1%}")
        table.add_row("Content Hash", summary['content_hash'][:16] + "...")
        table.add_row("Last Updated", summary.get('last_updated', 'Unknown'))
        
        console.print(table)
        
        # Show metrics if available
        if model_card.metrics:
            console.print("\n")
            metrics_table = Table(title="Metrics")
            metrics_table.add_column("Name", style="cyan")
            metrics_table.add_column("Value", style="white")
            metrics_table.add_column("Unit", style="dim")
            
            for metric in model_card.metrics:
                metrics_table.add_row(
                    metric.name,
                    str(metric.value),
                    metric.unit or ""
                )
            
            console.print(metrics_table)
        
    except Exception as e:
        console.print(f"❌ Error reading model card: {str(e)}", style="red")
        sys.exit(1)


# Helper functions for displaying results

def _display_validation_results(result, standard: str) -> None:
    """Display validation results."""
    if result.is_valid:
        console.print(f"✅ Model card is valid for {standard} standard", style="green")
    else:
        console.print(f"❌ Model card validation failed for {standard} standard", style="red")
    
    console.print(f"📊 Validation Score: {result.score:.1%}")
    
    if result.errors:
        console.print("\n❌ Errors:", style="red")
        for error in result.errors:
            console.print(f"  • {error}")
    
    if result.warnings:
        console.print("\n⚠️ Warnings:", style="yellow")
        for warning in result.warnings:
            console.print(f"  • {warning}")


def _display_drift_results(report) -> None:
    """Display drift detection results."""
    if report.has_drift:
        console.print(f"⚠️ Model drift detected (Severity: {report.severity})", style="yellow")
    else:
        console.print("✅ No significant drift detected", style="green")
    
    console.print(f"\n📄 Summary: {report.summary}")
    console.print(f"💡 Recommendation: {report.recommendation}")
    
    if report.changes:
        console.print("\n📊 Changes Detected:")
        
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Old Value", style="white")
        table.add_column("New Value", style="white")
        table.add_column("Change", style="white")
        table.add_column("Significance", style="white")
        
        for change in report.changes:
            delta_str = ""
            if change.delta is not None:
                delta_str = f"{change.delta:+.4f}"
                if change.delta_percent is not None:
                    delta_str += f" ({change.delta_percent:+.1f}%)"
            
            significance_style = {
                "low": "dim",
                "medium": "yellow",
                "high": "red",
                "critical": "bold red"
            }.get(change.significance, "white")
            
            table.add_row(
                change.metric_name,
                str(change.old_value) if change.old_value is not None else "N/A",
                str(change.new_value) if change.new_value is not None else "N/A",
                delta_str,
                f"[{significance_style}]{change.significance}[/{significance_style}]"
            )
        
        console.print(table)


def _display_compliance_results(results: Dict[str, Any]) -> None:
    """Display compliance check results."""
    table = Table(title="Compliance Check Results")
    table.add_column("Standard", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Score", style="white")
    table.add_column("Missing Requirements", style="white")
    
    for standard, result in results.items():
        status = "✅ Compliant" if result.compliant else "❌ Non-compliant"
        status_style = "green" if result.compliant else "red"
        
        table.add_row(
            standard.upper(),
            f"[{status_style}]{status}[/{status_style}]",
            f"{result.score:.1%}",
            str(len(result.missing_requirements))
        )
    
    console.print(table)
    
    # Show recommendations
    all_recommendations = []
    for result in results.values():
        all_recommendations.extend(result.recommendations)
    
    if all_recommendations:
        console.print("\n💡 Recommendations:")
        for recommendation in set(all_recommendations):  # Remove duplicates
            console.print(f"  • {recommendation}")


def _display_security_results(report: Dict[str, Any]) -> None:
    """Display security scan results."""
    status_style = {
        "clean": "green",
        "low_risk": "yellow", 
        "medium_risk": "yellow",
        "high_risk": "red",
        "critical": "bold red"
    }.get(report['status'], "white")
    
    console.print(f"🔒 Security Status: [{status_style}]{report['status'].upper()}[/{status_style}]")
    console.print(f"📄 {report['summary']}")
    
    if report['findings']:
        console.print("\n🔍 Security Findings:")
        
        table = Table()
        table.add_column("Type", style="cyan")
        table.add_column("Severity", style="white")
        table.add_column("Message", style="white")
        table.add_column("Location", style="dim")
        
        for finding in report['findings']:
            severity_style = {
                "low": "dim",
                "medium": "yellow", 
                "high": "red",
                "critical": "bold red"
            }.get(finding['severity'], "white")
            
            table.add_row(
                finding['type'],
                f"[{severity_style}]{finding['severity'].upper()}[/{severity_style}]",
                finding['message'],
                finding.get('location', 'N/A')
            )
        
        console.print(table)
    
    if report.get('recommendations'):
        console.print("\n💡 Security Recommendations:")
        for rec in report['recommendations']:
            console.print(f"  • {rec}")


# Helper functions for saving reports

def _save_validation_report(result, output_path: Path) -> None:
    """Save validation report to file."""
    report_data = {
        "validation_result": {
            "is_valid": result.is_valid,
            "score": result.score,
            "errors": result.errors,
            "warnings": result.warnings,
            "details": result.details
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)


def _save_drift_report(report, output_path: Path) -> None:
    """Save drift report to file."""
    report_data = {
        "drift_report": {
            "has_drift": report.has_drift,
            "timestamp": report.timestamp,
            "summary": report.summary,
            "recommendation": report.recommendation,
            "severity": report.severity,
            "changes": [
                {
                    "metric_name": change.metric_name,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "delta": change.delta,
                    "delta_percent": change.delta_percent,
                    "threshold_exceeded": change.threshold_exceeded,
                    "significance": change.significance
                }
                for change in report.changes
            ]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)


def _save_compliance_report(results: Dict[str, Any], output_path: Path) -> None:
    """Save compliance report to file."""
    report_data = {
        "compliance_results": {
            standard: {
                "compliant": result.compliant,
                "score": result.score,
                "missing_requirements": result.missing_requirements,
                "satisfied_requirements": result.satisfied_requirements,
                "warnings": result.warnings,
                "recommendations": result.recommendations
            }
            for standard, result in results.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)


def _save_security_report(report: Dict[str, Any], output_path: Path) -> None:
    """Save security report to file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()