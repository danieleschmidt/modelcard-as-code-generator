#!/usr/bin/env python3
"""Autonomous SDLC Executor - Standalone execution script for the TERRAGON methodology."""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modelcard_generator.research.autonomous_executor import AutonomousExecutor
from modelcard_generator.core.logging_config import configure_logging, get_logger

configure_logging(level="INFO", structured=True)
logger = get_logger(__name__)


async def main():
    """Main execution function for autonomous SDLC."""
    logger.info("🧠 TERRAGON SDLC MASTER PROMPT v4.0 - AUTONOMOUS EXECUTION")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Analyze current project context
    project_context = await analyze_project_context()
    
    # Initialize autonomous executor
    executor = AutonomousExecutor()
    
    try:
        # Execute autonomous SDLC cycle
        result = await executor.execute_autonomous_sdlc(project_context)
        
        # Display results
        display_execution_results(result, time.time() - start_time)
        
        # Save execution report
        await save_execution_report(result)
        
        # Determine exit code
        exit_code = 0 if result["status"] == "completed" and result.get("quality_gates_passed", False) else 1
        
        if exit_code == 0:
            logger.info("✅ AUTONOMOUS SDLC EXECUTION COMPLETED SUCCESSFULLY")
        else:
            logger.error("❌ AUTONOMOUS SDLC EXECUTION COMPLETED WITH ISSUES")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"💥 AUTONOMOUS SDLC EXECUTION FAILED: {e}")
        return 1


async def analyze_project_context() -> Dict[str, Any]:
    """Analyze the current project to understand context."""
    logger.info("🔍 Analyzing project context...")
    
    context = {
        "project_root": str(Path.cwd()),
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "project_files": {}
    }
    
    # Read key project files
    key_files = [
        "pyproject.toml",
        "setup.py", 
        "requirements.txt",
        "README.md",
        "CHANGELOG.md"
    ]
    
    for file_name in key_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                context["project_files"][file_name] = file_path.read_text()
                logger.debug(f"Read {file_name}")
            except Exception as e:
                logger.warning(f"Could not read {file_name}: {e}")
    
    # Analyze directory structure
    context["structure"] = analyze_directory_structure(Path.cwd())
    
    # Detect project type and dependencies
    context["type"] = detect_project_type(context)
    context["dependencies"] = extract_dependencies(context)
    context["dev_dependencies"] = extract_dev_dependencies(context)
    
    logger.info(f"📊 Detected project type: {context['type']}")
    logger.info(f"📦 Found {len(context.get('dependencies', []))} dependencies")
    
    return context


def analyze_directory_structure(root_path: Path, max_depth: int = 3) -> Dict[str, Any]:
    """Analyze directory structure."""
    structure = {"directories": [], "files": [], "key_patterns": []}
    
    try:
        for item in root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                structure["directories"].append(item.name)
                if item.name in ["src", "tests", "docs", "scripts", "deployment"]:
                    structure["key_patterns"].append(f"has_{item.name}_directory")
            elif item.is_file():
                structure["files"].append(item.name)
                if item.name in ["Dockerfile", "docker-compose.yml", "Makefile"]:
                    structure["key_patterns"].append(f"has_{item.name}")
    except PermissionError:
        logger.warning("Permission denied reading directory structure")
    
    return structure


def detect_project_type(context: Dict[str, Any]) -> str:
    """Detect the type of project."""
    files = context.get("project_files", {})
    structure = context.get("structure", {})
    
    # Check for Python package
    if "pyproject.toml" in files or "setup.py" in files:
        # Check for CLI entry points
        if "entry_points" in str(files.get("pyproject.toml", "")) or "console_scripts" in str(files.get("setup.py", "")):
            return "Python CLI Package"
        return "Python Package"
    
    # Check for web application
    if any("flask" in str(f).lower() or "django" in str(f).lower() or "fastapi" in str(f).lower() 
           for f in files.values()):
        return "Web Application"
    
    # Check for machine learning
    if any("torch" in str(f).lower() or "tensorflow" in str(f).lower() or "sklearn" in str(f).lower()
           for f in files.values()):
        return "Machine Learning Project"
    
    return "General Python Project"


def extract_dependencies(context: Dict[str, Any]) -> List[str]:
    """Extract project dependencies."""
    dependencies = []
    
    files = context.get("project_files", {})
    
    # From requirements.txt
    if "requirements.txt" in files:
        deps = [line.strip() for line in files["requirements.txt"].split('\n') 
                if line.strip() and not line.startswith('#')]
        dependencies.extend(deps)
    
    # From pyproject.toml (simplified parsing)
    if "pyproject.toml" in files:
        content = files["pyproject.toml"]
        if "dependencies" in content:
            # Simple regex-like extraction (would use proper TOML parser in production)
            import re
            deps = re.findall(r'"([^"]+)"', content)
            dependencies.extend([d for d in deps if '=' in d or '>' in d or '<' in d])
    
    return dependencies


def extract_dev_dependencies(context: Dict[str, Any]) -> List[str]:
    """Extract development dependencies."""
    dev_deps = []
    
    files = context.get("project_files", {})
    
    # From requirements-dev.txt
    if "requirements-dev.txt" in files:
        deps = [line.strip() for line in files["requirements-dev.txt"].split('\n')
                if line.strip() and not line.startswith('#')]
        dev_deps.extend(deps)
    
    # From pyproject.toml dev section
    if "pyproject.toml" in files:
        content = files["pyproject.toml"]
        if "dev" in content:
            # Simple extraction (would use proper TOML parser in production)
            import re
            if '[project.optional-dependencies.dev]' in content or '[tool.poetry.group.dev.dependencies]' in content:
                # Extract dev dependencies section
                pass  # Simplified for demo
    
    return dev_deps


def display_execution_results(result: Dict[str, Any], total_duration: float) -> None:
    """Display comprehensive execution results."""
    logger.info("\n" + "=" * 80)
    logger.info("🎯 AUTONOMOUS SDLC EXECUTION RESULTS")
    logger.info("=" * 80)
    
    # Overall status
    status_emoji = "✅" if result["status"] == "completed" else "❌"
    logger.info(f"{status_emoji} Status: {result['status'].upper()}")
    logger.info(f"⏱️  Total Duration: {total_duration:.2f} seconds")
    logger.info(f"📊 Phases Completed: {result.get('phases_completed', 0)}")
    logger.info(f"🔒 Quality Gates: {'PASSED' if result.get('quality_gates_passed', False) else 'FAILED'}")
    
    # Artifacts generated
    artifacts = result.get('artifacts_generated', [])
    if artifacts:
        logger.info(f"📦 Artifacts Generated: {len(artifacts)}")
        for artifact in artifacts[:5]:  # Show first 5
            logger.info(f"   - {artifact}")
        if len(artifacts) > 5:
            logger.info(f"   ... and {len(artifacts) - 5} more")
    
    # Performance metrics
    metrics = result.get('performance_metrics', {})
    if metrics:
        logger.info("📈 Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"   - {metric}: {value}")
    
    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        logger.info("💡 Recommendations:")
        for rec in recommendations[:3]:  # Show first 3
            logger.info(f"   - {rec}")
        if len(recommendations) > 3:
            logger.info(f"   ... and {len(recommendations) - 3} more")
    
    # Error information if failed
    if result["status"] == "failed":
        error = result.get("error", "Unknown error")
        logger.error(f"💥 Error: {error}")
    
    logger.info("=" * 80)


async def save_execution_report(result: Dict[str, Any]) -> None:
    """Save detailed execution report."""
    report_path = Path("autonomous_sdlc_report.json")
    
    try:
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"📄 Execution report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save execution report: {e}")


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)