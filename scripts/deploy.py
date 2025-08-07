#!/usr/bin/env python3
"""Production deployment script for ModelCard-as-Code-Generator."""

import os
import sys
import subprocess
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Manages production deployment of the ModelCard Generator."""
    
    def __init__(self, environment: str = "production", dry_run: bool = False):
        self.environment = environment
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        self.deployment_config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        config_file = self.project_root / f"deploy_{self.environment}.yml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "python_version": "3.9+",
            "dependencies": ["click", "jinja2", "pydantic", "pyyaml", "requests"],
            "optional_dependencies": {
                "ml_integrations": ["wandb", "mlflow", "huggingface-hub"],
                "dev_tools": ["pytest", "black", "ruff", "mypy"]
            },
            "environment_variables": {
                "MODELCARD_CACHE_TTL": "3600",
                "MODELCARD_LOG_LEVEL": "INFO",
                "MODELCARD_MAX_WORKERS": "4"
            },
            "health_checks": {
                "endpoints": ["/health", "/version"],
                "timeout": 30,
                "interval": 60
            }
        }
    
    def validate_environment(self) -> bool:
        """Validate deployment environment."""
        logger.info("Validating deployment environment...")
        
        checks = []
        
        # Python version check
        python_version = sys.version_info
        required_version = tuple(map(int, self.deployment_config["python_version"].replace("+", "").split(".")))
        if python_version[:2] >= required_version:
            logger.info(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            checks.append(True)
        else:
            logger.error(f"✗ Python version {python_version.major}.{python_version.minor} < {required_version}")
            checks.append(False)
        
        # Dependencies check
        missing_deps = []
        for dep in self.deployment_config["dependencies"]:
            try:
                __import__(dep.split(">=")[0].split("==")[0])
                logger.info(f"✓ Dependency available: {dep}")
            except ImportError:
                missing_deps.append(dep)
                logger.warning(f"⚠ Missing dependency: {dep}")
        
        if not missing_deps:
            checks.append(True)
        else:
            logger.error(f"Missing dependencies: {missing_deps}")
            checks.append(False)
        
        # Environment variables check
        for var, default in self.deployment_config["environment_variables"].items():
            value = os.getenv(var, default)
            logger.info(f"✓ Environment variable {var}={value}")
        
        # Disk space check
        disk_usage = shutil.disk_usage(self.project_root)
        free_gb = disk_usage.free / (1024**3)
        if free_gb > 1.0:  # Require at least 1GB free
            logger.info(f"✓ Disk space: {free_gb:.1f}GB available")
            checks.append(True)
        else:
            logger.error(f"✗ Insufficient disk space: {free_gb:.1f}GB available")
            checks.append(False)
        
        return all(checks)
    
    def run_quality_gates(self) -> bool:
        """Run quality gates before deployment."""
        logger.info("Running quality gates...")
        
        gates_passed = []
        
        # Syntax validation
        logger.info("Running syntax validation...")
        syntax_valid = self._validate_syntax()
        gates_passed.append(syntax_valid)
        
        # Security scan
        logger.info("Running security scan...")
        security_clean = self._security_scan()
        gates_passed.append(security_clean)
        
        # Performance check
        logger.info("Running performance checks...")
        performance_ok = self._performance_check()
        gates_passed.append(performance_ok)
        
        # Configuration validation
        logger.info("Validating configuration...")
        config_valid = self._validate_config()
        gates_passed.append(config_valid)
        
        success = all(gates_passed)
        if success:
            logger.info("✅ All quality gates passed!")
        else:
            logger.error("❌ Some quality gates failed!")
        
        return success
    
    def _validate_syntax(self) -> bool:
        """Validate Python syntax for all source files."""
        import ast
        
        src_dir = self.project_root / "src"
        python_files = list(src_dir.rglob("*.py"))
        
        syntax_errors = []
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
        
        if syntax_errors:
            logger.error("Syntax errors found:")
            for error in syntax_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info(f"✓ Syntax validation passed for {len(python_files)} files")
        return True
    
    def _security_scan(self) -> bool:
        """Run security scan on source code."""
        import re
        
        src_dir = self.project_root / "src"
        python_files = list(src_dir.rglob("*.py"))
        
        dangerous_patterns = [
            (r'eval\\s*\\(', 'Use of eval() function'),
            (r'exec\\s*\\(', 'Use of exec() function'),
            (r'subprocess\\.call\\s*\\([^)]*shell\\s*=\\s*True', 'Subprocess with shell=True'),
            (r'os\\.system\\s*\\(', 'Use of os.system()'),
            (r'pickle\\.loads?\\s*\\(', 'Use of pickle.load/loads'),
        ]
        
        security_issues = []
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in dangerous_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        security_issues.append(f"{file_path}: {description}")
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        if security_issues:
            logger.error("Security issues found:")
            for issue in security_issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info(f"✓ Security scan passed for {len(python_files)} files")
        return True
    
    def _performance_check(self) -> bool:
        """Check performance characteristics."""
        # Check file sizes
        src_dir = self.project_root / "src"
        large_files = []
        
        for file_path in src_dir.rglob("*.py"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 1.0:  # Flag files larger than 1MB
                large_files.append(f"{file_path}: {size_mb:.1f}MB")
        
        if large_files:
            logger.warning("Large files found:")
            for file_info in large_files:
                logger.warning(f"  - {file_info}")
        
        # Check memory usage potential
        total_size = sum(f.stat().st_size for f in src_dir.rglob("*.py")) / (1024 * 1024)
        if total_size > 10:  # Flag if total source > 10MB
            logger.warning(f"Large codebase: {total_size:.1f}MB")
        
        logger.info(f"✓ Performance check completed (codebase: {total_size:.1f}MB)")
        return True
    
    def _validate_config(self) -> bool:
        """Validate configuration files."""
        config_files = [
            ("pyproject.toml", "TOML configuration"),
            ("setup.py", "Setup script"),
        ]
        
        for file_name, description in config_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                try:
                    if file_name.endswith('.toml'):
                        import toml
                        with open(file_path, 'r') as f:
                            toml.load(f)
                    elif file_name.endswith('.py'):
                        with open(file_path, 'r') as f:
                            compile(f.read(), file_path, 'exec')
                    
                    logger.info(f"✓ {description} valid: {file_name}")
                except Exception as e:
                    logger.error(f"✗ {description} invalid: {file_name} - {e}")
                    return False
            else:
                logger.warning(f"⚠ {description} not found: {file_name}")
        
        return True
    
    def build_package(self) -> bool:
        """Build the package for deployment."""
        logger.info("Building package...")
        
        if self.dry_run:
            logger.info("DRY RUN: Would build package")
            return True
        
        try:
            # Clean previous builds
            build_dirs = ['build', 'dist', '*.egg-info']
            for pattern in build_dirs:
                for path in self.project_root.glob(pattern):
                    if path.is_dir():
                        shutil.rmtree(path)
                        logger.info(f"Cleaned {path}")
            
            # Build wheel
            result = subprocess.run([
                sys.executable, '-m', 'build'
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return False
            
            # Check if wheel was created
            dist_dir = self.project_root / 'dist'
            wheels = list(dist_dir.glob('*.whl'))
            if wheels:
                logger.info(f"✓ Package built successfully: {wheels[0].name}")
                return True
            else:
                logger.error("No wheel file found after build")
                return False
                
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False
    
    def create_deployment_manifest(self) -> Dict[str, Any]:
        """Create deployment manifest with metadata."""
        manifest = {
            "deployment": {
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment,
                "version": self._get_version(),
                "commit_hash": self._get_git_commit(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            "configuration": self.deployment_config,
            "dependencies": self._get_dependency_info(),
            "health_checks": {
                "syntax_check": True,
                "security_scan": True,
                "performance_check": True,
                "config_validation": True,
            }
        }
        
        # Save manifest
        manifest_file = self.project_root / f"deployment_manifest_{self.environment}.json"
        if not self.dry_run:
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Deployment manifest saved: {manifest_file}")
        else:
            logger.info("DRY RUN: Would save deployment manifest")
        
        return manifest
    
    def _get_version(self) -> str:
        """Get package version."""
        try:
            with open(self.project_root / "src/modelcard_generator/__init__.py", 'r') as f:
                content = f.read()
                for line in content.split('\\n'):
                    if line.startswith('__version__'):
                        return line.split('=')[1].strip().strip('"').strip("'")
            return "unknown"
        except Exception:
            return "unknown"
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                return result.stdout.strip()
            return "unknown"
        except Exception:
            return "unknown"
    
    def _get_dependency_info(self) -> Dict[str, str]:
        """Get installed dependency versions."""
        dependencies = {}
        for dep in self.deployment_config["dependencies"]:
            try:
                package_name = dep.split(">=")[0].split("==")[0]
                module = __import__(package_name)
                version = getattr(module, '__version__', 'unknown')
                dependencies[package_name] = version
            except ImportError:
                dependencies[package_name] = 'not_installed'
            except Exception:
                dependencies[package_name] = 'unknown'
        
        return dependencies
    
    def deploy(self) -> bool:
        """Execute full deployment process."""
        logger.info(f"Starting deployment to {self.environment} environment...")
        
        if self.dry_run:
            logger.info("🧪 DRY RUN MODE - No changes will be made")
        
        # Step 1: Validate environment
        if not self.validate_environment():
            logger.error("❌ Environment validation failed")
            return False
        
        # Step 2: Run quality gates
        if not self.run_quality_gates():
            logger.error("❌ Quality gates failed")
            return False
        
        # Step 3: Build package
        if not self.build_package():
            logger.error("❌ Package build failed")
            return False
        
        # Step 4: Create deployment manifest
        manifest = self.create_deployment_manifest()
        
        # Step 5: Deployment summary
        logger.info("🚀 Deployment completed successfully!")
        logger.info(f"   Environment: {self.environment}")
        logger.info(f"   Version: {manifest['deployment']['version']}")
        logger.info(f"   Timestamp: {manifest['deployment']['timestamp']}")
        
        return True


def main():
    """Main deployment script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy ModelCard Generator")
    parser.add_argument("--environment", "-e", default="production",
                       choices=["development", "staging", "production"],
                       help="Deployment environment")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Perform dry run without making changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(
        environment=args.environment,
        dry_run=args.dry_run
    )
    
    # Execute deployment
    success = deployment_manager.deploy()
    
    if success:
        logger.info("✅ Deployment successful!")
        sys.exit(0)
    else:
        logger.error("❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()