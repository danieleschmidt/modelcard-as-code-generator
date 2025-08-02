"""
Test runners and utilities for different testing scenarios.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class TestRunner:
    """Enhanced test runner with different execution modes."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
    
    def run_quick_tests(self) -> int:
        """Run quick smoke tests for fast feedback."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "unit"),
            "-m", "not slow",
            "-x",  # Stop on first failure
            "--tb=short",
            "-q"
        ]
        return subprocess.call(cmd)
    
    def run_full_test_suite(self) -> int:
        """Run the complete test suite with coverage."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--cov=src/modelcard_generator",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "-v"
        ]
        return subprocess.call(cmd)
    
    def run_security_tests(self) -> int:
        """Run security-focused tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "security"),
            "-m", "security",
            "-v"
        ]
        return subprocess.call(cmd)
    
    def run_performance_tests(self) -> int:
        """Run performance benchmarks."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "performance"),
            "-m", "performance",
            "--benchmark-only",
            "-v"
        ]
        return subprocess.call(cmd)
    
    def run_integration_tests(self) -> int:
        """Run integration tests."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "integration"),
            "-m", "integration",
            "-v"
        ]
        return subprocess.call(cmd)
    
    def run_parallel_tests(self, num_workers: int = None) -> int:
        """Run tests in parallel."""
        if num_workers is None:
            num_workers = "auto"
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-n", str(num_workers),
            "-m", "not slow",
            "-v"
        ]
        return subprocess.call(cmd)


class ContinuousTestRunner:
    """Continuous testing with file watching."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        self.last_run_time = 0
        self.test_runner = TestRunner(project_root)
    
    def watch_and_test(self, debounce_seconds: float = 2.0):
        """Watch for file changes and run tests automatically."""
        try:
            import watchdog.observers
            import watchdog.events
        except ImportError:
            print("watchdog not installed. Install with: pip install watchdog")
            return 1
        
        class TestEventHandler(watchdog.events.FileSystemEventHandler):
            def __init__(self, runner_instance):
                self.runner = runner_instance
            
            def on_modified(self, event):
                if event.is_directory:
                    return
                
                if self._should_trigger_tests(event.src_path):
                    current_time = time.time()
                    if current_time - self.runner.last_run_time > debounce_seconds:
                        self.runner.last_run_time = current_time
                        print(f"\n📁 File changed: {event.src_path}")
                        print("🧪 Running tests...")
                        self.runner.test_runner.run_quick_tests()
            
            def _should_trigger_tests(self, file_path: str) -> bool:
                """Check if file change should trigger test run."""
                path = Path(file_path)
                
                # Python files in src or tests
                if path.suffix == ".py" and (
                    str(self.runner.src_dir) in str(path) or
                    str(self.runner.test_dir) in str(path)
                ):
                    return True
                
                # Configuration files
                config_files = [
                    "pyproject.toml", "pytest.ini", "conftest.py",
                    ".env", ".env.example"
                ]
                if path.name in config_files:
                    return True
                
                return False
        
        observer = watchdog.observers.Observer()
        event_handler = TestEventHandler(self)
        
        # Watch source and test directories
        observer.schedule(event_handler, str(self.src_dir), recursive=True)
        observer.schedule(event_handler, str(self.test_dir), recursive=True)
        
        observer.start()
        print("👀 Watching for file changes... Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n🛑 Test watching stopped")
        
        observer.join()
        return 0


class TestReporter:
    """Generate test reports and summaries."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
    
    def generate_test_summary(self) -> Dict[str, int]:
        """Generate a summary of test counts by category."""
        summary = {
            "total": 0,
            "unit": 0,
            "integration": 0,
            "security": 0,
            "performance": 0,
            "slow": 0
        }
        
        # Collect tests
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--collect-only", "-q"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            
            # Parse output to count tests
            lines = result.stdout.split('\n')
            for line in lines:
                if " test" in line and "collected" in line:
                    # Extract number from "collected X items"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "collected" and i + 1 < len(parts):
                            summary["total"] = int(parts[i + 1])
                            break
            
            # Count by markers (approximate)
            for test_type in ["unit", "integration", "security", "performance"]:
                cmd_marked = [
                    sys.executable, "-m", "pytest",
                    str(self.test_dir),
                    "-m", test_type,
                    "--collect-only", "-q"
                ]
                
                try:
                    result_marked = subprocess.run(
                        cmd_marked, capture_output=True, text=True, check=True
                    )
                    lines_marked = result_marked.stdout.split('\n')
                    for line in lines_marked:
                        if " test" in line and "collected" in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == "collected" and i + 1 < len(parts):
                                    summary[test_type] = int(parts[i + 1])
                                    break
                except subprocess.CalledProcessError:
                    summary[test_type] = 0
            
        except subprocess.CalledProcessError:
            print("Failed to collect test information")
        
        return summary
    
    def print_test_summary(self):
        """Print a formatted test summary."""
        summary = self.generate_test_summary()
        
        print("\n📊 Test Summary")
        print("=" * 40)
        print(f"Total Tests:      {summary['total']:3d}")
        print(f"Unit Tests:       {summary['unit']:3d}")
        print(f"Integration:      {summary['integration']:3d}")
        print(f"Security:         {summary['security']:3d}")
        print(f"Performance:      {summary['performance']:3d}")
        print("=" * 40)
    
    def generate_coverage_report(self) -> Optional[float]:
        """Generate coverage report and return coverage percentage."""
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--cov=src/modelcard_generator",
            "--cov-report=term",
            "--cov-report=html",
            "-q"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            
            # Parse coverage percentage from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "TOTAL" in line and "%" in line:
                    # Extract percentage
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            return float(part[:-1])
            
        except subprocess.CalledProcessError as e:
            print(f"Coverage report failed: {e}")
        
        return None


def main():
    """Command-line interface for test runners."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Card Generator Test Runner")
    parser.add_argument(
        "command",
        choices=[
            "quick", "full", "security", "performance", 
            "integration", "parallel", "watch", "summary"
        ],
        help="Test command to run"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None,
        help="Number of parallel workers (for parallel command)"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    continuous_runner = ContinuousTestRunner()
    reporter = TestReporter()
    
    if args.command == "quick":
        return runner.run_quick_tests()
    elif args.command == "full":
        return runner.run_full_test_suite()
    elif args.command == "security":
        return runner.run_security_tests()
    elif args.command == "performance":
        return runner.run_performance_tests()
    elif args.command == "integration":
        return runner.run_integration_tests()
    elif args.command == "parallel":
        return runner.run_parallel_tests(args.workers)
    elif args.command == "watch":
        return continuous_runner.watch_and_test()
    elif args.command == "summary":
        reporter.print_test_summary()
        coverage = reporter.generate_coverage_report()
        if coverage is not None:
            print(f"\n📈 Code Coverage: {coverage:.1f}%")
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())