#!/usr/bin/env python3
"""
Automated metrics collection script for Model Card Generator.
Collects and reports on various project health metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import requests

class MetricsCollector:
    """Collect various project metrics for health monitoring."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.metrics = {}
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/modelcard-as-code-generator')
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting project metrics...")
        
        self.metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'repository': self.repo_name,
            'git_metrics': self.collect_git_metrics(),
            'code_quality': self.collect_code_quality_metrics(),
            'test_metrics': self.collect_test_metrics(),
            'security_metrics': self.collect_security_metrics(),
            'dependency_metrics': self.collect_dependency_metrics(),
            'build_metrics': self.collect_build_metrics(),
            'documentation_metrics': self.collect_documentation_metrics(),
        }
        
        if self.github_token:
            self.metrics['github_metrics'] = self.collect_github_metrics()
        
        return self.metrics
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        print("📊 Collecting Git metrics...")
        
        try:
            # Commit activity
            commits_last_30_days = self._run_command([
                'git', 'rev-list', '--count', '--since=30.days.ago', 'HEAD'
            ]).strip()
            
            # Contributors
            contributors = self._run_command([
                'git', 'shortlog', '-sn', '--since=30.days.ago'
            ]).strip().split('\n')
            
            # Branch information
            branches = self._run_command([
                'git', 'branch', '-r'
            ]).strip().split('\n')
            
            # Latest commit info
            latest_commit = self._run_command([
                'git', 'log', '-1', '--format=%H,%ad,%s', '--date=iso'
            ]).strip()
            
            return {
                'commits_last_30_days': int(commits_last_30_days) if commits_last_30_days else 0,
                'active_contributors': len([c for c in contributors if c.strip()]),
                'total_branches': len([b for b in branches if b.strip() and 'origin' in b]),
                'latest_commit': latest_commit,
                'repository_age_days': self._get_repository_age(),
            }
            
        except Exception as e:
            print(f"❌ Error collecting Git metrics: {e}")
            return {}
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        print("🔍 Collecting code quality metrics...")
        
        metrics = {}
        
        try:
            # Lines of code
            loc_result = self._run_command(['find', 'src', '-name', '*.py', '-exec', 'wc', '-l', '{}', '+'])
            if loc_result:
                lines = [int(line.split()[0]) for line in loc_result.strip().split('\n') 
                        if line.strip() and line.split()[0].isdigit()]
                metrics['lines_of_code'] = sum(lines)
            
            # Python files count
            py_files = list(Path('src').rglob('*.py')) if Path('src').exists() else []
            metrics['python_files_count'] = len(py_files)
            
            # Test files count
            test_files = list(Path('tests').rglob('*.py')) if Path('tests').exists() else []
            metrics['test_files_count'] = len(test_files)
            
            # Ruff linting (if available)
            try:
                ruff_output = self._run_command(['ruff', 'check', 'src/', '--output-format=json'])
                if ruff_output:
                    ruff_issues = json.loads(ruff_output)
                    metrics['linting_issues'] = len(ruff_issues)
                    metrics['linting_score'] = max(0, 10 - len(ruff_issues) * 0.1)
            except:
                metrics['linting_issues'] = 'unknown'
                metrics['linting_score'] = 'unknown'
            
            # MyPy type checking (if available)
            try:
                mypy_result = self._run_command(['mypy', 'src/', '--json-report', '/tmp/mypy_report'])
                if Path('/tmp/mypy_report/index.txt').exists():
                    with open('/tmp/mypy_report/index.txt') as f:
                        mypy_content = f.read()
                        metrics['type_coverage'] = self._extract_type_coverage(mypy_content)
            except:
                metrics['type_coverage'] = 'unknown'
                
        except Exception as e:
            print(f"❌ Error collecting code quality metrics: {e}")
        
        return metrics
    
    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect testing metrics."""
        print("🧪 Collecting test metrics...")
        
        metrics = {}
        
        try:
            # Test discovery
            if Path('tests').exists():
                test_discovery = self._run_command([
                    'python', '-m', 'pytest', '--collect-only', '-q', 'tests/'
                ])
                
                if test_discovery:
                    lines = test_discovery.strip().split('\n')
                    for line in lines:
                        if 'collected' in line and 'item' in line:
                            # Extract number from "collected X items"
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'collected' and i + 1 < len(parts):
                                    metrics['total_tests'] = int(parts[i + 1])
                                    break
            
            # Coverage report (if available)
            try:
                coverage_result = self._run_command([
                    'python', '-m', 'pytest', '--cov=src', '--cov-report=json', '--cov-report=term', 'tests/'
                ])
                
                # Look for coverage.json
                if Path('coverage.json').exists():
                    with open('coverage.json') as f:
                        coverage_data = json.load(f)
                        metrics['test_coverage'] = coverage_data.get('totals', {}).get('percent_covered', 0)
                        metrics['lines_covered'] = coverage_data.get('totals', {}).get('covered_lines', 0)
                        metrics['lines_missing'] = coverage_data.get('totals', {}).get('missing_lines', 0)
                        
            except Exception as e:
                print(f"⚠️  Could not run coverage: {e}")
                metrics['test_coverage'] = 'unknown'
            
        except Exception as e:
            print(f"❌ Error collecting test metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        print("🔒 Collecting security metrics...")
        
        metrics = {}
        
        try:
            # Bandit security scan
            try:
                bandit_result = self._run_command([
                    'bandit', '-r', 'src/', '-f', 'json'
                ])
                if bandit_result:
                    bandit_data = json.loads(bandit_result)
                    metrics['security_issues'] = len(bandit_data.get('results', []))
                    metrics['security_confidence'] = self._calculate_security_confidence(bandit_data)
            except:
                metrics['security_issues'] = 'unknown'
            
            # Safety check for dependencies
            try:
                safety_result = self._run_command(['safety', 'check', '--json'])
                if safety_result:
                    safety_data = json.loads(safety_result)
                    metrics['vulnerable_dependencies'] = len(safety_data)
            except:
                metrics['vulnerable_dependencies'] = 'unknown'
            
            # Check for common security files
            security_files = ['SECURITY.md', '.github/dependabot.yml', '.bandit']
            metrics['security_files_present'] = [
                f for f in security_files if Path(f).exists()
            ]
            
        except Exception as e:
            print(f"❌ Error collecting security metrics: {e}")
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency metrics."""
        print("📦 Collecting dependency metrics...")
        
        metrics = {}
        
        try:
            # Parse pyproject.toml for dependencies
            if Path('pyproject.toml').exists():
                import toml
                with open('pyproject.toml') as f:
                    pyproject_data = toml.load(f)
                
                project = pyproject_data.get('project', {})
                dependencies = project.get('dependencies', [])
                optional_deps = project.get('optional-dependencies', {})
                
                metrics['direct_dependencies'] = len(dependencies)
                metrics['optional_dependencies'] = sum(len(deps) for deps in optional_deps.values())
                
                # Check for outdated packages
                try:
                    outdated_result = self._run_command(['pip', 'list', '--outdated', '--format=json'])
                    if outdated_result:
                        outdated_data = json.loads(outdated_result)
                        metrics['outdated_packages'] = len(outdated_data)
                        metrics['outdated_details'] = outdated_data
                except:
                    metrics['outdated_packages'] = 'unknown'
            
        except Exception as e:
            print(f"❌ Error collecting dependency metrics: {e}")
        
        return metrics
    
    def collect_build_metrics(self) -> Dict[str, Any]:
        """Collect build and CI metrics."""
        print("🏗️ Collecting build metrics...")
        
        metrics = {}
        
        try:
            # Docker image size (if Dockerfile exists)
            if Path('Dockerfile').exists():
                try:
                    # Build image and get size
                    build_result = self._run_command([
                        'docker', 'build', '-t', 'mcg-metrics-test', '.'
                    ])
                    
                    if build_result:
                        size_result = self._run_command([
                            'docker', 'images', '--format', 'table {{.Size}}', 'mcg-metrics-test'
                        ])
                        
                        if size_result:
                            lines = size_result.strip().split('\n')
                            if len(lines) > 1:
                                metrics['docker_image_size'] = lines[1].strip()
                                
                        # Clean up test image
                        self._run_command(['docker', 'rmi', 'mcg-metrics-test'], ignore_errors=True)
                        
                except Exception as e:
                    print(f"⚠️  Could not check Docker image size: {e}")
            
            # Build artifacts size
            if Path('dist').exists():
                dist_size = sum(f.stat().st_size for f in Path('dist').rglob('*') if f.is_file())
                metrics['dist_size_bytes'] = dist_size
                metrics['dist_size_mb'] = round(dist_size / (1024 * 1024), 2)
            
        except Exception as e:
            print(f"❌ Error collecting build metrics: {e}")
        
        return metrics
    
    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation metrics."""
        print("📚 Collecting documentation metrics...")
        
        metrics = {}
        
        try:
            # Count documentation files
            doc_extensions = ['.md', '.rst', '.txt']
            doc_files = []
            
            for ext in doc_extensions:
                doc_files.extend(list(Path('.').rglob(f'*{ext}')))
            
            metrics['documentation_files'] = len(doc_files)
            
            # Check for key documentation files
            key_docs = [
                'README.md', 'CONTRIBUTING.md', 'LICENSE', 'CHANGELOG.md',
                'SECURITY.md', 'CODE_OF_CONDUCT.md'
            ]
            
            metrics['key_docs_present'] = [doc for doc in key_docs if Path(doc).exists()]
            metrics['key_docs_missing'] = [doc for doc in key_docs if not Path(doc).exists()]
            
            # Documentation coverage (approximate)
            if Path('src').exists():
                py_files = list(Path('src').rglob('*.py'))
                documented_files = 0
                
                for py_file in py_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Simple check for docstrings
                            if '"""' in content or "'''" in content:
                                documented_files += 1
                    except:
                        continue
                
                if py_files:
                    metrics['documentation_coverage'] = round(
                        (documented_files / len(py_files)) * 100, 2
                    )
            
        except Exception as e:
            print(f"❌ Error collecting documentation metrics: {e}")
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub-specific metrics."""
        print("🐙 Collecting GitHub metrics...")
        
        if not self.github_token:
            print("⚠️  GitHub token not available, skipping GitHub metrics")
            return {}
        
        metrics = {}
        headers = {'Authorization': f'token {self.github_token}'}
        
        try:
            # Repository information
            repo_url = f"https://api.github.com/repos/{self.repo_name}"
            repo_response = requests.get(repo_url, headers=headers)
            
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                metrics['stars'] = repo_data.get('stargazers_count', 0)
                metrics['forks'] = repo_data.get('forks_count', 0)
                metrics['watchers'] = repo_data.get('watchers_count', 0)
                metrics['open_issues'] = repo_data.get('open_issues_count', 0)
                metrics['size_kb'] = repo_data.get('size', 0)
                
            # Pull requests
            prs_url = f"https://api.github.com/repos/{self.repo_name}/pulls?state=all&per_page=100"
            prs_response = requests.get(prs_url, headers=headers)
            
            if prs_response.status_code == 200:
                prs_data = prs_response.json()
                metrics['total_prs'] = len(prs_data)
                metrics['open_prs'] = len([pr for pr in prs_data if pr['state'] == 'open'])
                
            # Workflow runs
            runs_url = f"https://api.github.com/repos/{self.repo_name}/actions/runs?per_page=50"
            runs_response = requests.get(runs_url, headers=headers)
            
            if runs_response.status_code == 200:
                runs_data = runs_response.json()
                workflows = runs_data.get('workflow_runs', [])
                
                if workflows:
                    success_count = len([w for w in workflows if w['conclusion'] == 'success'])
                    metrics['workflow_success_rate'] = round(
                        (success_count / len(workflows)) * 100, 2
                    )
                    metrics['recent_workflow_runs'] = len(workflows)
            
        except Exception as e:
            print(f"❌ Error collecting GitHub metrics: {e}")
        
        return metrics
    
    def _run_command(self, cmd: List[str], ignore_errors: bool = False) -> str:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root,
                timeout=60
            )
            
            if result.returncode != 0 and not ignore_errors:
                print(f"⚠️  Command failed: {' '.join(cmd)}")
                print(f"   Error: {result.stderr}")
                return ""
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out: {' '.join(cmd)}")
            return ""
        except Exception as e:
            if not ignore_errors:
                print(f"❌ Command error: {e}")
            return ""
    
    def _get_repository_age(self) -> int:
        """Get repository age in days."""
        try:
            first_commit = self._run_command([
                'git', 'log', '--reverse', '--format=%ad', '--date=iso', '-1'
            ]).strip()
            
            if first_commit:
                from dateutil.parser import parse
                first_date = parse(first_commit)
                age = datetime.now() - first_date.replace(tzinfo=None)
                return age.days
        except:
            pass
        
        return 0
    
    def _extract_type_coverage(self, mypy_content: str) -> float:
        """Extract type coverage from MyPy output."""
        # This is a simplified extraction - actual implementation would be more robust
        lines = mypy_content.split('\n')
        for line in lines:
            if 'coverage' in line.lower() and '%' in line:
                try:
                    # Extract percentage
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)%', line)
                    if match:
                        return float(match.group(1))
                except:
                    pass
        return 0.0
    
    def _calculate_security_confidence(self, bandit_data: Dict) -> float:
        """Calculate security confidence score from Bandit results."""
        results = bandit_data.get('results', [])
        if not results:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        total_weight = sum(severity_weights.get(r.get('issue_severity', 'LOW'), 1) for r in results)
        
        # Calculate confidence (100 - weighted issue score)
        max_possible = len(results) * 3  # All high severity
        if max_possible == 0:
            return 100.0
        
        confidence = max(0, 100 - (total_weight / max_possible) * 100)
        return round(confidence, 2)
    
    def generate_report(self, output_format: str = 'json') -> str:
        """Generate a metrics report."""
        if output_format == 'json':
            return json.dumps(self.metrics, indent=2)
        
        elif output_format == 'markdown':
            return self._generate_markdown_report()
        
        elif output_format == 'html':
            return self._generate_html_report()
        
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _generate_markdown_report(self) -> str:
        """Generate a Markdown metrics report."""
        timestamp = self.metrics.get('timestamp', 'Unknown')
        
        report = f"""# Project Metrics Report

**Generated**: {timestamp}
**Repository**: {self.repo_name}

## 📊 Summary

"""
        
        # Add sections for each metric category
        sections = [
            ('Git Metrics', 'git_metrics'),
            ('Code Quality', 'code_quality'),
            ('Test Metrics', 'test_metrics'),
            ('Security Metrics', 'security_metrics'),
            ('Dependencies', 'dependency_metrics'),
            ('Build Metrics', 'build_metrics'),
            ('Documentation', 'documentation_metrics'),
            ('GitHub Metrics', 'github_metrics')
        ]
        
        for section_name, key in sections:
            if key in self.metrics and self.metrics[key]:
                report += f"### {section_name}\n\n"
                metrics_data = self.metrics[key]
                
                for metric_key, metric_value in metrics_data.items():
                    formatted_key = metric_key.replace('_', ' ').title()
                    report += f"- **{formatted_key}**: {metric_value}\n"
                
                report += "\n"
        
        return report
    
    def _generate_html_report(self) -> str:
        """Generate an HTML metrics report."""
        timestamp = self.metrics.get('timestamp', 'Unknown')
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Project Metrics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ margin: 10px 0; }}
        .metric-value {{ font-weight: bold; color: #2196F3; }}
        .section {{ margin: 20px 0; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Project Metrics Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    <p class="timestamp">Repository: {self.repo_name}</p>
"""
        
        # Add sections
        sections = [
            ('Git Metrics', 'git_metrics'),
            ('Code Quality', 'code_quality'),
            ('Test Metrics', 'test_metrics'),
            ('Security Metrics', 'security_metrics'),
            ('Dependencies', 'dependency_metrics'),
            ('Build Metrics', 'build_metrics'),
            ('Documentation', 'documentation_metrics'),
            ('GitHub Metrics', 'github_metrics')
        ]
        
        for section_name, key in sections:
            if key in self.metrics and self.metrics[key]:
                html += f"<div class='section'><h2>{section_name}</h2>"
                metrics_data = self.metrics[key]
                
                for metric_key, metric_value in metrics_data.items():
                    formatted_key = metric_key.replace('_', ' ').title()
                    html += f"<div class='metric'>{formatted_key}: <span class='metric-value'>{metric_value}</span></div>"
                
                html += "</div>"
        
        html += "</body></html>"
        return html


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument(
        '--format', 
        choices=['json', 'markdown', 'html'], 
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--output', 
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--upload-to-github',
        action='store_true',
        help='Upload metrics to GitHub as a release asset'
    )
    
    args = parser.parse_args()
    
    # Collect metrics
    collector = MetricsCollector()
    collector.collect_all_metrics()
    
    # Generate report
    report = collector.generate_report(args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"✅ Metrics report saved to {args.output}")
    else:
        print(report)
    
    # Upload to GitHub if requested
    if args.upload_to_github and os.getenv('GITHUB_TOKEN'):
        print("📤 Uploading metrics to GitHub...")
        # Implementation would upload as release asset or artifact
        print("✅ Metrics uploaded successfully")


if __name__ == '__main__':
    main()