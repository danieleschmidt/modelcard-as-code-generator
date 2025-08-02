#!/usr/bin/env python3
"""
Automated health check script for Model Card Generator.
Monitors system health and triggers alerts when issues are detected.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import psutil

class HealthChecker:
    """Comprehensive health checking for the Model Card Generator."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path(__file__).parent / 'health-config.json'
        self.config = self._load_config()
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'checks': {},
            'alerts': []
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load health check configuration."""
        default_config = {
            'checks': {
                'system': {
                    'enabled': True,
                    'cpu_threshold': 80,
                    'memory_threshold': 85,
                    'disk_threshold': 90
                },
                'application': {
                    'enabled': True,
                    'response_timeout': 30,
                    'health_endpoint': '/health'
                },
                'dependencies': {
                    'enabled': True,
                    'check_outdated': True,
                    'check_vulnerabilities': True
                },
                'git': {
                    'enabled': True,
                    'check_uncommitted': True,
                    'check_remote_sync': True
                },
                'docker': {
                    'enabled': True,
                    'check_containers': True,
                    'check_images': True
                }
            },
            'alerts': {
                'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
                'email_enabled': False,
                'pagerduty_enabled': False
            },
            'thresholds': {
                'critical_failure_count': 2,
                'warning_failure_count': 1
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    return {**default_config, **user_config}
            except Exception as e:
                print(f"⚠️  Could not load config: {e}, using defaults")
        
        return default_config
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all enabled health checks."""
        print("🏥 Running comprehensive health checks...")
        
        checks = self.config.get('checks', {})
        
        if checks.get('system', {}).get('enabled', True):
            self.results['checks']['system'] = self._check_system_health()
        
        if checks.get('application', {}).get('enabled', True):
            self.results['checks']['application'] = self._check_application_health()
        
        if checks.get('dependencies', {}).get('enabled', True):
            self.results['checks']['dependencies'] = self._check_dependencies()
        
        if checks.get('git', {}).get('enabled', True):
            self.results['checks']['git'] = self._check_git_status()
        
        if checks.get('docker', {}).get('enabled', True):
            self.results['checks']['docker'] = self._check_docker_health()
        
        # Determine overall status
        self._determine_overall_status()
        
        # Generate alerts if needed
        self._generate_alerts()
        
        return self.results
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        print("🖥️  Checking system health...")
        
        system_config = self.config['checks']['system']
        check_result = {
            'status': 'healthy',
            'details': {},
            'issues': []
        }
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            check_result['details']['cpu_usage'] = cpu_percent
            
            if cpu_percent > system_config['cpu_threshold']:
                issue = f"High CPU usage: {cpu_percent}%"
                check_result['issues'].append(issue)
                check_result['status'] = 'warning'
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            check_result['details']['memory_usage'] = memory_percent
            check_result['details']['memory_available_mb'] = memory.available // (1024 * 1024)
            
            if memory_percent > system_config['memory_threshold']:
                issue = f"High memory usage: {memory_percent}%"
                check_result['issues'].append(issue)
                check_result['status'] = 'warning'
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            check_result['details']['disk_usage'] = round(disk_percent, 2)
            check_result['details']['disk_free_gb'] = disk.free // (1024 ** 3)
            
            if disk_percent > system_config['disk_threshold']:
                issue = f"High disk usage: {disk_percent}%"
                check_result['issues'].append(issue)
                check_result['status'] = 'critical'
            
            # Load average (Unix systems)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                check_result['details']['load_average'] = {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
                
                cpu_count = psutil.cpu_count()
                if load_avg[0] > cpu_count * 2:
                    issue = f"High load average: {load_avg[0]} (CPUs: {cpu_count})"
                    check_result['issues'].append(issue)
                    check_result['status'] = 'warning'
            
        except Exception as e:
            check_result['status'] = 'error'
            check_result['issues'].append(f"System check failed: {str(e)}")
        
        return check_result
    
    def _check_application_health(self) -> Dict[str, Any]:
        """Check application health."""
        print("🚀 Checking application health...")
        
        app_config = self.config['checks']['application']
        check_result = {
            'status': 'healthy',
            'details': {},
            'issues': []
        }
        
        try:
            # Check if the application process is running
            app_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'modelcard' in proc.info['name'].lower() or \
                       any('modelcard' in arg.lower() for arg in proc.info['cmdline']):
                        app_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            check_result['details']['running_processes'] = len(app_processes)
            check_result['details']['processes'] = app_processes
            
            # Check HTTP endpoint if configured
            health_endpoint = app_config.get('health_endpoint')
            if health_endpoint:
                try:
                    # Try common ports
                    ports = [8080, 8000, 5000]
                    endpoint_healthy = False
                    
                    for port in ports:
                        try:
                            url = f"http://localhost:{port}{health_endpoint}"
                            response = requests.get(
                                url, 
                                timeout=app_config['response_timeout']
                            )
                            
                            if response.status_code == 200:
                                endpoint_healthy = True
                                check_result['details']['health_endpoint'] = {
                                    'url': url,
                                    'status_code': response.status_code,
                                    'response_time_ms': response.elapsed.total_seconds() * 1000
                                }
                                break
                                
                        except requests.exceptions.ConnectionError:
                            continue
                        except Exception as e:
                            check_result['details']['endpoint_error'] = str(e)
                    
                    if not endpoint_healthy:
                        check_result['issues'].append("Health endpoint not responding")
                        check_result['status'] = 'warning'
                        
                except Exception as e:
                    check_result['issues'].append(f"Endpoint check failed: {str(e)}")
            
            # Check log files for errors
            log_dirs = ['logs', '/var/log', '/tmp']
            error_count = 0
            
            for log_dir in log_dirs:
                log_path = Path(log_dir)
                if log_path.exists():
                    for log_file in log_path.glob('*modelcard*.log'):
                        try:
                            # Check last 100 lines for errors
                            with open(log_file) as f:
                                lines = f.readlines()[-100:]
                                error_count += sum(1 for line in lines 
                                                 if 'ERROR' in line.upper() or 'CRITICAL' in line.upper())
                        except Exception:
                            continue
            
            check_result['details']['recent_errors'] = error_count
            if error_count > 10:
                check_result['issues'].append(f"High error count in logs: {error_count}")
                check_result['status'] = 'warning'
                
        except Exception as e:
            check_result['status'] = 'error'
            check_result['issues'].append(f"Application check failed: {str(e)}")
        
        return check_result
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check dependency health."""
        print("📦 Checking dependencies...")
        
        deps_config = self.config['checks']['dependencies']
        check_result = {
            'status': 'healthy',
            'details': {},
            'issues': []
        }
        
        try:
            if deps_config.get('check_outdated', True):
                # Check for outdated packages
                try:
                    result = subprocess.run(
                        ['pip', 'list', '--outdated', '--format=json'],
                        capture_output=True, text=True, timeout=60
                    )
                    
                    if result.returncode == 0 and result.stdout:
                        outdated = json.loads(result.stdout)
                        check_result['details']['outdated_packages'] = len(outdated)
                        
                        if len(outdated) > 5:
                            check_result['issues'].append(f"Many outdated packages: {len(outdated)}")
                            check_result['status'] = 'warning'
                            
                except Exception as e:
                    check_result['issues'].append(f"Could not check outdated packages: {str(e)}")
            
            if deps_config.get('check_vulnerabilities', True):
                # Check for security vulnerabilities
                try:
                    result = subprocess.run(
                        ['safety', 'check', '--json'],
                        capture_output=True, text=True, timeout=60
                    )
                    
                    if result.stdout:
                        try:
                            vulnerabilities = json.loads(result.stdout)
                            vuln_count = len(vulnerabilities)
                            check_result['details']['vulnerabilities'] = vuln_count
                            
                            if vuln_count > 0:
                                check_result['issues'].append(f"Security vulnerabilities found: {vuln_count}")
                                check_result['status'] = 'critical'
                                
                        except json.JSONDecodeError:
                            # Safety might return non-JSON output for no issues
                            check_result['details']['vulnerabilities'] = 0
                            
                except Exception as e:
                    check_result['issues'].append(f"Could not check vulnerabilities: {str(e)}")
            
            # Check if requirements files exist and are readable
            req_files = ['pyproject.toml', 'requirements.txt', 'requirements-dev.txt']
            missing_files = []
            
            for req_file in req_files:
                if not Path(req_file).exists():
                    missing_files.append(req_file)
            
            if 'pyproject.toml' in missing_files and 'requirements.txt' in missing_files:
                check_result['issues'].append("No dependency files found")
                check_result['status'] = 'warning'
                
        except Exception as e:
            check_result['status'] = 'error'
            check_result['issues'].append(f"Dependency check failed: {str(e)}")
        
        return check_result
    
    def _check_git_status(self) -> Dict[str, Any]:
        """Check Git repository status."""
        print("🔀 Checking Git status...")
        
        git_config = self.config['checks']['git']
        check_result = {
            'status': 'healthy',
            'details': {},
            'issues': []
        }
        
        try:
            # Check if we're in a Git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                check_result['status'] = 'warning'
                check_result['issues'].append("Not in a Git repository")
                return check_result
            
            if git_config.get('check_uncommitted', True):
                # Check for uncommitted changes
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    capture_output=True, text=True
                )
                
                if result.stdout.strip():
                    uncommitted_files = len(result.stdout.strip().split('\n'))
                    check_result['details']['uncommitted_files'] = uncommitted_files
                    
                    if uncommitted_files > 10:
                        check_result['issues'].append(f"Many uncommitted files: {uncommitted_files}")
                        check_result['status'] = 'warning'
            
            if git_config.get('check_remote_sync', True):
                # Check if branch is up to date with remote
                try:
                    # Fetch latest
                    subprocess.run(['git', 'fetch'], capture_output=True, timeout=30)
                    
                    # Check behind/ahead status
                    result = subprocess.run(
                        ['git', 'rev-list', '--count', '--left-right', 'HEAD...@{upstream}'],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        behind, ahead = result.stdout.strip().split('\t')
                        check_result['details']['commits_behind'] = int(behind)
                        check_result['details']['commits_ahead'] = int(ahead)
                        
                        if int(behind) > 5:
                            check_result['issues'].append(f"Branch behind remote: {behind} commits")
                            check_result['status'] = 'warning'
                            
                except Exception as e:
                    check_result['details']['remote_sync_error'] = str(e)
            
            # Get current branch and last commit
            try:
                branch_result = subprocess.run(
                    ['git', 'branch', '--show-current'],
                    capture_output=True, text=True
                )
                if branch_result.returncode == 0:
                    check_result['details']['current_branch'] = branch_result.stdout.strip()
                
                commit_result = subprocess.run(
                    ['git', 'log', '-1', '--format=%H,%ad,%s', '--date=iso'],
                    capture_output=True, text=True
                )
                if commit_result.returncode == 0:
                    check_result['details']['last_commit'] = commit_result.stdout.strip()
                    
            except Exception:
                pass
                
        except Exception as e:
            check_result['status'] = 'error'
            check_result['issues'].append(f"Git check failed: {str(e)}")
        
        return check_result
    
    def _check_docker_health(self) -> Dict[str, Any]:
        """Check Docker health."""
        print("🐳 Checking Docker health...")
        
        docker_config = self.config['checks']['docker']
        check_result = {
            'status': 'healthy',
            'details': {},
            'issues': []
        }
        
        try:
            # Check if Docker is available
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                check_result['status'] = 'warning'
                check_result['issues'].append("Docker not available")
                return check_result
            
            check_result['details']['docker_version'] = result.stdout.strip()
            
            if docker_config.get('check_containers', True):
                # Check running containers
                result = subprocess.run(
                    ['docker', 'ps', '--format', 'json'],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    containers = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            try:
                                containers.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    
                    check_result['details']['running_containers'] = len(containers)
                    
                    # Check for unhealthy containers
                    unhealthy = [c for c in containers if 'unhealthy' in c.get('Status', '').lower()]
                    if unhealthy:
                        check_result['issues'].append(f"Unhealthy containers: {len(unhealthy)}")
                        check_result['status'] = 'warning'
            
            if docker_config.get('check_images', True):
                # Check for dangling images
                result = subprocess.run(
                    ['docker', 'images', '--filter', 'dangling=true', '--format', 'json'],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    dangling_count = len(result.stdout.strip().split('\n'))
                    check_result['details']['dangling_images'] = dangling_count
                    
                    if dangling_count > 10:
                        check_result['issues'].append(f"Many dangling images: {dangling_count}")
                        check_result['status'] = 'warning'
                        
        except Exception as e:
            check_result['status'] = 'error'
            check_result['issues'].append(f"Docker check failed: {str(e)}")
        
        return check_result
    
    def _determine_overall_status(self):
        """Determine overall health status based on individual checks."""
        critical_count = 0
        warning_count = 0
        error_count = 0
        
        for check_name, check_result in self.results['checks'].items():
            status = check_result.get('status', 'unknown')
            
            if status == 'critical':
                critical_count += 1
            elif status == 'warning':
                warning_count += 1
            elif status == 'error':
                error_count += 1
        
        thresholds = self.config.get('thresholds', {})
        critical_threshold = thresholds.get('critical_failure_count', 2)
        warning_threshold = thresholds.get('warning_failure_count', 1)
        
        if critical_count >= critical_threshold or error_count > 0:
            self.results['status'] = 'critical'
        elif warning_count >= warning_threshold:
            self.results['status'] = 'warning'
        else:
            self.results['status'] = 'healthy'
        
        self.results['summary'] = {
            'critical_issues': critical_count,
            'warnings': warning_count,
            'errors': error_count,
            'total_checks': len(self.results['checks'])
        }
    
    def _generate_alerts(self):
        """Generate alerts based on health check results."""
        if self.results['status'] in ['critical', 'warning']:
            alert = {
                'level': self.results['status'],
                'message': f"Health check {self.results['status']} status detected",
                'timestamp': datetime.utcnow().isoformat(),
                'details': self.results['summary']
            }
            
            self.results['alerts'].append(alert)
            
            # Send notifications
            self._send_notifications(alert)
    
    def _send_notifications(self, alert: Dict[str, Any]):
        """Send notifications for alerts."""
        alerts_config = self.config.get('alerts', {})
        
        # Slack notification
        slack_webhook = alerts_config.get('slack_webhook')
        if slack_webhook:
            self._send_slack_notification(slack_webhook, alert)
        
        # Email notification (placeholder)
        if alerts_config.get('email_enabled', False):
            self._send_email_notification(alert)
        
        # PagerDuty notification (placeholder)
        if alerts_config.get('pagerduty_enabled', False):
            self._send_pagerduty_notification(alert)
    
    def _send_slack_notification(self, webhook_url: str, alert: Dict[str, Any]):
        """Send Slack notification."""
        try:
            color = '#ff0000' if alert['level'] == 'critical' else '#ffff00'
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"🚨 Model Card Generator Health Alert",
                    'text': alert['message'],
                    'fields': [
                        {
                            'title': 'Status',
                            'value': alert['level'].upper(),
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert['timestamp'],
                            'short': True
                        },
                        {
                            'title': 'Critical Issues',
                            'value': str(alert['details']['critical_issues']),
                            'short': True
                        },
                        {
                            'title': 'Warnings',
                            'value': str(alert['details']['warnings']),
                            'short': True
                        }
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                print("✅ Slack notification sent")
            else:
                print(f"❌ Failed to send Slack notification: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error sending Slack notification: {e}")
    
    def _send_email_notification(self, alert: Dict[str, Any]):
        """Send email notification (placeholder)."""
        print("📧 Email notification would be sent here")
    
    def _send_pagerduty_notification(self, alert: Dict[str, Any]):
        """Send PagerDuty notification (placeholder)."""
        print("📟 PagerDuty notification would be sent here")
    
    def save_results(self, output_file: Path):
        """Save health check results to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ Health check results saved to {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health checker for Model Card Generator")
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for results'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output, only return exit code'
    )
    
    args = parser.parse_args()
    
    # Run health checks
    checker = HealthChecker(args.config)
    results = checker.run_all_checks()
    
    # Output results
    if not args.quiet:
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n🏥 Health Check Summary")
            print(f"Status: {results['status'].upper()}")
            print(f"Timestamp: {results['timestamp']}")
            print(f"Checks Run: {results['summary']['total_checks']}")
            print(f"Critical Issues: {results['summary']['critical_issues']}")
            print(f"Warnings: {results['summary']['warnings']}")
            print(f"Errors: {results['summary']['errors']}")
            
            if results['summary']['critical_issues'] > 0 or results['summary']['warnings'] > 0:
                print(f"\n⚠️  Issues Found:")
                for check_name, check_result in results['checks'].items():
                    if check_result.get('issues'):
                        print(f"  {check_name}:")
                        for issue in check_result['issues']:
                            print(f"    - {issue}")
    
    # Save results if requested
    if args.output:
        checker.save_results(args.output)
    
    # Exit with appropriate code
    if results['status'] == 'critical':
        sys.exit(2)
    elif results['status'] == 'warning':
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()