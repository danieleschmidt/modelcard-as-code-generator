#!/usr/bin/env python3
"""
Advanced health check script for Model Card Generator.

Provides comprehensive health monitoring including:
- Service health checks
- Dependency health checks
- Resource utilization monitoring
- Performance baseline validation
- Integration status verification
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import subprocess
import os
import socket
import psutil


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "healthy", "warning", "critical"
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: str


class HealthChecker:
    """Comprehensive health checker for Model Card Generator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[HealthCheckResult] = []
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report."""
        print("🔍 Running comprehensive health checks...")
        
        # Core service checks
        self._check_service_endpoint()
        self._check_metrics_endpoint()
        self._check_database_connectivity()
        
        # Dependency checks
        self._check_redis_connectivity()
        self._check_mlflow_integration()
        self._check_wandb_integration()
        
        # Resource checks
        self._check_system_resources()
        self._check_disk_space()
        self._check_network_connectivity()
        
        # Performance checks
        self._check_response_times()
        self._check_error_rates()
        
        # Generate summary report
        return self._generate_report()
    
    def _make_request(self, url: str, timeout: int = 10, headers: Dict = None) -> tuple:
        """Make HTTP request and return response data and timing."""
        start_time = time.perf_counter()
        
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read().decode('utf-8')
                end_time = time.perf_counter()
                return response.status, data, (end_time - start_time) * 1000
        except urllib.error.HTTPError as e:
            end_time = time.perf_counter()
            return e.code, str(e), (end_time - start_time) * 1000
        except Exception as e:
            end_time = time.perf_counter()
            return 0, str(e), (end_time - start_time) * 1000
    
    def _check_service_endpoint(self):
        """Check main service health endpoint."""
        url = f"{self.config['service_url']}/health"
        status_code, response, duration = self._make_request(url)
        
        if status_code == 200:
            try:
                health_data = json.loads(response)
                status = "healthy" if health_data.get("status") == "healthy" else "warning"
                message = f"Service is {health_data.get('status', 'unknown')}"
                details = health_data
            except json.JSONDecodeError:
                status = "warning"
                message = "Service responded but with invalid JSON"
                details = {"response": response[:200]}
        elif status_code == 0:
            status = "critical"
            message = f"Service unreachable: {response}"
            details = {"error": response}
        else:
            status = "warning"
            message = f"Service returned status {status_code}"
            details = {"status_code": status_code, "response": response[:200]}
        
        self.results.append(HealthCheckResult(
            name="service_health",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_metrics_endpoint(self):
        """Check metrics endpoint availability."""
        url = f"{self.config['service_url']}/metrics"
        status_code, response, duration = self._make_request(url)
        
        if status_code == 200:
            # Basic validation of Prometheus metrics format
            if "# HELP" in response and "# TYPE" in response:
                status = "healthy"
                message = "Metrics endpoint is working"
                metric_count = response.count("\n# HELP")
                details = {"metric_count": metric_count}
            else:
                status = "warning"
                message = "Metrics endpoint responds but format may be invalid"
                details = {"response_length": len(response)}
        else:
            status = "warning"
            message = f"Metrics endpoint returned status {status_code}"
            details = {"status_code": status_code}
        
        self.results.append(HealthCheckResult(
            name="metrics_endpoint",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_database_connectivity(self):
        """Check database connectivity if configured."""
        db_url = self.config.get('database_url')
        if not db_url:
            self.results.append(HealthCheckResult(
                name="database",
                status="healthy",
                message="No database configured",
                details={},
                duration_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
            return
        
        url = f"{self.config['service_url']}/health/db"
        status_code, response, duration = self._make_request(url)
        
        if status_code == 200:
            status = "healthy"
            message = "Database is accessible"
            details = {"connection_time_ms": duration}
        else:
            status = "critical"
            message = "Database connection failed"
            details = {"error": response}
        
        self.results.append(HealthCheckResult(
            name="database",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_redis_connectivity(self):
        """Check Redis connectivity."""
        redis_host = self.config.get('redis_host', 'localhost')
        redis_port = self.config.get('redis_port', 6379)
        
        start_time = time.perf_counter()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((redis_host, redis_port))
            sock.close()
            end_time = time.perf_counter()
            
            if result == 0:
                status = "healthy"
                message = "Redis is accessible"
                details = {"host": redis_host, "port": redis_port}
            else:
                status = "warning"
                message = "Redis connection failed"
                details = {"host": redis_host, "port": redis_port, "error": "Connection refused"}
        except Exception as e:
            end_time = time.perf_counter()
            status = "warning"
            message = f"Redis check failed: {str(e)}"
            details = {"error": str(e)}
        
        duration = (end_time - start_time) * 1000
        
        self.results.append(HealthCheckResult(
            name="redis",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_mlflow_integration(self):
        """Check MLflow integration status."""
        mlflow_url = self.config.get('mlflow_url')
        if not mlflow_url:
            self.results.append(HealthCheckResult(
                name="mlflow_integration",
                status="healthy",
                message="MLflow integration not configured",
                details={},
                duration_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
            return
        
        # Check MLflow health
        health_url = f"{mlflow_url}/health"
        status_code, response, duration = self._make_request(health_url, timeout=5)
        
        if status_code == 200:
            status = "healthy"
            message = "MLflow is accessible"
            details = {"mlflow_url": mlflow_url}
        else:
            status = "warning"
            message = f"MLflow health check failed (status: {status_code})"
            details = {"mlflow_url": mlflow_url, "error": response}
        
        self.results.append(HealthCheckResult(
            name="mlflow_integration",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_wandb_integration(self):
        """Check Weights & Biases integration."""
        wandb_api_key = self.config.get('wandb_api_key')
        if not wandb_api_key:
            self.results.append(HealthCheckResult(
                name="wandb_integration",
                status="healthy",
                message="W&B integration not configured",
                details={},
                duration_ms=0,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
            return
        
        # Simple API connectivity check
        api_url = "https://api.wandb.ai/api/v1/viewer"
        headers = {"Authorization": f"Bearer {wandb_api_key}"}
        status_code, response, duration = self._make_request(api_url, headers=headers, timeout=10)
        
        if status_code == 200:
            status = "healthy"
            message = "W&B API is accessible"
            details = {"api_accessible": True}
        elif status_code == 401:
            status = "warning"
            message = "W&B API key is invalid"
            details = {"api_accessible": False, "error": "authentication_failed"}
        else:
            status = "warning"
            message = f"W&B API check failed (status: {status_code})"
            details = {"error": response}
        
        self.results.append(HealthCheckResult(
            name="wandb_integration",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_system_resources(self):
        """Check system resource utilization."""
        start_time = time.perf_counter()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Determine status based on thresholds
            if cpu_percent > 80 or memory_percent > 85:
                status = "critical"
                message = "High resource utilization"
            elif cpu_percent > 60 or memory_percent > 70:
                status = "warning"
                message = "Moderate resource utilization"
            else:
                status = "healthy"
                message = "Resource utilization is normal"
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2)
            }
            
        except Exception as e:
            status = "warning"
            message = f"Resource check failed: {str(e)}"
            details = {"error": str(e)}
        
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000
        
        self.results.append(HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_disk_space(self):
        """Check available disk space."""
        start_time = time.perf_counter()
        
        try:
            disk_usage = psutil.disk_usage('/')
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            if used_percent > 90:
                status = "critical"
                message = "Disk space critically low"
            elif used_percent > 80:
                status = "warning"
                message = "Disk space is getting low"
            else:
                status = "healthy"
                message = "Disk space is adequate"
            
            details = {
                "used_percent": round(used_percent, 2),
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2)
            }
            
        except Exception as e:
            status = "warning"
            message = f"Disk space check failed: {str(e)}"
            details = {"error": str(e)}
        
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000
        
        self.results.append(HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_network_connectivity(self):
        """Check network connectivity to external services."""
        external_hosts = [
            ("google.com", 80),
            ("github.com", 443),
            ("pypi.org", 443)
        ]
        
        results = {}
        all_good = True
        
        for host, port in external_hosts:
            start_time = time.perf_counter()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                end_time = time.perf_counter()
                
                if result == 0:
                    results[host] = {"status": "ok", "duration_ms": (end_time - start_time) * 1000}
                else:
                    results[host] = {"status": "failed", "error": "Connection refused"}
                    all_good = False
            except Exception as e:
                results[host] = {"status": "error", "error": str(e)}
                all_good = False
        
        if all_good:
            status = "healthy"
            message = "Network connectivity is good"
        else:
            status = "warning"
            message = "Some network connectivity issues detected"
        
        self.results.append(HealthCheckResult(
            name="network_connectivity",
            status=status,
            message=message,
            details=results,
            duration_ms=sum(r.get("duration_ms", 0) for r in results.values()),
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_response_times(self):
        """Check service response time performance."""
        test_endpoints = [
            "/health",
            "/metrics",
            "/api/v1/formats"
        ]
        
        response_times = []
        base_url = self.config['service_url']
        
        for endpoint in test_endpoints:
            url = f"{base_url}{endpoint}"
            status_code, _, duration = self._make_request(url, timeout=10)
            
            if status_code == 200:
                response_times.append(duration)
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            if max_response_time > 5000:  # 5 seconds
                status = "critical"
                message = "Response times are very slow"
            elif avg_response_time > 1000:  # 1 second
                status = "warning"
                message = "Response times are slower than expected"
            else:
                status = "healthy"
                message = "Response times are good"
            
            details = {
                "avg_response_time_ms": round(avg_response_time, 2),
                "max_response_time_ms": round(max_response_time, 2),
                "endpoints_tested": len(response_times)
            }
        else:
            status = "critical"
            message = "No endpoints responded successfully"
            details = {"endpoints_tested": 0}
        
        self.results.append(HealthCheckResult(
            name="response_times",
            status=status,
            message=message,
            details=details,
            duration_ms=sum(response_times),
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _check_error_rates(self):
        """Check recent error rates from metrics."""
        metrics_url = f"{self.config['service_url']}/metrics"
        status_code, response, duration = self._make_request(metrics_url)
        
        if status_code != 200:
            self.results.append(HealthCheckResult(
                name="error_rates",
                status="warning",
                message="Could not retrieve metrics for error rate check",
                details={"error": "Metrics endpoint unavailable"},
                duration_ms=duration,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
            return
        
        # Simple parsing of Prometheus metrics
        # In production, you'd use a proper Prometheus client
        error_count = 0
        total_requests = 0
        
        for line in response.split('\n'):
            if line.startswith('http_requests_total{') and 'status="5' in line:
                # Extract count from the line
                try:
                    count = float(line.split()[-1])
                    error_count += count
                except (ValueError, IndexError):
                    pass
            elif line.startswith('http_requests_total{'):
                try:
                    count = float(line.split()[-1])
                    total_requests += count
                except (ValueError, IndexError):
                    pass
        
        if total_requests > 0:
            error_rate = (error_count / total_requests) * 100
            
            if error_rate > 10:
                status = "critical"
                message = f"High error rate: {error_rate:.2f}%"
            elif error_rate > 5:
                status = "warning"
                message = f"Elevated error rate: {error_rate:.2f}%"
            else:
                status = "healthy"
                message = f"Error rate is acceptable: {error_rate:.2f}%"
            
            details = {
                "error_rate_percent": round(error_rate, 2),
                "error_count": error_count,
                "total_requests": total_requests
            }
        else:
            status = "healthy"
            message = "No request metrics available"
            details = {"note": "Service may be newly started"}
        
        self.results.append(HealthCheckResult(
            name="error_rates",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        # Count status types
        status_counts = {"healthy": 0, "warning": 0, "critical": 0}
        for result in self.results:
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts["critical"] > 0:
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Calculate total check time
        total_duration = sum(result.duration_ms for result in self.results)
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "healthy": status_counts["healthy"],
                "warning": status_counts["warning"],
                "critical": status_counts["critical"],
                "total_checks": len(self.results)
            },
            "total_duration_ms": round(total_duration, 2),
            "checks": [asdict(result) for result in self.results]
        }


def main():
    """Main entry point for health check script."""
    parser = argparse.ArgumentParser(
        description="Advanced health check for Model Card Generator"
    )
    parser.add_argument(
        "--service-url",
        default=os.getenv("SERVICE_URL", "http://localhost:8080"),
        help="Base URL for the service"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text", "prometheus"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--config-file",
        help="JSON configuration file with check parameters"
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if any checks fail"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {"service_url": args.service_url}
    
    if args.config_file:
        try:
            with open(args.config_file) as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Add environment variable overrides
    env_vars = {
        'redis_host': os.getenv('REDIS_HOST'),
        'redis_port': int(os.getenv('REDIS_PORT', '6379')),
        'mlflow_url': os.getenv('MLFLOW_URL'),
        'wandb_api_key': os.getenv('WANDB_API_KEY'),
        'database_url': os.getenv('DATABASE_URL')
    }
    
    # Only add non-None values
    config.update({k: v for k, v in env_vars.items() if v is not None})
    
    # Run health checks
    checker = HealthChecker(config)
    report = checker.run_all_checks()
    
    # Output results
    if args.output_format == "json":
        print(json.dumps(report, indent=2))
    elif args.output_format == "prometheus":
        # Output Prometheus metrics format
        print("# HELP health_check_status Health check status (0=healthy, 1=warning, 2=critical)")
        print("# TYPE health_check_status gauge")
        for check in report["checks"]:
            status_value = {"healthy": 0, "warning": 1, "critical": 2}[check["status"]]
            print(f'health_check_status{{check="{check["name"]}"}} {status_value}')
        
        print("# HELP health_check_duration_ms Health check duration in milliseconds")
        print("# TYPE health_check_duration_ms gauge")
        for check in report["checks"]:
            print(f'health_check_duration_ms{{check="{check["name"]}"}} {check["duration_ms"]}')
    else:
        # Text format
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}
        overall_emoji = status_emoji[report["overall_status"]]
        
        print(f"\n{overall_emoji} Overall Health: {report['overall_status'].upper()}")
        print(f"📊 Summary: {report['summary']['healthy']} healthy, {report['summary']['warning']} warnings, {report['summary']['critical']} critical")
        print(f"⏱️ Total check time: {report['total_duration_ms']:.0f}ms")
        print("\n" + "="*60)
        
        for check in report["checks"]:
            emoji = status_emoji[check["status"]]
            print(f"\n{emoji} {check['name']}: {check['message']}")
            print(f"   Duration: {check['duration_ms']:.0f}ms")
            
            if check["details"]:
                for key, value in check["details"].items():
                    if isinstance(value, dict):
                        print(f"   {key}: {json.dumps(value, indent=4)}")
                    else:
                        print(f"   {key}: {value}")
    
    # Exit with appropriate code
    if args.exit_code and report["overall_status"] != "healthy":
        sys.exit(1)


if __name__ == "__main__":
    main()