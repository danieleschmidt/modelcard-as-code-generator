"""
Health check system for Model Card Generator.

Provides comprehensive health monitoring including:
- Application health status
- Dependency health checks
- Resource availability checks
- Service connectivity tests
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil
import requests


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Optional[Dict[str, Any]] = None


class HealthCheck:
    """Base class for health checks."""

    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout

    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._execute_check(),
                timeout=self.timeout
            )
            duration = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name=self.name,
                status=result.get("status", HealthStatus.UNKNOWN),
                message=result.get("message", "Check completed"),
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration,
                metadata=result.get("metadata")
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout}s",
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                duration_ms=duration
            )

    async def _execute_check(self) -> Dict[str, Any]:
        """Override this method to implement the actual check."""
        raise NotImplementedError


class ApplicationHealthCheck(HealthCheck):
    """Basic application health check."""

    def __init__(self):
        super().__init__("application")

    async def _execute_check(self) -> Dict[str, Any]:
        """Check basic application health."""
        try:
            # Check if we can import core modules
            # Check basic functionality
            import tempfile

            from modelcard_generator import __version__
            with tempfile.NamedTemporaryFile() as f:
                f.write(b"health check")
                f.flush()

            return {
                "status": HealthStatus.HEALTHY,
                "message": "Application is healthy",
                "metadata": {
                    "version": __version__,
                    "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Application health check failed: {str(e)}"
            }


class ResourceHealthCheck(HealthCheck):
    """System resource health check."""

    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0, disk_threshold: float = 90.0):
        super().__init__("resources")
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def _execute_check(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            # Determine status
            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.WARNING if cpu_percent < 95 else HealthStatus.CRITICAL
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            if memory_percent > self.memory_threshold:
                status = HealthStatus.WARNING if memory_percent < 95 else HealthStatus.CRITICAL
                issues.append(f"High memory usage: {memory_percent:.1f}%")

            if disk_percent > self.disk_threshold:
                status = HealthStatus.WARNING if disk_percent < 95 else HealthStatus.CRITICAL
                issues.append(f"High disk usage: {disk_percent:.1f}%")

            message = "Resource usage is normal"
            if issues:
                message = f"Resource issues detected: {', '.join(issues)}"

            return {
                "status": status,
                "message": message,
                "metadata": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_total_gb": memory.total / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_total_gb": disk.total / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Resource check failed: {str(e)}"
            }


class FileSystemHealthCheck(HealthCheck):
    """File system health check."""

    def __init__(self, paths: List[str]):
        super().__init__("filesystem")
        self.paths = paths

    async def _execute_check(self) -> Dict[str, Any]:
        """Check file system access and permissions."""
        try:
            issues = []
            checked_paths = {}

            for path_str in self.paths:
                path = Path(path_str)

                try:
                    # Check if path exists
                    exists = path.exists()

                    # Check if we can read
                    readable = path.is_dir() and path.stat().st_mode & 0o444 if exists else False

                    # Check if we can write (for directories)
                    writable = False
                    if exists and path.is_dir():
                        try:
                            test_file = path / ".health_check_test"
                            test_file.touch()
                            test_file.unlink()
                            writable = True
                        except:
                            writable = False

                    checked_paths[str(path)] = {
                        "exists": exists,
                        "readable": readable,
                        "writable": writable
                    }

                    if not exists:
                        issues.append(f"Path does not exist: {path}")
                    elif not readable:
                        issues.append(f"Cannot read path: {path}")
                    elif path.is_dir() and not writable:
                        issues.append(f"Cannot write to directory: {path}")

                except Exception as e:
                    issues.append(f"Error checking path {path}: {str(e)}")
                    checked_paths[str(path)] = {"error": str(e)}

            status = HealthStatus.HEALTHY if not issues else HealthStatus.WARNING
            message = "All paths accessible" if not issues else f"Path issues: {', '.join(issues)}"

            return {
                "status": status,
                "message": message,
                "metadata": {
                    "checked_paths": checked_paths,
                    "total_paths": len(self.paths),
                    "accessible_paths": len([p for p in checked_paths.values() if p.get("exists", False)])
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Filesystem check failed: {str(e)}"
            }


class ExternalServiceHealthCheck(HealthCheck):
    """External service connectivity health check."""

    def __init__(self, services: Dict[str, str]):
        super().__init__("external_services")
        self.services = services  # name -> URL mapping

    async def _execute_check(self) -> Dict[str, Any]:
        """Check connectivity to external services."""
        try:
            service_status = {}
            failed_services = []

            for service_name, url in self.services.items():
                try:
                    response = requests.get(url, timeout=5)
                    is_healthy = response.status_code < 400

                    service_status[service_name] = {
                        "url": url,
                        "status_code": response.status_code,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "healthy": is_healthy
                    }

                    if not is_healthy:
                        failed_services.append(f"{service_name} ({response.status_code})")

                except requests.RequestException as e:
                    service_status[service_name] = {
                        "url": url,
                        "error": str(e),
                        "healthy": False
                    }
                    failed_services.append(f"{service_name} (connection error)")

            status = HealthStatus.HEALTHY if not failed_services else HealthStatus.WARNING
            message = "All services accessible" if not failed_services else f"Service issues: {', '.join(failed_services)}"

            return {
                "status": status,
                "message": message,
                "metadata": {
                    "services": service_status,
                    "total_services": len(self.services),
                    "healthy_services": len([s for s in service_status.values() if s.get("healthy", False)])
                }
            }

        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"External service check failed: {str(e)}"
            }


class HealthChecker:
    """Main health checker that orchestrates all health checks."""

    def __init__(self):
        self.checks: List[HealthCheck] = []
        self.last_results: Dict[str, HealthCheckResult] = {}

    def register_check(self, check: HealthCheck) -> None:
        """Register a health check."""
        self.checks.append(check)

    def register_default_checks(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Register default health checks."""
        config = config or {}

        # Application health
        self.register_check(ApplicationHealthCheck())

        # Resource health
        resource_config = config.get("resources", {})
        self.register_check(ResourceHealthCheck(
            cpu_threshold=resource_config.get("cpu_threshold", 90.0),
            memory_threshold=resource_config.get("memory_threshold", 90.0),
            disk_threshold=resource_config.get("disk_threshold", 90.0)
        ))

        # File system health
        filesystem_config = config.get("filesystem", {})
        paths = filesystem_config.get("paths", ["/tmp", "output", "cache"])
        self.register_check(FileSystemHealthCheck(paths))

        # External services health
        services_config = config.get("external_services", {})
        if services_config:
            self.register_check(ExternalServiceHealthCheck(services_config))

    async def check_health(self, check_names: Optional[List[str]] = None) -> Dict[str, HealthCheckResult]:
        """Run health checks and return results."""
        checks_to_run = self.checks

        if check_names:
            checks_to_run = [c for c in self.checks if c.name in check_names]

        # Run all checks concurrently
        tasks = [check.check() for check in checks_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        health_results = {}
        for i, result in enumerate(results):
            check = checks_to_run[i]

            if isinstance(result, Exception):
                # Handle exceptions from checks
                health_results[check.name] = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check exception: {str(result)}",
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=0.0
                )
            else:
                health_results[check.name] = result

        # Update last results
        self.last_results.update(health_results)

        return health_results

    def get_overall_status(self, results: Optional[Dict[str, HealthCheckResult]] = None) -> HealthStatus:
        """Get overall health status from check results."""
        if results is None:
            results = self.last_results

        if not results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_summary(self, results: Optional[Dict[str, HealthCheckResult]] = None) -> Dict[str, Any]:
        """Get a summary of health check results."""
        if results is None:
            results = self.last_results

        overall_status = self.get_overall_status(results)

        summary = {
            "overall_status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in results.items()
            },
            "statistics": {
                "total_checks": len(results),
                "healthy_checks": len([r for r in results.values() if r.status == HealthStatus.HEALTHY]),
                "warning_checks": len([r for r in results.values() if r.status == HealthStatus.WARNING]),
                "critical_checks": len([r for r in results.values() if r.status == HealthStatus.CRITICAL]),
                "average_duration_ms": sum(r.duration_ms for r in results.values()) / len(results) if results else 0
            }
        }

        return summary


# Convenience functions
async def check_application_health(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to check application health."""
    checker = HealthChecker()
    checker.register_default_checks(config)

    results = await checker.check_health()
    return checker.get_health_summary(results)


def create_health_check_endpoint(checker: HealthChecker) -> Callable:
    """Create a health check endpoint for web frameworks."""
    async def health_endpoint():
        try:
            results = await checker.check_health()
            summary = checker.get_health_summary(results)

            # Return appropriate HTTP status code
            status_code = 200
            if summary["overall_status"] == "warning":
                status_code = 200  # Still operational
            elif summary["overall_status"] == "critical":
                status_code = 503  # Service unavailable
            elif summary["overall_status"] == "unknown":
                status_code = 500  # Internal server error

            return summary, status_code

        except Exception as e:
            return {
                "overall_status": "critical",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }, 500

    return health_endpoint
