"""Security utilities for model card generator."""

import re
import json
import hashlib
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from urllib.parse import urlparse

from .exceptions import SecurityError, ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


class InputSanitizer:
    """Sanitize and validate inputs for security."""
    
    # Regex patterns for validation
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9./_-]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    URL_PATTERN = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
    VERSION_PATTERN = re.compile(r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$')
    
    # Dangerous content patterns
    SCRIPT_PATTERNS = [
        re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'vbscript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
    ]
    
    SQL_INJECTION_PATTERNS = [
        re.compile(r'(union|select|insert|update|delete|drop|create|alter)\s', re.IGNORECASE),
        re.compile(r'[\'";]', re.IGNORECASE),
        re.compile(r'--', re.IGNORECASE),
        re.compile(r'/\*.*?\*/', re.IGNORECASE | re.DOTALL),
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r'[;&|`$()]'),
        re.compile(r'\b(rm|del|format|exec|eval|system)\b', re.IGNORECASE),
    ]
    
    def __init__(self):
        self.blocked_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', 
            '.jar', '.app', '.deb', '.rpm', '.dmg', '.pkg', '.sh', '.ps1'
        }
        self.allowed_extensions = {
            '.md', '.txt', '.json', '.yaml', '.yml', '.csv', '.tsv', 
            '.xml', '.html', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.svg'
        }
    
    def sanitize_string(self, value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")
        
        # Check length
        if len(value) > max_length:
            logger.warning(f"String truncated from {len(value)} to {max_length} characters")
            value = value[:max_length]
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Check for script injection
        if not allow_html:
            for pattern in self.SCRIPT_PATTERNS:
                if pattern.search(value):
                    raise SecurityError("Potential script injection detected", 
                                       vulnerability_type="xss",
                                       details={"pattern": pattern.pattern})
        
        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if pattern.search(value):
                raise SecurityError("Potential SQL injection detected",
                                   vulnerability_type="sql_injection", 
                                   details={"pattern": pattern.pattern})
        
        # Check for command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if pattern.search(value):
                raise SecurityError("Potential command injection detected",
                                   vulnerability_type="command_injection",
                                   details={"pattern": pattern.pattern})
        
        return value.strip()
    
    def validate_filename(self, filename: str) -> str:
        """Validate and sanitize filename."""
        filename = self.sanitize_string(filename, max_length=255)
        
        if not self.SAFE_FILENAME_PATTERN.match(filename):
            raise ValidationError(f"Invalid filename: {filename}")
        
        # Check extension
        path = Path(filename)
        if path.suffix.lower() in self.blocked_extensions:
            raise SecurityError(f"Blocked file extension: {path.suffix}",
                               vulnerability_type="malicious_file")
        
        return filename
    
    def validate_path(self, path: str, base_path: Optional[str] = None) -> str:
        """Validate and sanitize file path."""
        path = self.sanitize_string(path, max_length=4096)
        
        # Normalize path
        normalized = Path(path).resolve()
        
        # Check for path traversal
        if '..' in path or path.startswith('/'):
            raise SecurityError("Path traversal attempt detected",
                               vulnerability_type="path_traversal",
                               details={"path": path})
        
        # Check against base path if provided
        if base_path:
            base_resolved = Path(base_path).resolve()
            try:
                normalized.relative_to(base_resolved)
            except ValueError:
                raise SecurityError("Path outside allowed directory",
                                   vulnerability_type="path_traversal",
                                   details={"path": str(normalized), "base": str(base_resolved)})
        
        return str(normalized)
    
    def validate_email(self, email: str) -> str:
        """Validate email address."""
        email = self.sanitize_string(email, max_length=254).lower()
        
        if not self.EMAIL_PATTERN.match(email):
            raise ValidationError(f"Invalid email format: {email}")
        
        return email
    
    def validate_url(self, url: str) -> str:
        """Validate URL."""
        url = self.sanitize_string(url, max_length=2048)
        
        if not self.URL_PATTERN.match(url):
            raise ValidationError(f"Invalid URL format: {url}")
        
        # Parse and validate components
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            raise ValidationError(f"Unsupported URL scheme: {parsed.scheme}")
        
        return url
    
    def validate_version(self, version: str) -> str:
        """Validate version string."""
        version = self.sanitize_string(version, max_length=50)
        
        if not self.VERSION_PATTERN.match(version):
            raise ValidationError(f"Invalid version format: {version}")
        
        return version
    
    def validate_json(self, data: Union[str, Dict[str, Any]], max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
        """Validate and parse JSON data."""
        if isinstance(data, str):
            if len(data.encode('utf-8')) > max_size:
                raise ValidationError(f"JSON data too large: {len(data)} bytes")
            
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON: {e}")
        else:
            parsed = data
        
        if not isinstance(parsed, dict):
            raise ValidationError("JSON must be an object")
        
        # Recursively sanitize string values
        return self._sanitize_dict(parsed)
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary values."""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = self.sanitize_string(str(key), max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = self.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[clean_key] = self._sanitize_list(value)
            else:
                sanitized[clean_key] = value
        
        return sanitized
    
    def _sanitize_list(self, data: List[Any]) -> List[Any]:
        """Recursively sanitize list values."""
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                sanitized.append(self.sanitize_string(item))
            elif isinstance(item, dict):
                sanitized.append(self._sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(self._sanitize_list(item))
            else:
                sanitized.append(item)
        
        return sanitized


class SecurityScanner:
    """Security scanner for model card content."""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.sensitive_patterns = [
            # API keys and tokens
            re.compile(r'[a-zA-Z0-9]{32,}', re.IGNORECASE),
            # Email addresses
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            # Phone numbers
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            # Credit card numbers
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            # Social security numbers
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        ]
        
        self.vulnerability_checks = [
            self._check_sensitive_data_exposure,
            self._check_injection_vulnerabilities,
            self._check_file_inclusion,
            self._check_malicious_content,
        ]
    
    def scan_content(self, content: str) -> Dict[str, Any]:
        """Scan content for security vulnerabilities."""
        logger.info("Starting security scan")
        
        results = {
            "passed": True,
            "vulnerabilities": [],
            "warnings": [],
            "scan_timestamp": hashlib.sha256(content.encode()).hexdigest()[:16]
        }
        
        for check in self.vulnerability_checks:
            try:
                check_result = check(content)
                if check_result["vulnerabilities"]:
                    results["passed"] = False
                    results["vulnerabilities"].extend(check_result["vulnerabilities"])
                if check_result["warnings"]:
                    results["warnings"].extend(check_result["warnings"])
            except Exception as e:
                logger.error(f"Security check failed: {e}")
                results["vulnerabilities"].append({
                    "type": "scan_error",
                    "severity": "medium",
                    "message": f"Security check failed: {e}"
                })
        
        logger.log_security_check(
            "content_scan",
            results["passed"],
            {"vulnerabilities": len(results["vulnerabilities"]), "warnings": len(results["warnings"])}
        )
        
        return results
    
    def _check_sensitive_data_exposure(self, content: str) -> Dict[str, Any]:
        """Check for potential sensitive data exposure."""
        vulnerabilities = []
        warnings = []
        
        for pattern in self.sensitive_patterns:
            matches = pattern.findall(content)
            if matches:
                # Check if it's likely a real sensitive value (basic heuristics)
                for match in matches:
                    if len(match) > 20:  # Likely API key
                        vulnerabilities.append({
                            "type": "sensitive_data_exposure",
                            "severity": "high",
                            "message": f"Potential API key or token detected",
                            "context": match[:10] + "..."
                        })
                    else:
                        warnings.append({
                            "type": "potential_sensitive_data",
                            "severity": "low",
                            "message": f"Potential sensitive data pattern detected",
                            "context": match[:10] + "..."
                        })
        
        return {"vulnerabilities": vulnerabilities, "warnings": warnings}
    
    def _check_injection_vulnerabilities(self, content: str) -> Dict[str, Any]:
        """Check for injection vulnerabilities."""
        vulnerabilities = []
        warnings = []
        
        # Check for script injection
        for pattern in self.sanitizer.SCRIPT_PATTERNS:
            if pattern.search(content):
                vulnerabilities.append({
                    "type": "script_injection",
                    "severity": "high",
                    "message": "Potential script injection detected"
                })
        
        # Check for SQL injection patterns
        for pattern in self.sanitizer.SQL_INJECTION_PATTERNS:
            if pattern.search(content):
                warnings.append({
                    "type": "sql_injection_pattern",
                    "severity": "medium", 
                    "message": "SQL injection pattern detected"
                })
        
        return {"vulnerabilities": vulnerabilities, "warnings": warnings}
    
    def _check_file_inclusion(self, content: str) -> Dict[str, Any]:
        """Check for file inclusion vulnerabilities."""
        vulnerabilities = []
        warnings = []
        
        file_inclusion_patterns = [
            re.compile(r'\.\./', re.IGNORECASE),
            re.compile(r'file://', re.IGNORECASE),
            re.compile(r'/etc/passwd', re.IGNORECASE),
            re.compile(r'\\\\', re.IGNORECASE),
        ]
        
        for pattern in file_inclusion_patterns:
            if pattern.search(content):
                vulnerabilities.append({
                    "type": "file_inclusion",
                    "severity": "medium",
                    "message": "Potential file inclusion vulnerability detected"
                })
        
        return {"vulnerabilities": vulnerabilities, "warnings": warnings}
    
    def _check_malicious_content(self, content: str) -> Dict[str, Any]:
        """Check for malicious content patterns."""
        vulnerabilities = []
        warnings = []
        
        malicious_patterns = [
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'system\s*\(', re.IGNORECASE),
            re.compile(r'shell_exec\s*\(', re.IGNORECASE),
        ]
        
        for pattern in malicious_patterns:
            if pattern.search(content):
                vulnerabilities.append({
                    "type": "malicious_content",
                    "severity": "high",
                    "message": f"Malicious function call detected: {pattern.pattern}"
                })
        
        return {"vulnerabilities": vulnerabilities, "warnings": warnings}
    
    def generate_security_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate human-readable security report."""
        lines = []
        lines.append("# Security Scan Report")
        lines.append(f"**Status**: {'✅ PASSED' if scan_results['passed'] else '❌ FAILED'}")
        lines.append(f"**Scan ID**: {scan_results['scan_timestamp']}")
        
        if scan_results["vulnerabilities"]:
            lines.append("\n## 🚨 Vulnerabilities Found")
            for vuln in scan_results["vulnerabilities"]:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(vuln["severity"], "⚪")
                lines.append(f"- {severity_emoji} **{vuln['type'].replace('_', ' ').title()}** ({vuln['severity']})")
                lines.append(f"  - {vuln['message']}")
                if "context" in vuln:
                    lines.append(f"  - Context: `{vuln['context']}`")
        
        if scan_results["warnings"]:
            lines.append("\n## ⚠️ Warnings")
            for warning in scan_results["warnings"]:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(warning["severity"], "⚪")
                lines.append(f"- {severity_emoji} **{warning['type'].replace('_', ' ').title()}** ({warning['severity']})")
                lines.append(f"  - {warning['message']}")
                if "context" in warning:
                    lines.append(f"  - Context: `{warning['context']}`")
        
        if scan_results["passed"] and not scan_results["warnings"]:
            lines.append("\n## ✅ All Security Checks Passed")
            lines.append("No vulnerabilities or security warnings detected.")
        
        return "\n".join(lines)


# Global instances
sanitizer = InputSanitizer()
scanner = SecurityScanner()


def sanitize_input(value: Any, **kwargs) -> Any:
    """Sanitize input value based on type."""
    if isinstance(value, str):
        return sanitizer.sanitize_string(value, **kwargs)
    elif isinstance(value, dict):
        return sanitizer.validate_json(value)
    elif isinstance(value, list):
        return sanitizer._sanitize_list(value)
    else:
        return value


def scan_for_vulnerabilities(content: str) -> Dict[str, Any]:
    """Scan content for security vulnerabilities."""
    return scanner.scan_content(content)