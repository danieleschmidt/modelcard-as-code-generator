"""
Secret scanning and security validation for Model Card Generator.

This module provides comprehensive security scanning including:
- Secret detection in text content
- Sensitive information identification
- Security policy validation
- Compliance checking
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SecurityFinding:
    """Represents a security finding from scanning."""
    type: str
    severity: str  # low, medium, high, critical
    message: str
    location: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[str] = None
    remediation: Optional[str] = None


class SecretPattern:
    """Represents a pattern for detecting secrets."""

    def __init__(self, name: str, pattern: str, description: str, severity: str = "high"):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        self.description = description
        self.severity = severity


class SecretScanner:
    """Comprehensive secret scanner for model card content."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = self._initialize_patterns()
        self.whitelist_patterns = self._initialize_whitelist()

    def _initialize_patterns(self) -> List[SecretPattern]:
        """Initialize secret detection patterns."""
        return [
            # API Keys
            SecretPattern(
                "aws_access_key",
                r"AKIA[0-9A-Z]{16}",
                "AWS Access Key ID",
                "critical"
            ),
            SecretPattern(
                "aws_secret_key",
                r"[A-Za-z0-9/+=]{40}",
                "AWS Secret Access Key",
                "critical"
            ),
            SecretPattern(
                "google_api_key",
                r"AIza[0-9A-Za-z\\-_]{35}",
                "Google API Key",
                "high"
            ),
            SecretPattern(
                "github_token",
                r"gh[ps]_[a-zA-Z0-9]{36}",
                "GitHub Personal Access Token",
                "high"
            ),
            SecretPattern(
                "slack_token",
                r"xox[baprs]-([0-9a-zA-Z]{10,48})?",
                "Slack Token",
                "medium"
            ),

            # Database Credentials
            SecretPattern(
                "connection_string",
                r"(mongodb|mysql|postgresql|redis)://[^\\s]+",
                "Database Connection String",
                "high"
            ),
            SecretPattern(
                "sql_password",
                r"(?i)(password|pwd|passwd)\\s*[=:]\\s*['\"][^'\"]{3,}['\"]",
                "SQL Password",
                "high"
            ),

            # Generic Secrets
            SecretPattern(
                "api_key_generic",
                r"(?i)(api[_-]?key|apikey)\\s*[=:]\\s*['\"][a-zA-Z0-9]{10,}['\"]",
                "Generic API Key",
                "medium"
            ),
            SecretPattern(
                "secret_key_generic",
                r"(?i)(secret[_-]?key|secretkey)\\s*[=:]\\s*['\"][a-zA-Z0-9]{10,}['\"]",
                "Generic Secret Key",
                "medium"
            ),
            SecretPattern(
                "private_key",
                r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----",
                "Private Key",
                "critical"
            ),

            # URLs with credentials
            SecretPattern(
                "url_with_credentials",
                r"https?://[^\\s:@]+:[^\\s:@]+@[^\\s]+",
                "URL with embedded credentials",
                "high"
            ),

            # JWT Tokens
            SecretPattern(
                "jwt_token",
                r"eyJ[a-zA-Z0-9_-]+\\.eyJ[a-zA-Z0-9_-]+\\.[a-zA-Z0-9_-]+",
                "JWT Token",
                "medium"
            ),

            # Encryption Keys
            SecretPattern(
                "encryption_key",
                r"(?i)(encryption[_-]?key|encrypt[_-]?key)\\s*[=:]\\s*['\"][a-zA-Z0-9+/=]{20,}['\"]",
                "Encryption Key",
                "high"
            ),

            # ML Platform Tokens
            SecretPattern(
                "wandb_key",
                r"[a-f0-9]{40}",  # W&B API keys are 40 char hex
                "Weights & Biases API Key",
                "medium"
            ),
            SecretPattern(
                "huggingface_token",
                r"hf_[a-zA-Z0-9]{34}",
                "Hugging Face Token",
                "medium"
            ),
        ]

    def _initialize_whitelist(self) -> List[Pattern]:
        """Initialize patterns for whitelisting false positives."""
        return [
            re.compile(r"example\.com", re.IGNORECASE),
            re.compile(r"your[_-]?key[_-]?here", re.IGNORECASE),
            re.compile(r"placeholder", re.IGNORECASE),
            re.compile(r"test[_-]?key", re.IGNORECASE),
            re.compile(r"dummy[_-]?key", re.IGNORECASE),
            re.compile(r"fake[_-]?key", re.IGNORECASE),
            re.compile(r"sample[_-]?key", re.IGNORECASE),
            re.compile(r"{{\\s*[^}]+\\s*}}", re.IGNORECASE),  # Template variables
            re.compile(r"\\$\\{[^}]+\\}", re.IGNORECASE),  # Environment variables
            re.compile(r"<[^>]+>", re.IGNORECASE),  # XML/HTML tags
        ]

    def scan_text(self, text: str, source_name: str = "content") -> List[SecurityFinding]:
        """Scan text content for secrets and sensitive information."""
        findings = []
        lines = text.split("\\n")

        for line_num, line in enumerate(lines, 1):
            for pattern in self.patterns:
                matches = pattern.pattern.finditer(line)

                for match in matches:
                    # Check if this is a whitelisted false positive
                    if self._is_whitelisted(match.group()):
                        continue

                    # Create security finding
                    finding = SecurityFinding(
                        type=f"secret_{pattern.name}",
                        severity=pattern.severity,
                        message=f"Potential {pattern.description} detected",
                        location=source_name,
                        line_number=line_num,
                        column_number=match.start(),
                        context=self._get_context(lines, line_num - 1),
                        remediation=self._get_remediation(pattern.name)
                    )
                    findings.append(finding)

        return findings

    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a file for secrets and sensitive information."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return self.scan_text(content, str(file_path))

        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
            return [SecurityFinding(
                type="scan_error",
                severity="medium",
                message=f"Failed to scan file: {e}",
                location=str(file_path)
            )]

    def scan_directory(self, directory_path: Path, file_extensions: Optional[List[str]] = None) -> List[SecurityFinding]:
        """Scan a directory recursively for secrets."""
        if file_extensions is None:
            file_extensions = [".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".txt", ".env"]

        findings = []

        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                file_findings = self.scan_file(file_path)
                findings.extend(file_findings)

        return findings

    def scan_model_card_data(self, data: Dict[str, Any]) -> List[SecurityFinding]:
        """Scan model card data structure for secrets."""
        findings = []

        # Convert data to JSON string for scanning
        try:
            json_content = json.dumps(data, indent=2)
            findings.extend(self.scan_text(json_content, "model_card_data"))
        except Exception as e:
            findings.append(SecurityFinding(
                type="data_scan_error",
                severity="medium",
                message=f"Failed to scan model card data: {e}",
                location="model_card_data"
            ))

        # Additional checks for model card specific concerns
        findings.extend(self._check_model_card_security(data))

        return findings

    def _check_model_card_security(self, data: Dict[str, Any]) -> List[SecurityFinding]:
        """Perform model card specific security checks."""
        findings = []

        # Check for PII in model descriptions
        pii_patterns = [
            (r"\\b\\d{3}-\\d{2}-\\d{4}\\b", "Social Security Number"),
            (r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b", "Email Address"),
            (r"\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\s?\\d{4}\\b", "Credit Card Number"),
            (r"\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b", "IP Address"),
        ]

        text_content = json.dumps(data, indent=2)

        for pattern, pii_type in pii_patterns:
            if re.search(pattern, text_content):
                findings.append(SecurityFinding(
                    type="pii_exposure",
                    severity="high",
                    message=f"Potential {pii_type} detected in model card",
                    location="model_card_content",
                    remediation=f"Remove or anonymize {pii_type} from model card"
                ))

        # Check for internal URLs or paths
        internal_patterns = [
            r"(?i)(localhost|127\\.0\\.0\\.1|192\\.168\\.|10\\.|172\\.(1[6-9]|2[0-9]|3[01])\\.))",
            r"(?i)(file://|/home/|/users/|c:\\\\|d:\\\\)",
            r"(?i)\\.internal\\b",
        ]

        for pattern in internal_patterns:
            if re.search(pattern, text_content):
                findings.append(SecurityFinding(
                    type="internal_reference",
                    severity="medium",
                    message="Internal URL or path detected in model card",
                    location="model_card_content",
                    remediation="Replace internal references with public equivalents"
                ))

        return findings

    def _is_whitelisted(self, text: str) -> bool:
        """Check if detected text matches whitelist patterns."""
        for pattern in self.whitelist_patterns:
            if pattern.search(text):
                return True
        return False

    def _get_context(self, lines: List[str], line_index: int, context_lines: int = 2) -> str:
        """Get context around a finding."""
        start = max(0, line_index - context_lines)
        end = min(len(lines), line_index + context_lines + 1)

        context_lines_with_numbers = []
        for i in range(start, end):
            marker = ">>> " if i == line_index else "    "
            context_lines_with_numbers.append(f"{marker}{i+1:3d}: {lines[i]}")

        return "\\n".join(context_lines_with_numbers)

    def _get_remediation(self, pattern_name: str) -> str:
        """Get remediation advice for a specific pattern."""
        remediations = {
            "aws_access_key": "Move AWS credentials to environment variables or AWS IAM roles",
            "aws_secret_key": "Use AWS IAM roles or store in AWS Secrets Manager",
            "google_api_key": "Store API keys in environment variables or secret management system",
            "github_token": "Use GitHub Secrets or environment variables",
            "connection_string": "Use environment variables or secret management for database credentials",
            "private_key": "Store private keys in secure key management systems",
            "jwt_token": "Avoid logging or exposing JWT tokens in configuration",
            "encryption_key": "Use secure key management systems for encryption keys",
        }

        return remediations.get(pattern_name, "Store sensitive information in environment variables or secret management systems")

    def generate_report(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Generate a comprehensive security scan report."""
        if not findings:
            return {
                "status": "clean",
                "summary": "No security issues found",
                "findings_count": 0,
                "severity_breakdown": {},
                "findings": []
            }

        # Count findings by severity
        severity_counts = {}
        for finding in findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

        # Determine overall status
        if any(f.severity == "critical" for f in findings):
            status = "critical"
        elif any(f.severity == "high" for f in findings):
            status = "high_risk"
        elif any(f.severity == "medium" for f in findings):
            status = "medium_risk"
        else:
            status = "low_risk"

        return {
            "status": status,
            "summary": f"Found {len(findings)} security issue(s)",
            "findings_count": len(findings),
            "severity_breakdown": severity_counts,
            "findings": [
                {
                    "type": f.type,
                    "severity": f.severity,
                    "message": f.message,
                    "location": f.location,
                    "line_number": f.line_number,
                    "remediation": f.remediation
                }
                for f in findings
            ],
            "recommendations": self._generate_recommendations(findings)
        }

    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = set()

        has_secrets = any("secret" in f.type for f in findings)
        has_pii = any("pii" in f.type for f in findings)
        has_internal_refs = any("internal" in f.type for f in findings)

        if has_secrets:
            recommendations.add("Implement a secret management system (e.g., HashiCorp Vault, AWS Secrets Manager)")
            recommendations.add("Use environment variables for sensitive configuration")
            recommendations.add("Implement secret scanning in CI/CD pipelines")

        if has_pii:
            recommendations.add("Review data handling policies for PII protection")
            recommendations.add("Implement data anonymization techniques")
            recommendations.add("Ensure GDPR/CCPA compliance for personal data")

        if has_internal_refs:
            recommendations.add("Replace internal references with public equivalents")
            recommendations.add("Review documentation for information disclosure")

        if findings:
            recommendations.add("Regular security audits and scans")
            recommendations.add("Security training for development team")

        return list(recommendations)


# Convenience functions
def scan_for_secrets(content: str, source_name: str = "content") -> List[SecurityFinding]:
    """Convenience function to scan content for secrets."""
    scanner = SecretScanner()
    return scanner.scan_text(content, source_name)


def scan_model_card_for_secrets(model_card_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to scan model card data and return a report."""
    scanner = SecretScanner()
    findings = scanner.scan_model_card_data(model_card_data)
    return scanner.generate_report(findings)


def is_content_safe(content: str) -> Tuple[bool, List[SecurityFinding]]:
    """Check if content is safe (no critical security issues)."""
    findings = scan_for_secrets(content)
    has_critical = any(f.severity == "critical" for f in findings)
    has_high = any(f.severity == "high" for f in findings)

    is_safe = not (has_critical or has_high)
    return is_safe, findings
