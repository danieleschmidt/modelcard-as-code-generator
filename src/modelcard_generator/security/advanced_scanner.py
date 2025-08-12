"""Advanced security scanning and threat detection for model card generation."""

import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: str
    severity: str  # low, medium, high, critical
    description: str
    location: str
    evidence: str
    timestamp: datetime = field(default_factory=datetime.now)
    remediation: Optional[str] = None


@dataclass
class SecurityScanResult:
    """Results of a security scan."""
    scan_type: str
    timestamp: datetime
    threats: List[SecurityThreat]
    safe: bool
    scan_duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_threats(self) -> List[SecurityThreat]:
        """Get critical threats."""
        return [t for t in self.threats if t.severity == "critical"]

    @property
    def high_threats(self) -> List[SecurityThreat]:
        """Get high severity threats."""
        return [t for t in self.threats if t.severity == "high"]


class ContentSecurityScanner:
    """Advanced content security scanner with ML-based detection."""

    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.safe_domains = self._load_safe_domains()
        self.suspicious_encodings = ["base64", "hex", "url", "rot13"]

    def _load_threat_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security threat patterns."""
        return {
            "injection": [
                {
                    "pattern": r"<script[^>]*>.*?</script>",
                    "description": "Potential XSS injection",
                    "severity": "high",
                    "flags": re.IGNORECASE | re.DOTALL
                },
                {
                    "pattern": r"javascript:",
                    "description": "JavaScript protocol in URL",
                    "severity": "medium",
                    "flags": re.IGNORECASE
                },
                {
                    "pattern": r"on\w+\s*=",
                    "description": "HTML event handler",
                    "severity": "medium",
                    "flags": re.IGNORECASE
                }
            ],
            "secrets": [
                {
                    "pattern": r"(?i)(api[_-]?key|secret|token|password|pwd)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{16,})",
                    "description": "Potential API key or secret",
                    "severity": "critical",
                    "flags": re.IGNORECASE
                },
                {
                    "pattern": r"sk-[a-zA-Z0-9]{48}",
                    "description": "OpenAI API key",
                    "severity": "critical",
                    "flags": 0
                },
                {
                    "pattern": r"ghp_[a-zA-Z0-9]{36}",
                    "description": "GitHub Personal Access Token",
                    "severity": "critical",
                    "flags": 0
                }
            ],
            "malicious_urls": [
                {
                    "pattern": r"https?://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",
                    "description": "Direct IP URL (potentially suspicious)",
                    "severity": "medium",
                    "flags": re.IGNORECASE
                },
                {
                    "pattern": r"https?://[^/]+\.(?:tk|ml|ga|cf|onion)",
                    "description": "Suspicious TLD",
                    "severity": "medium",
                    "flags": re.IGNORECASE
                }
            ],
            "code_injection": [
                {
                    "pattern": r"(?:exec|eval|subprocess|os\.system|__import__)\s*\(",
                    "description": "Dangerous code execution",
                    "severity": "critical",
                    "flags": re.IGNORECASE
                }
            ]
        }

    def _load_safe_domains(self) -> Set[str]:
        """Load list of known safe domains."""
        return {
            "github.com", "huggingface.co", "arxiv.org", "tensorflow.org",
            "pytorch.org", "scikit-learn.org", "numpy.org", "scipy.org",
            "matplotlib.org", "jupyter.org", "anaconda.org", "pypi.org"
        }

    def scan_content(self, content: str, content_type: str = "text") -> SecurityScanResult:
        """Perform comprehensive content security scan."""
        start_time = datetime.now()
        threats = []

        # Pattern-based scanning
        for category, patterns in self.threat_patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config["pattern"]
                flags = pattern_config.get("flags", 0)
                matches = re.finditer(pattern, content, flags)

                for match in matches:
                    threat = SecurityThreat(
                        threat_type=category,
                        severity=pattern_config["severity"],
                        description=pattern_config["description"],
                        location=f"Position {match.start()}-{match.end()}",
                        evidence=match.group(0)[:100] + "..." if len(match.group(0)) > 100 else match.group(0)
                    )
                    threats.append(threat)

        # Encoding detection
        encoding_threats = self._detect_suspicious_encodings(content)
        threats.extend(encoding_threats)

        # URL validation
        url_threats = self._validate_urls(content)
        threats.extend(url_threats)

        # File extension checks
        if content_type == "filename":
            file_threats = self._check_file_security(content)
            threats.extend(file_threats)

        duration = (datetime.now() - start_time).total_seconds() * 1000

        return SecurityScanResult(
            scan_type="content_security",
            timestamp=start_time,
            threats=threats,
            safe=len([t for t in threats if t.severity in ["critical", "high"]]) == 0,
            scan_duration_ms=duration,
            metadata={"content_length": len(content), "content_type": content_type}
        )

    def _detect_suspicious_encodings(self, content: str) -> List[SecurityThreat]:
        """Detect potentially malicious encoded content."""
        threats = []

        # Base64 detection
        base64_pattern = r"[A-Za-z0-9+/]{20,}={0,2}"
        for match in re.finditer(base64_pattern, content):
            encoded = match.group(0)
            try:
                decoded = base64.b64decode(encoded + "==").decode("utf-8", errors="ignore")
                # Check if decoded content contains suspicious patterns
                if any(keyword in decoded.lower() for keyword in ["script", "eval", "exec", "import"]):
                    threats.append(SecurityThreat(
                        threat_type="encoding",
                        severity="high",
                        description="Suspicious base64 encoded content",
                        location=f"Position {match.start()}-{match.end()}",
                        evidence=encoded[:50] + "..." if len(encoded) > 50 else encoded
                    ))
            except Exception:
                pass

        return threats

    def _validate_urls(self, content: str) -> List[SecurityThreat]:
        """Validate URLs in content for security risks."""
        threats = []

        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        for match in re.finditer(url_pattern, content):
            url = match.group(0)
            parsed = urlparse(url)

            # Check against safe domains
            if parsed.netloc and parsed.netloc.lower() not in self.safe_domains:
                threats.append(SecurityThreat(
                    threat_type="url",
                    severity="low",
                    description="URL from non-whitelisted domain",
                    location=f"Position {match.start()}-{match.end()}",
                    evidence=url,
                    remediation="Verify URL is safe before including"
                ))

        return threats

    def _check_file_security(self, filename: str) -> List[SecurityThreat]:
        """Check file security based on extension and name."""
        threats = []

        dangerous_extensions = {".exe", ".bat", ".cmd", ".com", ".scr", ".vbs", ".js", ".jar", ".app"}
        path = Path(filename)

        if path.suffix.lower() in dangerous_extensions:
            threats.append(SecurityThreat(
                threat_type="file",
                severity="critical",
                description=f"Dangerous file extension: {path.suffix}",
                location=filename,
                evidence=str(path),
                remediation="Remove executable files or store in secure location"
            ))

        # Check for hidden/system file indicators
        if filename.startswith(".") and not filename.startswith("./"):
            threats.append(SecurityThreat(
                threat_type="file",
                severity="low",
                description="Hidden file detected",
                location=filename,
                evidence=str(path)
            ))

        return threats


class ModelSecurityValidator:
    """Validate model-specific security aspects."""

    def __init__(self):
        self.risk_indicators = self._load_risk_indicators()

    def _load_risk_indicators(self) -> Dict[str, List[str]]:
        """Load model security risk indicators."""
        return {
            "bias_keywords": [
                "racial", "gender", "ethnic", "religious", "demographic",
                "stereotyp", "discriminat", "prejudice", "biased"
            ],
            "privacy_keywords": [
                "personal", "private", "confidential", "pii", "gdpr",
                "sensitive", "medical", "financial", "biometric"
            ],
            "harmful_keywords": [
                "weapon", "violence", "illegal", "fraud", "malware",
                "terrorism", "extremist", "hate speech"
            ]
        }

    def validate_model_card(self, model_card_content: str) -> SecurityScanResult:
        """Validate model card for security and ethical concerns."""
        start_time = datetime.now()
        threats = []

        # Check for missing ethical considerations
        if not self._has_ethical_considerations(model_card_content):
            threats.append(SecurityThreat(
                threat_type="ethics",
                severity="medium",
                description="Missing ethical considerations section",
                location="Document structure",
                evidence="No ethical considerations found",
                remediation="Add comprehensive ethical considerations section"
            ))

        # Check for bias-related content
        bias_threats = self._analyze_bias_content(model_card_content)
        threats.extend(bias_threats)

        # Check for privacy concerns
        privacy_threats = self._analyze_privacy_content(model_card_content)
        threats.extend(privacy_threats)

        # Validate training data information
        data_threats = self._validate_training_data_info(model_card_content)
        threats.extend(data_threats)

        duration = (datetime.now() - start_time).total_seconds() * 1000

        return SecurityScanResult(
            scan_type="model_security",
            timestamp=start_time,
            threats=threats,
            safe=len([t for t in threats if t.severity in ["critical", "high"]]) == 0,
            scan_duration_ms=duration
        )

    def _has_ethical_considerations(self, content: str) -> bool:
        """Check if content has ethical considerations."""
        ethical_patterns = [
            r"ethical?\s+consider",
            r"bias",
            r"fairness",
            r"limitation",
            r"risk",
            r"harm"
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in ethical_patterns)

    def _analyze_bias_content(self, content: str) -> List[SecurityThreat]:
        """Analyze content for bias-related concerns."""
        threats = []

        # Check if bias is mentioned but not addressed
        has_bias_mention = any(keyword in content.lower() for keyword in self.risk_indicators["bias_keywords"])
        has_bias_mitigation = re.search(r"bias\s+(mitigation|reduction|handling)", content, re.IGNORECASE)

        if has_bias_mention and not has_bias_mitigation:
            threats.append(SecurityThreat(
                threat_type="bias",
                severity="medium",
                description="Bias mentioned but no mitigation strategies described",
                location="Document content",
                evidence="Bias keywords found without mitigation discussion",
                remediation="Add bias mitigation strategies and fairness analysis"
            ))

        return threats

    def _analyze_privacy_content(self, content: str) -> List[SecurityThreat]:
        """Analyze content for privacy concerns."""
        threats = []

        privacy_mentions = sum(1 for keyword in self.risk_indicators["privacy_keywords"]
                              if keyword in content.lower())

        if privacy_mentions > 3:  # Multiple privacy-related keywords
            # Check for privacy protection measures
            has_privacy_measures = re.search(
                r"privacy\s+(protection|preservation|safeguard)",
                content,
                re.IGNORECASE
            )

            if not has_privacy_measures:
                threats.append(SecurityThreat(
                    threat_type="privacy",
                    severity="high",
                    description="High privacy risk without protection measures",
                    location="Document content",
                    evidence=f"Found {privacy_mentions} privacy-related keywords",
                    remediation="Add privacy protection measures and compliance information"
                ))

        return threats

    def _validate_training_data_info(self, content: str) -> List[SecurityThreat]:
        """Validate training data information disclosure."""
        threats = []

        # Check for overly detailed data source information
        data_patterns = [
            r"user[_\s]*data", r"customer[_\s]*data", r"internal[_\s]*data",
            r"proprietary[_\s]*data", r"confidential[_\s]*data"
        ]

        for pattern in data_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                threats.append(SecurityThreat(
                    threat_type="data_disclosure",
                    severity="medium",
                    description="Potentially sensitive data source mentioned",
                    location="Training data section",
                    evidence=f"Pattern '{pattern}' found",
                    remediation="Remove sensitive data source details or ensure they are anonymized"
                ))

        return threats


class SecurityReportGenerator:
    """Generate comprehensive security reports."""

    def generate_security_report(
        self,
        scan_results: List[SecurityScanResult],
        format_type: str = "json"
    ) -> str:
        """Generate comprehensive security report."""

        # Aggregate results
        all_threats = []
        total_scans = len(scan_results)
        safe_scans = sum(1 for result in scan_results if result.safe)

        for result in scan_results:
            all_threats.extend(result.threats)

        # Categorize threats
        threat_summary = self._categorize_threats(all_threats)

        # Risk assessment
        risk_level = self._assess_risk_level(all_threats)

        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_scans": total_scans,
                "safe_scans": safe_scans,
                "safety_rate": safe_scans / max(1, total_scans),
                "risk_level": risk_level
            },
            "threat_summary": threat_summary,
            "detailed_threats": [
                {
                    "type": threat.threat_type,
                    "severity": threat.severity,
                    "description": threat.description,
                    "location": threat.location,
                    "evidence": threat.evidence,
                    "timestamp": threat.timestamp.isoformat(),
                    "remediation": threat.remediation
                }
                for threat in sorted(all_threats, key=lambda t: {"critical": 0, "high": 1, "medium": 2, "low": 3}[t.severity])
            ],
            "scan_results": [
                {
                    "scan_type": result.scan_type,
                    "timestamp": result.timestamp.isoformat(),
                    "safe": result.safe,
                    "threat_count": len(result.threats),
                    "duration_ms": result.scan_duration_ms,
                    "metadata": result.metadata
                }
                for result in scan_results
            ],
            "recommendations": self._generate_recommendations(all_threats)
        }

        if format_type == "json":
            return json.dumps(report_data, indent=2)
        elif format_type == "html":
            return self._generate_html_report(report_data)
        else:
            return str(report_data)

    def _categorize_threats(self, threats: List[SecurityThreat]) -> Dict[str, Any]:
        """Categorize threats by type and severity."""
        categories = {}
        severities = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for threat in threats:
            if threat.threat_type not in categories:
                categories[threat.threat_type] = 0
            categories[threat.threat_type] += 1

            if threat.severity in severities:
                severities[threat.severity] += 1

        return {
            "by_type": categories,
            "by_severity": severities,
            "total_threats": len(threats)
        }

    def _assess_risk_level(self, threats: List[SecurityThreat]) -> str:
        """Assess overall risk level."""
        critical_count = sum(1 for t in threats if t.severity == "critical")
        high_count = sum(1 for t in threats if t.severity == "high")

        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "high"
        elif high_count > 0 or sum(1 for t in threats if t.severity == "medium") > 5:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, threats: List[SecurityThreat]) -> List[str]:
        """Generate security recommendations based on threats."""
        recommendations = []

        threat_types = {threat.threat_type for threat in threats}

        if "secrets" in threat_types:
            recommendations.append("Remove or encrypt all exposed secrets and API keys")

        if "injection" in threat_types:
            recommendations.append("Sanitize and validate all user inputs")

        if "ethics" in threat_types:
            recommendations.append("Add comprehensive ethical considerations and bias analysis")

        if "privacy" in threat_types:
            recommendations.append("Implement privacy protection measures and compliance documentation")

        if "file" in threat_types:
            recommendations.append("Review and secure file handling processes")

        return recommendations

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML security report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Scan Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .risk-level {{ padding: 5px 10px; border-radius: 3px; color: white; }}
                .critical {{ background: #dc3545; }}
                .high {{ background: #fd7e14; }}
                .medium {{ background: #ffc107; color: black; }}
                .low {{ background: #28a745; }}
                .threat {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .threat.critical {{ border-color: #dc3545; }}
                .threat.high {{ border-color: #fd7e14; }}
                .threat.medium {{ border-color: #ffc107; }}
                .threat.low {{ border-color: #28a745; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Security Scan Report</h1>
                <p>Generated: {report_data['report_metadata']['generated_at']}</p>
                <p>Risk Level: <span class="risk-level {report_data['report_metadata']['risk_level']}">{report_data['report_metadata']['risk_level'].upper()}</span></p>
            </div>
            
            <h2>Summary</h2>
            <ul>
                <li>Total Scans: {report_data['report_metadata']['total_scans']}</li>
                <li>Safe Scans: {report_data['report_metadata']['safe_scans']}</li>
                <li>Total Threats: {report_data['threat_summary']['total_threats']}</li>
            </ul>
            
            <h2>Threats by Severity</h2>
            <ul>
                <li>Critical: {report_data['threat_summary']['by_severity']['critical']}</li>
                <li>High: {report_data['threat_summary']['by_severity']['high']}</li>
                <li>Medium: {report_data['threat_summary']['by_severity']['medium']}</li>
                <li>Low: {report_data['threat_summary']['by_severity']['low']}</li>
            </ul>
            
            <h2>Detailed Threats</h2>
        """

        for threat in report_data["detailed_threats"]:
            html += f"""
            <div class="threat {threat['severity']}">
                <h3>{threat['description']}</h3>
                <p><strong>Type:</strong> {threat['type']}</p>
                <p><strong>Severity:</strong> {threat['severity']}</p>
                <p><strong>Location:</strong> {threat['location']}</p>
                <p><strong>Evidence:</strong> {threat['evidence']}</p>
                {f'<p><strong>Remediation:</strong> {threat["remediation"]}</p>' if threat['remediation'] else ''}
            </div>
            """

        html += """
            </body>
        </html>
        """

        return html


# Global instances
content_scanner = ContentSecurityScanner()
model_validator = ModelSecurityValidator()
report_generator = SecurityReportGenerator()


def perform_comprehensive_security_scan(
    content: str,
    content_type: str = "text",
    include_model_validation: bool = True
) -> List[SecurityScanResult]:
    """Perform comprehensive security scanning."""
    results = []

    # Content security scan
    content_result = content_scanner.scan_content(content, content_type)
    results.append(content_result)

    # Model-specific validation for model cards
    if include_model_validation and content_type in ["model_card", "text"]:
        model_result = model_validator.validate_model_card(content)
        results.append(model_result)

    return results


def generate_security_report(scan_results: List[SecurityScanResult], format_type: str = "json") -> str:
    """Generate security report from scan results."""
    return report_generator.generate_security_report(scan_results, format_type)
