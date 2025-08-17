"""
Enterprise-grade security module with advanced threat detection and compliance.

This module provides comprehensive security capabilities including:
- Advanced threat detection and analysis
- Content sanitization with ML-based detection
- Compliance checking for multiple standards
- Security audit logging
- Vulnerability assessment
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    EU_AI_ACT = "eu_ai_act"


@dataclass
class SecurityViolation:
    """Represents a security violation or threat."""
    threat_id: str
    threat_level: ThreatLevel
    category: str
    description: str
    timestamp: datetime
    source_location: Optional[str] = None
    remediation: Optional[str] = None
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Result of a security scan."""
    scan_id: str
    timestamp: datetime
    duration_ms: float
    violations: List[SecurityViolation]
    passed: bool
    score: float  # 0-100, higher is better
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedThreatDetector:
    """Advanced threat detection with ML-based analysis."""

    def __init__(self):
        self.patterns = self._load_threat_patterns()
        self.ml_models = self._initialize_ml_models()
        self.threat_history: List[SecurityViolation] = []

    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns."""
        return {
            "sql_injection": [
                r"(?i)(union\s+select|select\s+.*\s+from|insert\s+into|update\s+.*\s+set|delete\s+from)",
                r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1|'\s*or\s*'.*'|;\s*drop\s+table)",
                r"(?i)(exec\s*\(|sp_executesql|xp_cmdshell)"
            ],
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript\s*:",
                r"on\w+\s*=\s*[\"'][^\"']*[\"']",
                r"<iframe[^>]*src\s*=",
                r"eval\s*\(",
                r"document\.(cookie|location|domain)"
            ],
            "code_injection": [
                r"(?i)(exec|eval|system|shell_exec|passthru|file_get_contents)",
                r"__import__\s*\(",
                r"compile\s*\(",
                r"subprocess\.(call|run|Popen)",
                r"os\.(system|popen|execv?)"
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"..%c0%af",
                r"..%c1%9c"
            ],
            "sensitive_data": [
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"(?i)(password|passwd|pwd|secret|key|token)\s*[=:]\s*['\"]?[^\s'\"]+",
                r"(?i)(api[_-]?key|access[_-]?token|bearer\s+[a-zA-Z0-9]+)",
                r"(?i)(BEGIN\s+(RSA\s+)?PRIVATE\s+KEY|ssh-rsa\s+)"
            ],
            "malicious_urls": [
                r"(?i)(javascript|vbscript|data|file):",
                r"(?i)(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly)/[a-zA-Z0-9]+",
                r"(?i)https?://\d+\.\d+\.\d+\.\d+",  # IP addresses
                r"(?i)https?://[^/]*[a-z0-9-]{10,}\.tk/",  # Suspicious TLDs
            ]
        }

    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize ML models for threat detection."""
        # In a real implementation, this would load trained models
        return {
            "text_classifier": None,  # Would be a trained classifier
            "anomaly_detector": None,  # Would detect unusual patterns
            "intent_analyzer": None   # Would analyze content intent
        }

    async def scan_content(self, content: str, context: Optional[str] = None) -> List[SecurityViolation]:
        """Scan content for security threats."""
        violations = []
        scan_timestamp = datetime.now()

        # Pattern-based detection
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    violation = SecurityViolation(
                        threat_id=f"{category}_{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                        threat_level=self._assess_threat_level(category, match.group()),
                        category=category,
                        description=f"Detected {category} pattern: {match.group()[:50]}...",
                        timestamp=scan_timestamp,
                        source_location=f"Position {match.start()}-{match.end()}",
                        evidence={"matched_text": match.group(), "pattern": pattern}
                    )
                    violations.append(violation)

        # ML-based detection (placeholder)
        if self.ml_models["text_classifier"]:
            ml_violations = await self._ml_threat_detection(content, context)
            violations.extend(ml_violations)

        # Content analysis
        analysis_violations = self._analyze_content_structure(content)
        violations.extend(analysis_violations)

        return violations

    def _assess_threat_level(self, category: str, content: str) -> ThreatLevel:
        """Assess the threat level based on category and content."""
        threat_levels = {
            "sql_injection": ThreatLevel.HIGH,
            "xss": ThreatLevel.HIGH,
            "code_injection": ThreatLevel.CRITICAL,
            "path_traversal": ThreatLevel.MEDIUM,
            "sensitive_data": ThreatLevel.HIGH,
            "malicious_urls": ThreatLevel.MEDIUM
        }
        
        base_level = threat_levels.get(category, ThreatLevel.LOW)
        
        # Escalate based on content characteristics
        if any(keyword in content.lower() for keyword in ["drop", "delete", "truncate", "exec"]):
            if base_level == ThreatLevel.HIGH:
                return ThreatLevel.CRITICAL
        
        return base_level

    async def _ml_threat_detection(self, content: str, context: Optional[str]) -> List[SecurityViolation]:
        """Use ML models for advanced threat detection."""
        # Placeholder for ML-based detection
        # In a real implementation, this would use trained models
        return []

    def _analyze_content_structure(self, content: str) -> List[SecurityViolation]:
        """Analyze content structure for suspicious patterns."""
        violations = []
        
        # Check for excessive nested structures (potential DoS)
        if content.count('{') > 100 or content.count('[') > 100:
            violations.append(SecurityViolation(
                threat_id=f"structure_dos_{int(time.time())}",
                threat_level=ThreatLevel.MEDIUM,
                category="denial_of_service",
                description="Excessive nested structures detected",
                timestamp=datetime.now(),
                evidence={"brace_count": content.count('{'), "bracket_count": content.count('[')}
            ))

        # Check for suspiciously long lines (potential buffer overflow)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 10000:
                violations.append(SecurityViolation(
                    threat_id=f"long_line_{i}_{int(time.time())}",
                    threat_level=ThreatLevel.MEDIUM,
                    category="buffer_overflow",
                    description=f"Suspiciously long line detected (length: {len(line)})",
                    timestamp=datetime.now(),
                    source_location=f"Line {i+1}",
                    evidence={"line_length": len(line)}
                ))

        return violations


class ComplianceChecker:
    """Check content compliance with various regulatory frameworks."""

    def __init__(self):
        self.framework_rules = self._load_compliance_rules()

    def _load_compliance_rules(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Load compliance rules for each framework."""
        return {
            ComplianceFramework.GDPR: {
                "required_sections": [
                    "data_processing_purpose",
                    "data_retention_policy", 
                    "user_rights",
                    "consent_mechanism"
                ],
                "prohibited_content": [
                    r"(?i)personal\s+data.*without\s+consent",
                    r"(?i)unlimited\s+retention",
                    r"(?i)no\s+right\s+to\s+(deletion|erasure)"
                ],
                "required_disclaimers": [
                    r"(?i)gdpr\s+compliant",
                    r"(?i)data\s+protection"
                ]
            },
            ComplianceFramework.EU_AI_ACT: {
                "required_sections": [
                    "ai_system_classification",
                    "risk_assessment",
                    "human_oversight",
                    "transparency_measures",
                    "accuracy_requirements"
                ],
                "prohibited_content": [
                    r"(?i)biometric\s+identification.*real-time.*public",
                    r"(?i)social\s+scoring",
                    r"(?i)subliminal\s+techniques"
                ],
                "required_disclaimers": [
                    r"(?i)ai\s+act\s+compliant",
                    r"(?i)high-risk\s+ai\s+system"
                ]
            },
            ComplianceFramework.HIPAA: {
                "required_sections": [
                    "privacy_safeguards",
                    "access_controls",
                    "audit_trails",
                    "data_encryption"
                ],
                "prohibited_content": [
                    r"(?i)phi.*unencrypted",
                    r"(?i)health\s+information.*public",
                    r"(?i)medical\s+records.*shared.*without.*authorization"
                ]
            }
        }

    def check_compliance(self, content: str, frameworks: List[ComplianceFramework]) -> Dict[ComplianceFramework, ScanResult]:
        """Check compliance against specified frameworks."""
        results = {}
        
        for framework in frameworks:
            scan_id = f"compliance_{framework.value}_{int(time.time())}"
            start_time = time.time()
            violations = []
            
            if framework in self.framework_rules:
                violations = self._check_framework_compliance(content, framework)
            
            duration_ms = (time.time() - start_time) * 1000
            score = max(0, 100 - (len(violations) * 10))  # Penalty for each violation
            
            results[framework] = ScanResult(
                scan_id=scan_id,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                violations=violations,
                passed=len(violations) == 0,
                score=score,
                metadata={"framework": framework.value}
            )
        
        return results

    def _check_framework_compliance(self, content: str, framework: ComplianceFramework) -> List[SecurityViolation]:
        """Check compliance for a specific framework."""
        violations = []
        rules = self.framework_rules[framework]
        
        # Check for required sections
        required_sections = rules.get("required_sections", [])
        for section in required_sections:
            if not re.search(rf"(?i){section.replace('_', '\\s*')}", content):
                violations.append(SecurityViolation(
                    threat_id=f"missing_{section}_{framework.value}",
                    threat_level=ThreatLevel.HIGH,
                    category="compliance_violation",
                    description=f"Required section missing: {section}",
                    timestamp=datetime.now(),
                    compliance_frameworks=[framework],
                    remediation=f"Add section covering {section.replace('_', ' ')}"
                ))

        # Check for prohibited content
        prohibited_patterns = rules.get("prohibited_content", [])
        for pattern in prohibited_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                violations.append(SecurityViolation(
                    threat_id=f"prohibited_{framework.value}_{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                    threat_level=ThreatLevel.CRITICAL,
                    category="compliance_violation",
                    description=f"Prohibited content detected: {match.group()[:100]}...",
                    timestamp=datetime.now(),
                    source_location=f"Position {match.start()}-{match.end()}",
                    compliance_frameworks=[framework],
                    evidence={"matched_text": match.group()}
                ))

        return violations


class SecurityAuditor:
    """Comprehensive security auditing system."""

    def __init__(self):
        self.threat_detector = AdvancedThreatDetector()
        self.compliance_checker = ComplianceChecker()
        self.audit_log: List[Dict[str, Any]] = []
        self.scan_history: List[ScanResult] = []

    async def comprehensive_scan(
        self,
        content: str,
        context: Optional[str] = None,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None
    ) -> ScanResult:
        """Perform a comprehensive security scan."""
        scan_id = f"comprehensive_{int(time.time())}"
        start_time = time.time()
        all_violations = []

        # Threat detection
        threat_violations = await self.threat_detector.scan_content(content, context)
        all_violations.extend(threat_violations)

        # Compliance checking
        if compliance_frameworks:
            compliance_results = self.compliance_checker.check_compliance(content, compliance_frameworks)
            for framework_result in compliance_results.values():
                all_violations.extend(framework_result.violations)

        # Additional security checks
        additional_violations = self._additional_security_checks(content)
        all_violations.extend(additional_violations)

        duration_ms = (time.time() - start_time) * 1000
        
        # Calculate overall score
        score = self._calculate_security_score(all_violations, len(content))
        
        result = ScanResult(
            scan_id=scan_id,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            violations=all_violations,
            passed=len([v for v in all_violations if v.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]) == 0,
            score=score,
            metadata={
                "content_length": len(content),
                "context": context,
                "frameworks_checked": [f.value for f in (compliance_frameworks or [])]
            }
        )

        # Store in history
        self.scan_history.append(result)
        
        # Log audit entry
        self._log_audit_entry("security_scan", {
            "scan_id": scan_id,
            "violations_count": len(all_violations),
            "score": score,
            "passed": result.passed
        })

        return result

    def _additional_security_checks(self, content: str) -> List[SecurityViolation]:
        """Perform additional security checks."""
        violations = []
        
        # Check for potential data exfiltration patterns
        exfiltration_patterns = [
            r"(?i)(curl|wget|fetch)\s+.*http",
            r"(?i)POST\s+.*\/api\/",
            r"(?i)send\s+.*\s+to\s+.*@.*\.",
        ]
        
        for pattern in exfiltration_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                violations.append(SecurityViolation(
                    threat_id=f"exfiltration_{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                    threat_level=ThreatLevel.MEDIUM,
                    category="data_exfiltration",
                    description=f"Potential data exfiltration pattern: {match.group()[:50]}...",
                    timestamp=datetime.now(),
                    source_location=f"Position {match.start()}-{match.end()}",
                    evidence={"matched_text": match.group()}
                ))

        return violations

    def _calculate_security_score(self, violations: List[SecurityViolation], content_length: int) -> float:
        """Calculate overall security score (0-100)."""
        if not violations:
            return 100.0

        # Base penalty for each violation type
        penalties = {
            ThreatLevel.LOW: 2,
            ThreatLevel.MEDIUM: 5,
            ThreatLevel.HIGH: 15,
            ThreatLevel.CRITICAL: 25
        }

        total_penalty = sum(penalties.get(v.threat_level, 5) for v in violations)
        
        # Adjust penalty based on content length (longer content can have more violations)
        length_adjustment = min(1.0, content_length / 10000)  # Normalize to 10KB
        adjusted_penalty = total_penalty * (1 - length_adjustment * 0.3)
        
        score = max(0, 100 - adjusted_penalty)
        return round(score, 2)

    def _log_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Log an audit entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "user": "system",  # In real implementation, would get from context
            "session_id": "unknown"  # In real implementation, would get from context
        }
        
        self.audit_log.append(entry)
        logger.info(f"Security audit: {action}", extra={"audit_data": entry})

    def get_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a security report for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_scans = [
            scan for scan in self.scan_history 
            if scan.timestamp >= cutoff_date
        ]
        
        if not recent_scans:
            return {"message": "No scans in the specified period"}

        # Aggregate statistics
        total_scans = len(recent_scans)
        passed_scans = len([s for s in recent_scans if s.passed])
        total_violations = sum(len(s.violations) for s in recent_scans)
        
        violation_by_level = {}
        violation_by_category = {}
        
        for scan in recent_scans:
            for violation in scan.violations:
                level = violation.threat_level.value
                category = violation.category
                
                violation_by_level[level] = violation_by_level.get(level, 0) + 1
                violation_by_category[category] = violation_by_category.get(category, 0) + 1

        avg_score = sum(s.score for s in recent_scans) / total_scans if recent_scans else 0
        
        return {
            "period_days": days,
            "total_scans": total_scans,
            "passed_scans": passed_scans,
            "pass_rate": (passed_scans / total_scans) * 100 if total_scans > 0 else 0,
            "total_violations": total_violations,
            "average_score": round(avg_score, 2),
            "violations_by_level": violation_by_level,
            "violations_by_category": violation_by_category,
            "trend": self._calculate_trend(recent_scans),
            "recommendations": self._generate_recommendations(violation_by_category)
        }

    def _calculate_trend(self, scans: List[ScanResult]) -> str:
        """Calculate security trend."""
        if len(scans) < 2:
            return "insufficient_data"
        
        # Compare first half vs second half
        mid_point = len(scans) // 2
        first_half_score = sum(s.score for s in scans[:mid_point]) / mid_point
        second_half_score = sum(s.score for s in scans[mid_point:]) / (len(scans) - mid_point)
        
        if second_half_score > first_half_score + 5:
            return "improving"
        elif second_half_score < first_half_score - 5:
            return "degrading"
        else:
            return "stable"

    def _generate_recommendations(self, violations_by_category: Dict[str, int]) -> List[str]:
        """Generate security recommendations based on violation patterns."""
        recommendations = []
        
        if violations_by_category.get("sql_injection", 0) > 0:
            recommendations.append("Implement parameterized queries and input validation")
        
        if violations_by_category.get("xss", 0) > 0:
            recommendations.append("Implement proper output encoding and Content Security Policy")
        
        if violations_by_category.get("sensitive_data", 0) > 0:
            recommendations.append("Review data handling practices and implement data loss prevention")
        
        if violations_by_category.get("compliance_violation", 0) > 0:
            recommendations.append("Review compliance requirements and update documentation")
        
        if not recommendations:
            recommendations.append("Maintain current security practices")
        
        return recommendations


# Global security auditor instance
security_auditor = SecurityAuditor()


def secure_operation(compliance_frameworks: Optional[List[ComplianceFramework]] = None):
    """Decorator to add security scanning to operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                # Execute function
                result = await func(*args, **kwargs)
                
                # Scan result if it's content
                if isinstance(result, str) and len(result) > 100:
                    scan_result = await security_auditor.comprehensive_scan(
                        result,
                        context=func.__name__,
                        compliance_frameworks=compliance_frameworks
                    )
                    
                    if not scan_result.passed:
                        logger.warning(f"Security scan failed for {func.__name__}: {len(scan_result.violations)} violations")
                
                return result
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # Execute function
                result = func(*args, **kwargs)
                
                # Scan result if it's content (run in background)
                if isinstance(result, str) and len(result) > 100:
                    asyncio.create_task(
                        security_auditor.comprehensive_scan(
                            result,
                            context=func.__name__,
                            compliance_frameworks=compliance_frameworks
                        )
                    )
                
                return result
            return sync_wrapper
    return decorator