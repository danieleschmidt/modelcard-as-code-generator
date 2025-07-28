"""Security tests for Model Card Generator.

These tests verify that the model card generation process is secure
and doesn't introduce vulnerabilities or expose sensitive information.
"""

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


class MockSecureGenerator:
    """Mock generator for security testing."""
    
    def __init__(self):
        self.generated_content = ""
    
    def generate(self, data: Dict[str, Any]) -> str:
        """Generate model card, potentially with security issues."""
        content = f"# Model Card: {data.get('model_name', 'Unknown')}\n\n"
        
        # Include all data (this might expose secrets in real implementation)
        for key, value in data.items():
            content += f"**{key}**: {value}\n\n"
        
        self.generated_content = content
        return content
    
    def sanitize_content(self, content: str) -> str:
        """Remove potentially sensitive information."""
        # Remove API keys, tokens, passwords
        patterns = [
            r'api[_-]?key[_-]?[\w\d]+',
            r'token[_-]?[\w\d]+',
            r'password[_-]?[\w\d]+',
            r'secret[_-]?[\w\d]+',
            r'[a-zA-Z0-9]{32,}',  # Long strings that might be secrets
        ]
        
        sanitized = content
        for pattern in patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized


@pytest.fixture
def sensitive_data() -> Dict[str, Any]:
    """Test data containing sensitive information."""
    return {
        "model_name": "test-model",
        "api_key": "sk-1234567890abcdef1234567890abcdef",
        "database_password": "super_secret_password123",
        "auth_token": "bearer_token_xyz789",
        "aws_secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC...",
        "user_data": {
            "email": "user@example.com",
            "phone": "+1-555-123-4567",
            "ssn": "123-45-6789"
        },
        "config": {
            "database_url": "postgresql://user:password@localhost:5432/db",
            "redis_url": "redis://:password@localhost:6379/0"
        }
    }


@pytest.fixture
def malicious_input() -> Dict[str, Any]:
    """Test data with potential injection attacks."""
    return {
        "model_name": "<script>alert('XSS')</script>",
        "description": "'; DROP TABLE models; --",
        "template_injection": "{{ config.__class__.__init__.__globals__['os'].popen('ls').read() }}",
        "path_traversal": "../../../etc/passwd",
        "command_injection": "; rm -rf / #",
        "html_injection": "<img src=x onerror=alert('XSS')>",
        "yaml_injection": "!!python/object/apply:os.system ['ls']"
    }


@pytest.mark.security
class TestSecretsDetection:
    """Test detection and handling of secrets in model cards."""
    
    def test_api_key_detection(self, sensitive_data):
        """Test that API keys are detected and not exposed."""
        generator = MockSecureGenerator()
        content = generator.generate(sensitive_data)
        
        # Check that API key appears in raw content (vulnerable)
        assert "sk-1234567890abcdef" in content
        
        # Sanitize and verify API key is removed
        sanitized = generator.sanitize_content(content)
        assert "sk-1234567890abcdef" not in sanitized
        assert "[REDACTED]" in sanitized
    
    def test_password_detection(self, sensitive_data):
        """Test that passwords are detected and not exposed."""
        generator = MockSecureGenerator()
        content = generator.generate(sensitive_data)
        
        # Check for password patterns
        password_patterns = [
            r'password[_-]?[\w\d]+',
            r'super_secret_password123'
        ]
        
        # Should detect passwords in content
        found_passwords = []
        for pattern in password_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_passwords.extend(matches)
        
        assert len(found_passwords) > 0, "Should detect passwords in content"
        
        # After sanitization, passwords should be removed
        sanitized = generator.sanitize_content(content)
        for pattern in password_patterns:
            matches = re.findall(pattern, sanitized, re.IGNORECASE)
            assert len(matches) == 0, f"Password pattern still found after sanitization: {pattern}"
    
    def test_token_detection(self, sensitive_data):
        """Test that authentication tokens are detected."""
        generator = MockSecureGenerator()
        content = generator.generate(sensitive_data)
        
        # Should contain token before sanitization
        assert "bearer_token_xyz789" in content
        
        # Should be removed after sanitization
        sanitized = generator.sanitize_content(content)
        assert "bearer_token_xyz789" not in sanitized
    
    def test_private_key_detection(self, sensitive_data):
        """Test that private keys are detected and redacted."""
        generator = MockSecureGenerator()
        content = generator.generate(sensitive_data)
        
        # Should contain private key marker
        assert "BEGIN PRIVATE KEY" in content
        
        # After sanitization, should be redacted
        sanitized = generator.sanitize_content(content)
        # This is a basic check - real implementation should have better private key detection
        assert "[REDACTED]" in sanitized
    
    def test_pii_detection(self, sensitive_data):
        """Test detection of personally identifiable information."""
        generator = MockSecureGenerator()
        content = generator.generate(sensitive_data)
        
        # Should contain PII data
        assert "user@example.com" in content
        assert "555-123-4567" in content
        assert "123-45-6789" in content  # SSN
        
        # Real implementation should redact PII
        # For this test, we'll just verify it's detected
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\+?1?[-.]?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'  # Phone
        ]
        
        for pattern in pii_patterns:
            assert re.search(pattern, content), f"PII pattern not found: {pattern}"


@pytest.mark.security
class TestInjectionAttacks:
    """Test protection against various injection attacks."""
    
    def test_xss_prevention(self, malicious_input):
        """Test prevention of Cross-Site Scripting attacks."""
        generator = MockSecureGenerator()
        content = generator.generate(malicious_input)
        
        # Raw content might contain XSS
        assert "<script>" in content
        
        # Real implementation should escape HTML
        # For testing, we'll simulate proper escaping
        def escape_html(text: str) -> str:
            return text.replace("<", "&lt;").replace(">", "&gt;")
        
        escaped_content = escape_html(content)
        assert "&lt;script&gt;" in escaped_content
        assert "<script>" not in escaped_content
    
    def test_sql_injection_prevention(self, malicious_input):
        """Test prevention of SQL injection attacks."""
        generator = MockSecureGenerator()
        content = generator.generate(malicious_input)
        
        # Should contain SQL injection attempt
        assert "DROP TABLE" in content
        
        # Real implementation should sanitize SQL-like content
        # This is a basic example - real apps need proper parameterized queries
        def sanitize_sql(text: str) -> str:
            dangerous_patterns = [
                r"DROP\s+TABLE",
                r"DELETE\s+FROM",
                r"INSERT\s+INTO",
                r"UPDATE\s+.*\s+SET",
                r"UNION\s+SELECT"
            ]
            
            for pattern in dangerous_patterns:
                text = re.sub(pattern, "[SQL_BLOCKED]", text, flags=re.IGNORECASE)
            
            return text
        
        sanitized = sanitize_sql(content)
        assert "[SQL_BLOCKED]" in sanitized
        assert "DROP TABLE" not in sanitized.upper()
    
    def test_template_injection_prevention(self, malicious_input):
        """Test prevention of template injection attacks."""
        generator = MockSecureGenerator()
        content = generator.generate(malicious_input)
        
        # Should contain template injection attempt
        assert "{{" in content and "}}" in content
        
        # Real template engines should sandbox template execution
        def sanitize_template(text: str) -> str:
            # Remove template syntax that could be dangerous
            dangerous_template_patterns = [
                r"{{.*?__.*?}}",  # Dunder attributes
                r"{{.*?\.config.*?}}",  # Config access
                r"{{.*?\.globals.*?}}",  # Global access
                r"{{.*?\.import.*?}}",  # Import statements
            ]
            
            for pattern in dangerous_template_patterns:
                text = re.sub(pattern, "[TEMPLATE_BLOCKED]", text, flags=re.IGNORECASE)
            
            return text
        
        sanitized = sanitize_template(content)
        assert "[TEMPLATE_BLOCKED]" in sanitized
    
    def test_path_traversal_prevention(self, malicious_input):
        """Test prevention of path traversal attacks."""
        generator = MockSecureGenerator()
        content = generator.generate(malicious_input)
        
        # Should contain path traversal attempt
        assert "../../../" in content
        
        # Real implementation should validate and sanitize file paths
        def sanitize_path(text: str) -> str:
            # Remove path traversal patterns
            path_patterns = [
                r"\.\./",
                r"\.\.\\\\",
                r"%2e%2e%2f",  # URL encoded
                r"%2e%2e/",
            ]
            
            for pattern in path_patterns:
                text = re.sub(pattern, "[PATH_BLOCKED]", text, flags=re.IGNORECASE)
            
            return text
        
        sanitized = sanitize_path(content)
        assert "[PATH_BLOCKED]" in sanitized
        assert "../" not in sanitized
    
    def test_command_injection_prevention(self, malicious_input):
        """Test prevention of command injection attacks."""
        generator = MockSecureGenerator()
        content = generator.generate(malicious_input)
        
        # Should contain command injection attempt
        assert "; rm -rf" in content
        
        # Real implementation should never execute user input as commands
        def sanitize_commands(text: str) -> str:
            dangerous_commands = [
                r";\s*rm\s+-rf",
                r"&&\s*rm",
                r"\|\s*rm",
                r"`.*`",  # Backticks
                r"\$\(.*\)",  # Command substitution
            ]
            
            for pattern in dangerous_commands:
                text = re.sub(pattern, "[COMMAND_BLOCKED]", text, flags=re.IGNORECASE)
            
            return text
        
        sanitized = sanitize_commands(content)
        assert "[COMMAND_BLOCKED]" in sanitized


@pytest.mark.security
class TestFileSystemSecurity:
    """Test file system security measures."""
    
    def test_secure_file_creation(self, temp_dir):
        """Test that files are created with secure permissions."""
        import os
        import stat
        
        # Create a test file
        test_file = temp_dir / "test_model_card.md"
        test_file.write_text("# Test Model Card\n\nTest content.")
        
        # Check file permissions
        file_stat = test_file.stat()
        file_mode = stat.filemode(file_stat.st_mode)
        
        # File should not be world-readable if it contains sensitive data
        # This is a basic check - adjust based on your security requirements
        world_readable = bool(file_stat.st_mode & stat.S_IROTH)
        world_writable = bool(file_stat.st_mode & stat.S_IWOTH)
        
        # In production, you might want stricter permissions
        assert not world_writable, "File should not be world-writable"
    
    def test_directory_traversal_protection(self, temp_dir):
        """Test protection against directory traversal in file operations."""
        generator = MockSecureGenerator()
        
        # Attempt to write outside of allowed directory
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for malicious_path in malicious_paths:
            # Real implementation should validate paths
            def is_safe_path(path: str, base_dir: Path) -> bool:
                """Check if path is safe (doesn't escape base directory)."""
                try:
                    resolved_path = (base_dir / path).resolve()
                    return resolved_path.is_relative_to(base_dir)
                except (OSError, ValueError):
                    return False
            
            assert not is_safe_path(malicious_path, temp_dir), f"Path should be rejected: {malicious_path}"
    
    def test_symlink_attack_prevention(self, temp_dir):
        """Test prevention of symlink attacks."""
        import os
        
        # Create a symlink to a sensitive file
        sensitive_file = temp_dir / "sensitive.txt"
        sensitive_file.write_text("Sensitive content")
        
        symlink_path = temp_dir / "model_card.md"
        
        try:
            os.symlink(sensitive_file, symlink_path)
            
            # Real implementation should detect and handle symlinks appropriately
            def is_symlink_safe(path: Path) -> bool:
                """Check if following symlink is safe."""
                if path.is_symlink():
                    target = path.readlink()
                    # Check if target is within allowed directory
                    return target.is_relative_to(path.parent)
                return True
            
            assert not is_symlink_safe(symlink_path), "Symlink should be detected as unsafe"
            
        except OSError:
            # Symlinks might not be supported on all platforms
            pytest.skip("Symlinks not supported on this platform")


@pytest.mark.security
class TestDataValidation:
    """Test input data validation and sanitization."""
    
    def test_data_size_limits(self):
        """Test that input data size is limited to prevent DoS."""
        generator = MockSecureGenerator()
        
        # Create extremely large input
        huge_data = {
            "model_name": "test",
            "huge_field": "x" * (10 * 1024 * 1024)  # 10MB string
        }
        
        # Real implementation should limit input size
        def validate_data_size(data: Dict[str, Any], max_size_mb: int = 5) -> bool:
            """Validate that serialized data doesn't exceed size limit."""
            serialized = json.dumps(data)
            size_mb = len(serialized.encode('utf-8')) / (1024 * 1024)
            return size_mb <= max_size_mb
        
        assert not validate_data_size(huge_data), "Large data should be rejected"
        
        # Normal data should pass
        normal_data = {"model_name": "test", "metrics": {"accuracy": 0.95}}
        assert validate_data_size(normal_data), "Normal data should be accepted"
    
    def test_nested_data_depth_limits(self):
        """Test protection against deeply nested data structures."""
        generator = MockSecureGenerator()
        
        # Create deeply nested structure
        def create_nested_dict(depth: int) -> Dict[str, Any]:
            if depth == 0:
                return {"value": "leaf"}
            return {"nested": create_nested_dict(depth - 1)}
        
        deep_data = create_nested_dict(1000)  # Very deep nesting
        
        # Real implementation should limit nesting depth
        def validate_nesting_depth(data: Any, max_depth: int = 10, current_depth: int = 0) -> bool:
            """Validate that data doesn't exceed nesting depth limit."""
            if current_depth > max_depth:
                return False
            
            if isinstance(data, dict):
                for value in data.values():
                    if not validate_nesting_depth(value, max_depth, current_depth + 1):
                        return False
            elif isinstance(data, list):
                for item in data:
                    if not validate_nesting_depth(item, max_depth, current_depth + 1):
                        return False
            
            return True
        
        assert not validate_nesting_depth(deep_data), "Deeply nested data should be rejected"
        
        # Shallow data should pass
        shallow_data = {"model": {"metrics": {"accuracy": 0.95}}}
        assert validate_nesting_depth(shallow_data), "Shallow data should be accepted"
    
    def test_string_encoding_validation(self):
        """Test validation of string encoding and special characters."""
        generator = MockSecureGenerator()
        
        # Test various problematic strings
        problematic_strings = [
            "\x00\x01\x02",  # Null bytes and control characters
            "\uffff\ufffe",  # Unicode non-characters
            "\ud800\udc00",  # Surrogate pairs
            "\u202e\u202d",  # Text direction override
        ]
        
        for test_string in problematic_strings:
            data = {"model_name": test_string}
            
            # Real implementation should sanitize or reject problematic strings
            def sanitize_string(s: str) -> str:
                """Sanitize string by removing problematic characters."""
                # Remove null bytes and control characters
                sanitized = ''.join(char for char in s if ord(char) >= 32 or char in '\t\n\r')
                
                # Remove direction override characters
                direction_overrides = ['\u202a', '\u202b', '\u202c', '\u202d', '\u202e']
                for override in direction_overrides:
                    sanitized = sanitized.replace(override, '')
                
                return sanitized
            
            sanitized = sanitize_string(test_string)
            assert sanitized != test_string, f"String should be sanitized: {repr(test_string)}"


@pytest.mark.security
class TestComplianceAndAuditing:
    """Test compliance and auditing features."""
    
    def test_audit_trail_creation(self, temp_dir):
        """Test that audit trails are created for model card operations."""
        generator = MockSecureGenerator()
        
        # Mock audit logger
        audit_log = []
        
        def log_audit_event(event_type: str, details: Dict[str, Any]):
            """Log audit event."""
            import datetime
            audit_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "event_type": event_type,
                "details": details
            }
            audit_log.append(audit_entry)
        
        # Simulate model card generation with auditing
        test_data = {"model_name": "audit-test-model"}
        
        log_audit_event("model_card_generation_started", {
            "model_name": test_data["model_name"],
            "data_size": len(str(test_data))
        })
        
        result = generator.generate(test_data)
        
        log_audit_event("model_card_generation_completed", {
            "model_name": test_data["model_name"],
            "output_size": len(result),
            "success": True
        })
        
        # Verify audit trail
        assert len(audit_log) == 2
        assert audit_log[0]["event_type"] == "model_card_generation_started"
        assert audit_log[1]["event_type"] == "model_card_generation_completed"
        assert all("timestamp" in entry for entry in audit_log)
    
    def test_gdpr_compliance_data_handling(self):
        """Test GDPR compliance in data handling."""
        generator = MockSecureGenerator()
        
        # Data with potential personal information
        personal_data = {
            "model_name": "personal-data-model",
            "training_data": {
                "user_emails": ["user1@example.com", "user2@example.com"],
                "user_ids": ["12345", "67890"],
                "comments": ["Great product!", "Could be better"]
            }
        }
        
        # Real implementation should handle personal data according to GDPR
        def anonymize_personal_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """Anonymize personal data for GDPR compliance."""
            import hashlib
            
            def hash_personal_identifier(identifier: str) -> str:
                return hashlib.sha256(identifier.encode()).hexdigest()[:8]
            
            anonymized = data.copy()
            
            if "training_data" in anonymized:
                training_data = anonymized["training_data"].copy()
                
                # Hash email addresses
                if "user_emails" in training_data:
                    training_data["user_emails"] = [
                        f"user_{hash_personal_identifier(email)}@example.com"
                        for email in training_data["user_emails"]
                    ]
                
                # Hash user IDs
                if "user_ids" in training_data:
                    training_data["user_ids"] = [
                        hash_personal_identifier(user_id)
                        for user_id in training_data["user_ids"]
                    ]
                
                anonymized["training_data"] = training_data
            
            return anonymized
        
        anonymized_data = anonymize_personal_data(personal_data)
        
        # Verify personal data is anonymized
        assert "user1@example.com" not in str(anonymized_data)
        assert "12345" not in str(anonymized_data)
        
        # But structure is preserved
        assert "training_data" in anonymized_data
        assert "user_emails" in anonymized_data["training_data"]
        assert len(anonymized_data["training_data"]["user_emails"]) == 2


# Utility functions for security testing
def contains_secrets(content: str) -> List[str]:
    """Check if content contains potential secrets."""
    secret_patterns = [
        r'api[_-]?key[_-]?[\w\d]+',
        r'token[_-]?[\w\d]+',
        r'password[_-]?[\w\d]+',
        r'secret[_-]?[\w\d]+',
        r'[a-zA-Z0-9]{32,}',  # Long alphanumeric strings
        r'-----BEGIN [A-Z ]+-----',  # PEM format keys
    ]
    
    found_secrets = []
    for pattern in secret_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        found_secrets.extend(matches)
    
    return found_secrets


def validate_input_safety(data: Any, max_size_mb: int = 10, max_depth: int = 10) -> List[str]:
    """Validate input data for safety issues."""
    issues = []
    
    # Check size
    try:
        serialized = json.dumps(data)
        size_mb = len(serialized.encode('utf-8')) / (1024 * 1024)
        if size_mb > max_size_mb:
            issues.append(f"Data too large: {size_mb:.2f}MB > {max_size_mb}MB")
    except Exception as e:
        issues.append(f"Serialization error: {e}")
    
    # Check depth
    def check_depth(obj: Any, current_depth: int = 0) -> int:
        if current_depth > max_depth:
            return current_depth
        
        max_found_depth = current_depth
        
        if isinstance(obj, dict):
            for value in obj.values():
                depth = check_depth(value, current_depth + 1)
                max_found_depth = max(max_found_depth, depth)
        elif isinstance(obj, list):
            for item in obj:
                depth = check_depth(item, current_depth + 1)
                max_found_depth = max(max_found_depth, depth)
        
        return max_found_depth
    
    actual_depth = check_depth(data)
    if actual_depth > max_depth:
        issues.append(f"Data too deeply nested: {actual_depth} > {max_depth}")
    
    return issues
