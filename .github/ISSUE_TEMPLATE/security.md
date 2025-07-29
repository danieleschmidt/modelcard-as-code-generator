---
name: Security Vulnerability Report
about: Report a security vulnerability (Use this for security issues only)
title: '[SECURITY] '
labels: ['security', 'critical']
assignees: ''
---

## 🚨 Security Vulnerability Report

> **⚠️ IMPORTANT:** If this is a critical security vulnerability that could be actively exploited, please report it privately by emailing security@terragonlabs.com instead of creating a public issue.

### Vulnerability Details

#### Vulnerability Type
- [ ] SQL Injection
- [ ] Cross-Site Scripting (XSS)
- [ ] Command Injection
- [ ] Path Traversal
- [ ] Insecure Deserialization
- [ ] Authentication Bypass
- [ ] Authorization Flaw
- [ ] Information Disclosure
- [ ] Denial of Service
- [ ] Dependency Vulnerability
- [ ] Configuration Issue
- [ ] Other: _______________

#### Severity Level
- [ ] Critical (Immediate action required)
- [ ] High (Important security risk)
- [ ] Medium (Moderate security risk)
- [ ] Low (Minor security concern)

### Affected Components
- [ ] CLI interface
- [ ] Python API
- [ ] Web interface
- [ ] Container image
- [ ] Documentation
- [ ] Dependencies
- [ ] GitHub Actions workflows
- [ ] Other: _______________

### Environment Information
- **Version:** 
- **Operating System:** 
- **Python Version:** 
- **Installation Method:** (pip, container, source)
- **Configuration:** (relevant config details)

### Vulnerability Description

#### Summary
A clear and concise description of the vulnerability.

#### Technical Details
Detailed technical description of the vulnerability, including:
- How the vulnerability occurs
- What system components are affected
- What data or functionality is at risk

#### Attack Vector
How could an attacker exploit this vulnerability?
- [ ] Remote exploitation
- [ ] Local exploitation
- [ ] Requires authentication
- [ ] Requires user interaction
- [ ] Social engineering component

### Proof of Concept

#### Steps to Reproduce
1. 
2. 
3. 
4. 

#### Expected vs. Actual Behavior
**Expected:** 
**Actual:** 

#### Evidence
If applicable, provide:
- Code samples
- Screenshots (redact sensitive information)
- Log entries (sanitized)
- Network traces

### Impact Assessment

#### Confidentiality Impact
- [ ] No impact
- [ ] Limited disclosure
- [ ] Significant disclosure
- [ ] Complete compromise

#### Integrity Impact  
- [ ] No impact
- [ ] Limited modification
- [ ] Significant modification
- [ ] Complete compromise

#### Availability Impact
- [ ] No impact
- [ ] Limited disruption
- [ ] Significant disruption
- [ ] Complete service loss

#### Potential Business Impact
- [ ] Data breach
- [ ] Service disruption
- [ ] Compliance violation  
- [ ] Reputation damage
- [ ] Financial loss
- [ ] Legal liability

### Suggested Mitigation

#### Immediate Workarounds
Temporary measures users can take to reduce risk:

#### Proposed Fix
If you have suggestions for fixing the vulnerability:

#### Prevention Measures
How can similar vulnerabilities be prevented in the future?

### Additional Information

#### CVE Information
- [ ] CVE already assigned: CVE-YYYY-NNNN
- [ ] CVE requested
- [ ] No CVE needed

#### Discovery Method
- [ ] Security testing
- [ ] Code review
- [ ] Automated scanning
- [ ] External report
- [ ] Incident investigation
- [ ] Other: _______________

#### References
- Related CVEs:
- Security advisories:
- Documentation:

---

### For Maintainers

#### Response Checklist
- [ ] Acknowledge receipt within 24 hours
- [ ] Assess severity and impact
- [ ] Reproduce the vulnerability
- [ ] Develop and test fix
- [ ] Coordinate disclosure timeline
- [ ] Prepare security advisory
- [ ] Release patched version
- [ ] Update security documentation

#### Timeline Expectations
- **Acknowledgment:** Within 24 hours
- **Initial Assessment:** Within 72 hours  
- **Status Updates:** Every 7 days
- **Resolution Target:** Based on severity
  - Critical: 7 days
  - High: 30 days
  - Medium: 90 days
  - Low: Next major release

---

**Contact Information**
- Security Team: security@terragonlabs.com
- PGP Key: [Available on keyserver]
- Bug Bounty Program: [If applicable]