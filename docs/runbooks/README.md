# Operational Runbooks

This directory contains operational runbooks for the Model Card Generator service.

## Available Runbooks

### Incident Response
- [High Error Rate](incident-high-error-rate.md) - Response to elevated error rates
- [Service Unavailable](incident-service-unavailable.md) - Service downtime response
- [Performance Degradation](incident-performance-degradation.md) - Slow response times
- [Memory Issues](incident-memory-issues.md) - Memory leaks and high usage

### Maintenance
- [Deployment](maintenance-deployment.md) - Safe deployment procedures
- [Database Maintenance](maintenance-database.md) - Database operations
- [Cache Management](maintenance-cache.md) - Redis/cache operations
- [Log Management](maintenance-logs.md) - Log rotation and cleanup

### Monitoring
- [Alert Response](monitoring-alert-response.md) - How to respond to alerts
- [Metrics Investigation](monitoring-metrics.md) - Investigating metrics
- [Dashboard Usage](monitoring-dashboards.md) - Using Grafana dashboards

### Troubleshooting
- [Common Issues](troubleshooting-common.md) - Frequently encountered problems
- [Performance Issues](troubleshooting-performance.md) - Performance debugging
- [Integration Issues](troubleshooting-integrations.md) - ML platform integration issues

## Quick Reference

### Emergency Contacts
- **On-Call Engineer**: See PagerDuty rotation
- **Platform Team**: `@platform-team` in Slack
- **ML Team**: `@ml-team` in Slack

### Key Commands
```bash
# Check service health
curl http://localhost:8080/health

# View current metrics
curl http://localhost:8080/metrics

# Check container status
docker ps | grep modelcard-generator

# View recent logs
docker logs --tail 100 modelcard-generator-app

# Scale service
docker-compose up --scale app=3
```

### Key Metrics
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
- **Latency**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **Throughput**: `rate(http_requests_total[5m])`
- **Memory Usage**: `process_resident_memory_bytes`

### Alert Thresholds
- **High Error Rate**: >5% over 5 minutes
- **High Latency**: P95 >2 seconds over 10 minutes
- **Low Throughput**: <10 requests/minute for 5 minutes
- **High Memory**: >80% of available memory

## Escalation Matrix

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| P0 - Critical | 15 minutes | Immediate escalation to on-call manager |
| P1 - High | 1 hour | Escalate after 2 hours |
| P2 - Medium | 4 hours | Escalate after 1 day |
| P3 - Low | 1 business day | Escalate after 3 days |

## Getting Help

1. **Check this runbook** for known issues and solutions
2. **Search logs** for error patterns
3. **Check metrics** in Grafana dashboards
4. **Escalate** according to severity matrix
5. **Document** new issues and solutions