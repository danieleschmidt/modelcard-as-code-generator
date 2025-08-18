# Runbook: High Error Rate Incident

## Overview
This runbook covers responding to high error rates in the Model Card Generator service.

## Alert Details
- **Alert Name**: `ModelCardGenerator_HighErrorRate`
- **Threshold**: Error rate >5% over 5 minutes
- **Severity**: P1 - High

## Symptoms
- Increased 4xx/5xx HTTP response codes
- Customer reports of failed model card generation
- Elevated error metrics in dashboards

## Initial Response

### 1. Acknowledge Alert (0-5 minutes)
```bash
# Acknowledge in PagerDuty/AlertManager
# Update incident status in status page
```

### 2. Quick Assessment (5-10 minutes)

#### Check Current Error Rate
```bash
# Query Prometheus for current error rate
curl -G 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=rate(http_requests_total{job="modelcard-generator",status=~"4..|5.."}[5m]) / rate(http_requests_total{job="modelcard-generator"}[5m]) * 100'
```

#### Check Recent Deployments
```bash
# Check if any recent deployments occurred
kubectl rollout history deployment/modelcard-generator

# Check container restart events
kubectl get events --sort-by='.lastTimestamp' | grep modelcard-generator
```

#### View Recent Logs
```bash
# Check for error patterns in logs
kubectl logs -l app=modelcard-generator --tail=100 | grep -E "(ERROR|FATAL|Exception)"

# Or with Docker
docker logs --tail 100 modelcard-generator-app 2>&1 | grep -E "(ERROR|FATAL|Exception)"
```

## Investigation Steps

### 3. Identify Error Patterns (10-20 minutes)

#### Analyze Error Types
```bash
# Check HTTP status code distribution
curl -G 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=sum by (status) (rate(http_requests_total{job="modelcard-generator"}[5m]))'
```

#### Check Specific Error Messages
```bash
# Look for common error patterns
kubectl logs -l app=modelcard-generator --since=10m | \
  grep -E "(ERROR|Exception)" | \
  sort | uniq -c | sort -nr | head -20
```

#### Resource Utilization Check
```bash
# Check CPU usage
curl -G 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=rate(process_cpu_seconds_total{job="modelcard-generator"}[5m]) * 100'

# Check memory usage
curl -G 'http://prometheus:9090/api/v1/query' \
  --data-urlencode 'query=process_resident_memory_bytes{job="modelcard-generator"} / 1024 / 1024'
```

### 4. Common Root Causes

#### A. Downstream Service Issues
```bash
# Check ML platform integrations
curl -f http://mlflow:5000/health || echo "MLflow unavailable"
curl -f http://wandb-api/health || echo "W&B API unavailable"

# Check database connectivity
curl -f http://localhost:8080/health/db || echo "Database issues"
```

#### B. Resource Exhaustion
```bash
# Check disk space
df -h

# Check memory pressure
free -h

# Check container resource limits
kubectl describe pod -l app=modelcard-generator | grep -A 5 "Limits\|Requests"
```

#### C. Configuration Issues
```bash
# Check environment variables
kubectl get configmap modelcard-generator-config -o yaml

# Check secrets
kubectl get secret modelcard-generator-secrets -o yaml
```

#### D. Rate Limiting
```bash
# Check if hitting external API limits
kubectl logs -l app=modelcard-generator --since=10m | grep -i "rate limit\|quota\|throttle"
```

## Resolution Actions

### 5. Immediate Mitigation

#### A. Scale Up (if resource issues)
```bash
# Increase replicas
kubectl scale deployment modelcard-generator --replicas=5

# Or with docker-compose
docker-compose up --scale app=3 -d
```

#### B. Rollback (if deployment issue)
```bash
# Rollback to previous version
kubectl rollout undo deployment/modelcard-generator

# Check rollback status
kubectl rollout status deployment/modelcard-generator
```

#### C. Restart Service (if configuration issue)
```bash
# Restart pods
kubectl rollout restart deployment/modelcard-generator

# Or restart containers
docker-compose restart app
```

#### D. Circuit Breaker (if downstream issues)
```bash
# Enable circuit breaker for problematic integrations
curl -X POST http://localhost:8080/admin/circuit-breaker/enable \
  -H "Content-Type: application/json" \
  -d '{"service": "mlflow", "enabled": true}'
```

### 6. Verification

#### Confirm Error Rate Improvement
```bash
# Monitor error rate for 10-15 minutes
watch -n 30 'curl -G "http://prometheus:9090/api/v1/query" \
  --data-urlencode "query=rate(http_requests_total{job=\"modelcard-generator\",status=~\"4..|5..\"}[5m]) / rate(http_requests_total{job=\"modelcard-generator\"}[5m]) * 100"'
```

#### Test Functionality
```bash
# Test model card generation
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"eval_results": {"model_name": "test", "metrics": {"accuracy": 0.95}}, "format": "huggingface"}'
```

## Post-Incident Activities

### 7. Communication
- Update status page with resolution
- Notify stakeholders via Slack/email
- Close PagerDuty incident

### 8. Documentation
- Update incident timeline
- Document root cause
- Create follow-up tasks for prevention

### 9. Follow-up Actions
- Schedule post-incident review
- Implement monitoring improvements
- Create preventive measures

## Prevention Measures

### Monitoring Improvements
```yaml
# Add more specific alerts
- alert: ModelCardGenerator_SpecificErrorPattern
  expr: increase(log_errors_total{pattern="template_rendering_failed"}[5m]) > 10
  for: 2m
  labels:
    severity: warning
```

### Circuit Breaker Configuration
```yaml
# Configure circuit breakers for integrations
circuitBreaker:
  mlflow:
    failureThreshold: 5
    timeout: 30s
    resetTimeout: 60s
  wandb:
    failureThreshold: 3
    timeout: 20s
    resetTimeout: 45s
```

### Resource Limits
```yaml
# Ensure proper resource limits
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

## Contact Information

- **Primary On-Call**: See PagerDuty rotation
- **Engineering Manager**: @eng-manager
- **Platform Team**: @platform-team
- **Slack Channel**: `#alerts-modelcard-generator`

## Related Runbooks
- [Service Unavailable](incident-service-unavailable.md)
- [Performance Degradation](incident-performance-degradation.md)
- [Memory Issues](incident-memory-issues.md)