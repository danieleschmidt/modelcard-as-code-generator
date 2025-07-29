# Operational Excellence

This document outlines operational best practices, monitoring strategies, and incident response procedures for the Model Card Generator project.

## Operational Philosophy

Our operational excellence is built on these pillars:
- **Proactive Monitoring**: Detect issues before users experience them
- **Rapid Recovery**: Minimize time to resolution for incidents
- **Continuous Improvement**: Learn from every incident and outage
- **Automation First**: Reduce human error through automation

## Monitoring and Observability

### 1. Application Metrics

#### Core Business Metrics

```python
# src/modelcard_generator/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Business metrics
cards_generated_total = Counter(
    'modelcard_cards_generated_total',
    'Total number of model cards generated',
    ['format', 'template', 'status']
)

card_generation_duration = Histogram(
    'modelcard_generation_duration_seconds',
    'Time spent generating model cards',
    ['format', 'template'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
)

validation_errors_total = Counter(
    'modelcard_validation_errors_total',
    'Total validation errors',
    ['error_type', 'section']
)

active_users = Gauge(
    'modelcard_active_users',
    'Number of active users in the last 5 minutes'
)

# Technical metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
)

database_queries_total = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation', 'table']
)

database_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['operation', 'table']
)

def monitor_generation(format_type, template):
    """Decorator to monitor card generation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                cards_generated_total.labels(
                    format=format_type, 
                    template=template, 
                    status=status
                ).inc()
                card_generation_duration.labels(
                    format=format_type, 
                    template=template
                ).observe(duration)
        
        return wrapper
    return decorator
```

#### Health Check Implementation

```python
# src/modelcard_generator/monitoring/health.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import asyncio
import aiohttp
import psycopg2
from datetime import datetime, timedelta

@dataclass
class HealthStatus:
    """Health check status."""
    service: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'redis': self._check_redis,
            'external_apis': self._check_external_apis,
            'disk_space': self._check_disk_space,
            'memory': self._check_memory,
            'templates': self._check_templates
        }
    
    async def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                results[check_name] = await check_func()
            except Exception as e:
                results[check_name] = HealthStatus(
                    service=check_name,
                    status='unhealthy',
                    response_time_ms=0,
                    details={'error': str(e)},
                    timestamp=datetime.utcnow()
                )
        
        return results
    
    async def _check_database(self) -> HealthStatus:
        """Check database connectivity and performance."""
        start_time = datetime.utcnow()
        
        try:
            # Test connection
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                connect_timeout=5
            )
            
            # Test query performance
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            # Check connection pool
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            active_connections = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthStatus(
                service='database',
                status='healthy' if response_time < 100 else 'degraded',
                response_time_ms=response_time,
                details={
                    'active_connections': active_connections,
                    'connection_test': 'passed'
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthStatus(
                service='database',
                status='unhealthy',
                response_time_ms=0,
                details={'error': str(e)},
                timestamp=datetime.utcnow()
            )
    
    async def _check_external_apis(self) -> HealthStatus:
        """Check external API dependencies."""
        start_time = datetime.utcnow()
        apis_to_check = [
            'https://huggingface.co/api/health',
            'https://api.wandb.ai/health',
            'https://api.github.com'
        ]
        
        results = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for api_url in apis_to_check:
                try:
                    async with session.get(api_url) as response:
                        results[api_url] = {
                            'status_code': response.status,
                            'response_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
                        }
                except Exception as e:
                    results[api_url] = {'error': str(e)}
        
        # Determine overall status
        all_healthy = all(
            result.get('status_code', 0) < 400 
            for result in results.values() 
            if 'status_code' in result
        )
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return HealthStatus(
            service='external_apis',
            status='healthy' if all_healthy else 'degraded',
            response_time_ms=response_time,
            details=results,
            timestamp=datetime.utcnow()
        )
```

### 2. Infrastructure Monitoring

#### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'modelcard-generator'
    static_configs:
      - targets: ['app:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
    
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

#### Alerting Rules

```yaml
# monitoring/rules/application.yml
groups:
- name: application
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
      runbook_url: "https://runbooks.terragonlabs.com/high-error-rate"
  
  - alert: SlowResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow response times detected"
      description: "95th percentile response time is {{ $value }}s"
      
  - alert: DatabaseConnectionIssues
    expr: up{job="postgresql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database is down"
      description: "PostgreSQL database is not responding"
      
  - alert: LowDiskSpace
    expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low disk space on {{ $labels.instance }}"
      description: "Disk space is below 10% on {{ $labels.instance }}"
      
  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      description: "Memory usage is above 90% on {{ $labels.instance }}"
```

### 3. Logging Strategy

#### Structured Logging Configuration

```python
# src/modelcard_generator/logging/config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'card_id'):
            log_entry['card_id'] = record.card_id
        
        return json.dumps(log_entry)

def setup_logging():
    """Set up structured logging."""
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for audit logs
    audit_handler = logging.FileHandler('/var/log/modelcard/audit.log')
    audit_handler.setFormatter(StructuredFormatter())
    audit_logger = logging.getLogger('audit')
    audit_logger.addHandler(audit_handler)
    audit_logger.setLevel(logging.INFO)
    
    # Security logger
    security_handler = logging.FileHandler('/var/log/modelcard/security.log')
    security_handler.setFormatter(StructuredFormatter())
    security_logger = logging.getLogger('security')
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.WARNING)

# Usage example
def log_card_generation(user_id: str, card_id: str, format_type: str):
    """Log model card generation event."""
    logger = logging.getLogger('audit')
    logger.info(
        "Model card generated",
        extra={
            'user_id': user_id,
            'card_id': card_id,
            'format': format_type,
            'event_type': 'card_generation'
        }
    )
```

#### Log Aggregation with ELK Stack

```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
  
  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - /var/log/modelcard:/var/log/modelcard:ro
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

## Incident Response

### 1. Incident Classification

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| **P0 - Critical** | Complete service outage | 15 minutes | Database down, API completely unavailable |
| **P1 - High** | Major functionality impacted | 1 hour | Card generation failing, slow response times |
| **P2 - Medium** | Minor functionality affected | 4 hours | Specific template not working, UI glitches |
| **P3 - Low** | Cosmetic or enhancement | 24 hours | Documentation errors, minor UI improvements |

### 2. Incident Response Playbook

#### P0 - Critical Incident Response

```bash
#!/bin/bash
# scripts/incident-response-p0.sh

set -e

echo "🚨 P0 CRITICAL INCIDENT RESPONSE ACTIVATED"
echo "Time: $(date)"
echo "Reporter: $USER"

# Step 1: Immediate assessment
echo "📊 Running immediate health checks..."
curl -f http://localhost:8080/health || echo "❌ Health check failed"
curl -f http://localhost:8080/metrics || echo "❌ Metrics endpoint failed"

# Step 2: Check database connectivity
echo "🗄️ Checking database..."
pg_isready -h localhost -p 5432 || echo "❌ Database connection failed"

# Step 3: Check recent deployments
echo "🚀 Checking recent deployments..."
kubectl get deployments -o wide
kubectl get pods | grep -E "(Error|CrashLoopBackOff|Pending)"

# Step 4: Check logs for errors
echo "📋 Checking recent error logs..."
kubectl logs deployment/modelcard-generator --tail=100 | grep -E "(ERROR|FATAL|Exception)"

# Step 5: Check resource usage
echo "💾 Checking resource usage..."
kubectl top nodes
kubectl top pods

# Step 6: Alert team
echo "📢 Alerting incident response team..."
python scripts/alert_team.py --severity=P0 --message="Critical incident detected"

# Step 7: Create incident ticket
echo "🎫 Creating incident ticket..."
python scripts/create_incident.py --severity=P0 --description="Auto-detected critical incident"

echo "✅ P0 incident response steps completed"
echo "🔗 Next steps: https://runbooks.terragonlabs.com/p0-incident"
```

#### Incident Communication Template

```python
# scripts/incident_communication.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class IncidentUpdate:
    """Incident update structure."""
    timestamp: datetime
    status: str  # investigating, identified, monitoring, resolved
    summary: str
    impact: str
    next_update: Optional[datetime] = None
    resolution_eta: Optional[datetime] = None

class IncidentCommunication:
    """Handle incident communication."""
    
    def __init__(self, incident_id: str, severity: str):
        self.incident_id = incident_id
        self.severity = severity
        self.updates: List[IncidentUpdate] = []
    
    def post_update(self, status: str, summary: str, impact: str):
        """Post an incident update."""
        update = IncidentUpdate(
            timestamp=datetime.utcnow(),
            status=status,
            summary=summary,
            impact=impact
        )
        
        self.updates.append(update)
        
        # Send to multiple channels
        self._send_slack_update(update)
        self._send_email_update(update)
        self._update_status_page(update)
    
    def _send_slack_update(self, update: IncidentUpdate):
        """Send update to Slack."""
        message = {
            "text": f"🚨 Incident Update - {self.incident_id}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Incident:* {self.incident_id}\n*Severity:* {self.severity}\n*Status:* {update.status}\n*Summary:* {update.summary}\n*Impact:* {update.impact}"
                    }
                }
            ]
        }
        
        # Send to appropriate Slack channels based on severity
        channels = ['#incidents']
        if self.severity in ['P0', 'P1']:
            channels.extend(['#engineering', '#leadership'])
        
        for channel in channels:
            self._send_slack_message(channel, message)
    
    def generate_postmortem_template(self) -> str:
        """Generate postmortem template."""
        return f"""
# Incident Postmortem - {self.incident_id}

## Summary
Brief description of the incident.

## Timeline
{self._format_timeline()}

## Root Cause
What caused the incident?

## Impact
- Duration: X hours
- Users affected: X
- Services affected: X
- Revenue impact: $X

## Resolution
How was the incident resolved?

## Lessons Learned

### What went well?
- 

### What could be improved?
- 

### Action Items
- [ ] Action item 1 (Owner: @person, Due: date)
- [ ] Action item 2 (Owner: @person, Due: date)

## Prevention
How can we prevent this from happening again?
"""
    
    def _format_timeline(self) -> str:
        """Format incident timeline."""
        timeline = ""
        for update in self.updates:
            timeline += f"- {update.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}: {update.status.title()} - {update.summary}\n"
        return timeline
```

### 3. Runbooks

#### Database Connection Issues

```markdown
# Runbook: Database Connection Issues

## Symptoms
- Health checks failing
- "Connection refused" errors in logs
- Database-related 500 errors

## Immediate Actions

### 1. Check Database Status
```bash
# Check if PostgreSQL is running
pg_isready -h $DB_HOST -p $DB_PORT

# Check container status
docker ps | grep postgres
kubectl get pods | grep postgres
```

### 2. Check Connection Pool
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- Check connection limits
SELECT setting FROM pg_settings WHERE name = 'max_connections';

-- Check for long-running queries
SELECT pid, query_start, state, query 
FROM pg_stat_activity 
WHERE state != 'idle' 
AND query_start < now() - interval '1 minute';
```

### 3. Check Application Configuration
```bash
# Verify database connection string
echo $DATABASE_URL

# Check connection pool settings
grep -E "(pool_size|max_overflow)" config/database.yaml
```

## Resolution Steps

### If Database is Down
1. Check database logs: `kubectl logs postgres-pod-name`
2. Check storage: `df -h` on database server
3. Restart database: `kubectl rollout restart deployment/postgres`
4. Monitor recovery: `kubectl get pods -w`

### If Connection Pool Exhausted
1. Kill long-running queries: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE ...`
2. Increase pool size temporarily
3. Restart application pods: `kubectl rollout restart deployment/modelcard-generator`

### If Network Issues
1. Check network connectivity: `telnet $DB_HOST $DB_PORT`
2. Check firewall rules
3. Check DNS resolution: `nslookup $DB_HOST`

## Prevention
- Monitor connection pool metrics
- Set up alerting for connection exhaustion
- Regular database maintenance
- Connection leak detection
```

#### High CPU Usage

```markdown
# Runbook: High CPU Usage

## Symptoms
- Application response times increased
- CPU usage > 80% for extended periods
- Kubernetes HPA scaling out pods

## Investigation

### 1. Identify CPU-intensive processes
```bash
# Top processes by CPU
top -o +%CPU

# Application-specific processes
ps aux | grep python | sort -k3 -nr

# In Kubernetes
kubectl top pods --sort-by=cpu
```

### 2. Application profiling
```python
# Enable profiling in application
import cProfile
import pstats

# Profile card generation
profiler = cProfile.Profile()
profiler.enable()
# ... card generation code ...
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### 3. Check for resource limits
```yaml
# Kubernetes resource limits
kubectl describe pod modelcard-generator-xxx | grep -A 10 Limits
```

## Resolution Steps

### Immediate
1. Scale out application: `kubectl scale deployment modelcard-generator --replicas=6`
2. Check for memory leaks: Monitor memory usage trends
3. Restart high-CPU pods: `kubectl delete pod <pod-name>`

### Medium-term
1. Optimize slow database queries
2. Implement caching for expensive operations
3. Review and optimize algorithms
4. Consider increasing resource limits

### Long-term
1. Performance testing and optimization
2. Code review for inefficient patterns
3. Consider architectural changes
4. Implement better monitoring and alerting
```

## Performance Optimization

### 1. Application Performance

#### Database Query Optimization

```python
# src/modelcard_generator/database/optimization.py
import logging
from contextlib import contextmanager
from time import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

@contextmanager
def query_timer(query_name: str):
    """Context manager to time database queries."""
    start_time = time()
    try:
        yield
    finally:
        duration = time() - start_time
        if duration > 1.0:  # Log slow queries
            logger.warning(
                f"Slow query detected: {query_name} took {duration:.2f}s",
                extra={'query_name': query_name, 'duration': duration}
            )

# Example usage
def get_model_cards_optimized(user_id: str, limit: int = 10):
    """Optimized query for retrieving model cards."""
    with query_timer("get_model_cards"):
        # Use indexes, proper joins, and pagination
        query = """
        SELECT mc.id, mc.name, mc.created_at, u.username
        FROM model_cards mc
        JOIN users u ON mc.user_id = u.id
        WHERE mc.user_id = %s AND mc.deleted_at IS NULL
        ORDER BY mc.created_at DESC
        LIMIT %s
        """
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id, limit))
            return cursor.fetchall()
```

#### Caching Strategy

```python
# src/modelcard_generator/cache/redis_cache.py
import redis
import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps

class RedisCache:
    """Redis-based caching system."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def cache_result(self, ttl: int = 3600, key_prefix: str = "mcg"):
        """Decorator to cache function results."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs, key_prefix)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict, prefix: str) -> str:
        """Generate consistent cache key."""
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{func_name}:{key_hash}"

# Usage
cache = RedisCache()

@cache.cache_result(ttl=1800, key_prefix="templates")
def get_template_content(template_name: str, format_type: str) -> str:
    """Cached template retrieval."""
    # Expensive template processing
    return process_template(template_name, format_type)
```

### 2. Infrastructure Optimization

#### Auto-scaling Configuration

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: modelcard-generator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: modelcard-generator
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

#### Resource Management

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelcard-generator
spec:
  template:
    spec:
      containers:
      - name: app
        image: terragonlabs/modelcard-generator:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: JAVA_OPTS
          value: "-Xmx3g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
        - name: PYTHON_GIL_COUNT
          value: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
```

## Disaster Recovery

### 1. Backup Strategy

```bash
#!/bin/bash
# scripts/backup-system.sh

set -e

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

echo "🗄️ Starting database backup..."
pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/database.sql.gz

echo "📁 Backing up uploaded files..."
tar -czf $BACKUP_DIR/uploads.tar.gz /var/lib/modelcard/uploads/

echo "⚙️ Backing up configuration..."
kubectl get configmaps -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secrets -o yaml > $BACKUP_DIR/secrets.yaml

echo "📊 Backing up monitoring data..."
curl -X POST http://prometheus:9090/api/v1/admin/tsdb/snapshot
tar -czf $BACKUP_DIR/prometheus.tar.gz /prometheus/snapshots/

echo "☁️ Uploading to cloud storage..."
aws s3 sync $BACKUP_DIR s3://modelcard-backups/$(date +%Y%m%d)/

echo "✅ Backup completed successfully"
```

### 2. Recovery Procedures

```bash
#!/bin/bash
# scripts/disaster-recovery.sh

set -e

RESTORE_DATE=$1
if [ -z "$RESTORE_DATE" ]; then
    echo "Usage: $0 <YYYYMMDD>"
    exit 1
fi

echo "🚨 Starting disaster recovery for date: $RESTORE_DATE"

# Download backups
aws s3 sync s3://modelcard-backups/$RESTORE_DATE/ /tmp/restore/

# Restore database
echo "🗄️ Restoring database..."
gunzip -c /tmp/restore/database.sql.gz | psql $DATABASE_URL

# Restore files
echo "📁 Restoring files..."
tar -xzf /tmp/restore/uploads.tar.gz -C /

# Restore configuration
echo "⚙️ Restoring configuration..."
kubectl apply -f /tmp/restore/configmaps.yaml
kubectl apply -f /tmp/restore/secrets.yaml

# Restart services
echo "🔄 Restarting services..."
kubectl rollout restart deployment/modelcard-generator
kubectl rollout restart deployment/postgres
kubectl rollout restart deployment/redis

echo "✅ Disaster recovery completed"
echo "🔍 Please verify system functionality"
```

This comprehensive operational excellence framework ensures reliable, monitored, and maintainable systems with proper incident response and recovery procedures.