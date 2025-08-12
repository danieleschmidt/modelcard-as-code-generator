# ModelCard Generator - Production Deployment Guide

## Overview

This comprehensive guide covers deployment of the enhanced ModelCard Generator with all production features including resilience patterns, intelligent caching, distributed processing, and monitoring. The system supports multiple deployment architectures from local development to enterprise Kubernetes clusters.

### Installation Methods

#### 1. PyPI Installation (Recommended)

```bash
# Latest stable release
pip install modelcard-as-code-generator

# With all optional dependencies
pip install "modelcard-as-code-generator[all]"

# Specific extras
pip install "modelcard-as-code-generator[cli,integrations]"
```

#### 2. Development Installation

```bash
git clone https://github.com/terragonlabs/modelcard-as-code-generator.git
cd modelcard-as-code-generator
pip install -e ".[dev,test,docs]"
```

#### 3. Container Installation

```bash
# Pull official image
docker pull terragonlabs/modelcard-generator:latest

# Run interactive session
docker run -it --rm -v $(pwd):/workspace terragonlabs/modelcard-generator:latest

# Run specific command
docker run --rm -v $(pwd):/workspace terragonlabs/modelcard-generator:latest generate config.yaml
```

### Deployment Architectures

#### 1. Local Development

**Use Case**: Individual developer, prototyping, local testing

```bash
# Install in virtual environment
python -m venv mcg-env
source mcg-env/bin/activate
pip install modelcard-as-code-generator

# Basic usage
mcg generate --config model_config.yaml --output model_card.md
```

**Configuration**: Environment variables or local config files
**Security**: Local file system permissions
**Scaling**: Single user, limited throughput

#### 2. CI/CD Integration

**Use Case**: Automated model card generation in pipelines

```yaml
# .github/workflows/model-cards.yml
name: Generate Model Cards
on:
  push:
    paths: ['models/**', 'evaluation/**']

jobs:
  generate-cards:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install MCG
        run: pip install modelcard-as-code-generator
      - name: Generate Cards
        run: mcg generate-batch --config configs/ --output docs/model-cards/
      - name: Commit Cards
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add docs/model-cards/
          git commit -m "Update model cards" || exit 0
          git push
```

#### 3. Containerized Service

**Use Case**: Team deployment, consistent environments, scalable processing

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcg-service:
    image: terragonlabs/modelcard-generator:latest
    volumes:
      - ./config:/app/config:ro
      - ./output:/app/output
      - ./cache:/app/cache
    environment:
      - MCG_LOG_LEVEL=INFO
      - MCG_CACHE_DIR=/app/cache
      - MCG_OUTPUT_DIR=/app/output
    command: ["mcg", "serve", "--host", "0.0.0.0", "--port", "8080"]
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  mcg-worker:
    image: terragonlabs/modelcard-generator:latest
    volumes:
      - ./config:/app/config:ro
      - ./output:/app/output
      - ./cache:/app/cache
    environment:
      - MCG_LOG_LEVEL=INFO
      - MCG_WORKER_MODE=true
      - MCG_REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    command: ["mcg", "worker"]
    scale: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

#### 4. Kubernetes Deployment

**Use Case**: Enterprise deployment, high availability, auto-scaling

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: modelcard-generator
  labels:
    name: modelcard-generator

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcg-config
  namespace: modelcard-generator
data:
  MCG_LOG_LEVEL: "INFO"
  MCG_CACHE_DIR: "/app/cache"
  MCG_OUTPUT_DIR: "/app/output"
  MCG_REDIS_URL: "redis://redis-service:6379/0"

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcg-api
  namespace: modelcard-generator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcg-api
  template:
    metadata:
      labels:
        app: mcg-api
    spec:
      containers:
      - name: mcg-api
        image: terragonlabs/modelcard-generator:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: mcg-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: cache-volume
          mountPath: /app/cache
        - name: output-volume
          mountPath: /app/output
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: mcg-cache-pvc
      - name: output-volume
        persistentVolumeClaim:
          claimName: mcg-output-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mcg-service
  namespace: modelcard-generator
spec:
  selector:
    app: mcg-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mcg-ingress
  namespace: modelcard-generator
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - mcg.yourdomain.com
    secretName: mcg-tls
  rules:
  - host: mcg.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mcg-service
            port:
              number: 80

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcg-hpa
  namespace: modelcard-generator
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcg-api
  minReplicas: 2
  maxReplicas: 10
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
```

### Configuration Management

#### Environment Variables

```bash
# Core Configuration
export MCG_LOG_LEVEL=INFO
export MCG_ENVIRONMENT=production
export MCG_DEBUG=false

# Paths
export MCG_CONFIG_DIR=/etc/mcg
export MCG_OUTPUT_DIR=/var/lib/mcg/output
export MCG_CACHE_DIR=/var/lib/mcg/cache
export MCG_TEMPLATES_DIR=/usr/share/mcg/templates

# Database/Storage
export MCG_REDIS_URL=redis://localhost:6379/0
export MCG_DATABASE_URL=postgresql://user:pass@localhost/mcg

# Integrations
export MCG_MLFLOW_TRACKING_URI=http://mlflow:5000
export MCG_WANDB_API_KEY=your_wandb_key
export MCG_HF_TOKEN=your_huggingface_token

# Security
export MCG_SECRET_KEY=your-secret-key
export MCG_JWT_ALGORITHM=HS256
export MCG_JWT_EXPIRE_MINUTES=1440

# Performance
export MCG_MAX_WORKERS=4
export MCG_QUEUE_SIZE=100
export MCG_TIMEOUT_SECONDS=300
```

#### Configuration Files

```yaml
# /etc/mcg/config.yaml
api:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  timeout: 300

logging:
  level: INFO
  format: json
  file: /var/log/mcg/app.log
  max_size: 100MB
  backup_count: 5

storage:
  output_dir: /var/lib/mcg/output
  cache_dir: /var/lib/mcg/cache
  max_cache_size: 10GB
  cache_ttl: 86400  # 24 hours

templates:
  default_format: huggingface
  custom_templates_dir: /usr/share/mcg/templates
  strict_validation: true

integrations:
  mlflow:
    tracking_uri: http://mlflow:5000
    experiment_name: model-cards
  wandb:
    project: model-cards
    entity: your-team
  huggingface:
    model_hub_url: https://huggingface.co

security:
  enable_auth: true
  secret_key: ${MCG_SECRET_KEY}
  allowed_hosts:
    - mcg.yourdomain.com
    - localhost
  cors_origins:
    - https://yourdomain.com

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_path: /health
  ready_check_path: /ready
  enable_tracing: true
  jaeger_endpoint: http://jaeger:14268/api/traces
```

### Security Configuration

#### 1. Network Security

```bash
# Firewall rules (iptables)
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
iptables -A INPUT -p tcp --dport 9090 -j ACCEPT  # metrics
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT  # HTTPS outbound
iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT   # HTTP outbound
```

#### 2. TLS/SSL Configuration

```yaml
# nginx.conf
server {
    listen 443 ssl http2;
    server_name mcg.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/mcg.crt;
    ssl_certificate_key /etc/ssl/private/mcg.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://mcg-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 3. Authentication & Authorization

```python
# auth_config.py
AUTH_CONFIG = {
    'providers': {
        'jwt': {
            'algorithm': 'RS256',
            'public_key_url': 'https://auth.yourdomain.com/.well-known/jwks.json',
            'audience': 'mcg-api',
            'issuer': 'https://auth.yourdomain.com'
        },
        'api_key': {
            'header_name': 'X-API-Key',
            'valid_keys': ['key1', 'key2']  # In production, use secure storage
        }
    },
    'rbac': {
        'roles': {
            'admin': ['*'],
            'user': ['generate', 'validate'],
            'readonly': ['view']
        }
    }
}
```

### Performance Optimization

#### 1. Caching Strategy

```yaml
# Redis configuration
redis:
  host: localhost
  port: 6379
  db: 0
  ttl: 3600  # 1 hour
  max_memory: 2gb
  eviction_policy: allkeys-lru

cache:
  levels:
    - name: memory
      size: 100MB
      ttl: 300  # 5 minutes
    - name: redis
      size: 1GB
      ttl: 3600  # 1 hour
    - name: disk
      size: 10GB
      ttl: 86400  # 24 hours
```

#### 2. Load Balancing

```nginx
# nginx upstream configuration
upstream mcg_backend {
    least_conn;
    server mcg-1:8080 max_fails=3 fail_timeout=30s;
    server mcg-2:8080 max_fails=3 fail_timeout=30s;
    server mcg-3:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    location / {
        proxy_pass http://mcg_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Monitoring & Observability

#### 1. Health Checks

```python
# health.py
from typing import Dict, Any
import asyncio

async def health_check() -> Dict[str, Any]:
    """Comprehensive health check."""
    checks = {
        'database': await check_database(),
        'redis': await check_redis(),
        'storage': await check_storage(),
        'integrations': await check_integrations()
    }
    
    overall_status = 'healthy' if all(
        check['status'] == 'healthy' for check in checks.values()
    ) else 'unhealthy'
    
    return {
        'status': overall_status,
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }
```

#### 2. Metrics Collection

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mcg-api'
    static_configs:
      - targets: ['mcg-1:9090', 'mcg-2:9090', 'mcg-3:9090']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'mcg-worker'
    static_configs:
      - targets: ['mcg-worker-1:9090', 'mcg-worker-2:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

#### 3. Logging Configuration

```yaml
# logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /var/log/mcg/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 5
  
  syslog:
    class: logging.handlers.SysLogHandler
    level: WARNING
    formatter: json
    address: ['localhost', 514]

loggers:
  modelcard_generator:
    level: INFO
    handlers: [console, file, syslog]
    propagate: false

  uvicorn:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

### Backup & Recovery

#### 1. Data Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/mcg"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configuration
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" /etc/mcg/

# Backup output files
tar -czf "$BACKUP_DIR/output_$DATE.tar.gz" /var/lib/mcg/output/

# Backup database (if using)
pg_dump mcg > "$BACKUP_DIR/database_$DATE.sql"

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/" s3://mcg-backups/ --recursive

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -type f -mtime +30 -delete
```

#### 2. Disaster Recovery

```yaml
# disaster-recovery.yaml
recovery_procedures:
  data_loss:
    - restore_from_s3_backup
    - rebuild_cache
    - verify_data_integrity
  
  service_failure:
    - check_health_endpoints
    - restart_failed_pods
    - scale_up_if_needed
  
  infrastructure_failure:
    - failover_to_secondary_region
    - update_dns_records
    - notify_stakeholders

monitoring:
  rto: 4_hours      # Recovery Time Objective
  rpo: 1_hour       # Recovery Point Objective
  sla: 99.9         # Service Level Agreement
```

### Troubleshooting

#### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   kubectl top pods -n modelcard-generator
   
   # Solutions
   - Reduce batch size
   - Increase cache TTL
   - Scale horizontally
   ```

2. **Slow Response Times**
   ```bash
   # Check metrics
   curl http://localhost:9090/metrics | grep response_time
   
   # Solutions
   - Enable caching
   - Optimize templates
   - Add more workers
   ```

3. **Connection Failures**
   ```bash
   # Check connectivity
   telnet redis-host 6379
   curl -I http://mlflow:5000
   
   # Solutions
   - Verify network policies
   - Check firewall rules
   - Update connection strings
   ```

### Maintenance

#### Regular Tasks

```bash
# Weekly maintenance script
#!/bin/bash

# Update dependencies
pip list --outdated
pip install --upgrade modelcard-as-code-generator

# Clean cache
mcg cache clean --older-than 7d

# Backup configuration
./scripts/backup.sh

# Check security updates
safety check
bandit -r /app/

# Update documentation
mkdocs build

# Performance check
mcg benchmark --config production.yaml
```

#### Version Upgrades

```bash
# Upgrade process
1. Backup current deployment
2. Test new version in staging
3. Rolling update in production
4. Verify functionality
5. Rollback if issues

# Kubernetes rolling update
kubectl set image deployment/mcg-api mcg-api=terragonlabs/modelcard-generator:v2.0.0
kubectl rollout status deployment/mcg-api
kubectl rollout undo deployment/mcg-api  # if needed
```

This deployment guide provides comprehensive coverage for all deployment scenarios from development to enterprise production environments.