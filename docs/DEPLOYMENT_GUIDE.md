# Deployment Guide

## Overview

This guide covers deploying the Model Card Generator in production environments, including Docker, Kubernetes, and cloud platforms.

## Docker Deployment

### Build Image

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY setup.py .
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mcg
USER mcg

EXPOSE 8080

CMD ["uvicorn", "modelcard_generator.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Build and Run

```bash
# Build image
docker build -t modelcard-generator:latest .

# Run container
docker run -p 8080:8080 modelcard-generator:latest

# Run with volume mount
docker run -p 8080:8080 -v $(pwd)/data:/app/data modelcard-generator:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  modelcard-generator:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: modelcards
      POSTGRES_USER: mcg
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## Kubernetes Deployment

### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: modelcard-system
  labels:
    name: modelcard-system
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: modelcard-config
  namespace: modelcard-system
data:
  config.yaml: |
    app:
      name: "Model Card Generator"
      version: "1.0.0"
      environment: "production"
    
    logging:
      level: "info"
      format: "json"
    
    cache:
      type: "redis"
      host: "redis-service"
      port: 6379
    
    database:
      type: "postgresql"
      host: "postgres-service"
      port: 5432
      name: "modelcards"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelcard-generator
  namespace: modelcard-system
  labels:
    app: modelcard-generator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: modelcard-generator
  template:
    metadata:
      labels:
        app: modelcard-generator
    spec:
      containers:
      - name: modelcard-generator
        image: modelcard-generator:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: CONFIG_FILE
          value: "/etc/config/config.yaml"
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        resources:
          requests:
            memory: "512Mi"
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
      volumes:
      - name: config-volume
        configMap:
          name: modelcard-config
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: modelcard-service
  namespace: modelcard-system
spec:
  selector:
    app: modelcard-generator
  ports:
  - name: http
    port: 80
    targetPort: 8080
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: modelcard-ingress
  namespace: modelcard-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.modelcard-generator.com
    secretName: modelcard-tls
  rules:
  - host: api.modelcard-generator.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: modelcard-service
            port:
              number: 80
```

## Cloud Platform Deployment

### AWS

#### ECS Deployment

```yaml
# task-definition.yaml
family: modelcard-generator
networkMode: awsvpc
requiresCompatibilities:
  - FARGATE
cpu: '512'
memory: '1024'
executionRoleArn: arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole
taskRoleArn: arn:aws:iam::ACCOUNT:role/ecsTaskRole

containerDefinitions:
  - name: modelcard-generator
    image: ACCOUNT.dkr.ecr.REGION.amazonaws.com/modelcard-generator:latest
    portMappings:
      - containerPort: 8080
        protocol: tcp
    environment:
      - name: ENVIRONMENT
        value: production
      - name: AWS_REGION
        value: us-east-1
    logConfiguration:
      logDriver: awslogs
      options:
        awslogs-group: /ecs/modelcard-generator
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs
```

#### EKS Deployment

```bash
# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=modelcard-cluster

# Deploy application
kubectl apply -f k8s/
```

### Google Cloud Platform

#### Cloud Run Deployment

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: modelcard-generator
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/PROJECT/modelcard-generator:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: production
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
```

Deploy with:
```bash
gcloud run services replace service.yaml --region=us-central1
```

#### GKE Deployment

```bash
# Create cluster
gcloud container clusters create modelcard-cluster \
  --zone=us-central1-a \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10

# Deploy application
kubectl apply -f k8s/
```

### Azure

#### Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name modelcard-generator \
  --image myregistry.azurecr.io/modelcard-generator:latest \
  --dns-name-label modelcard-generator \
  --ports 8080 \
  --environment-variables ENVIRONMENT=production
```

#### AKS Deployment

```bash
# Create cluster
az aks create \
  --resource-group myResourceGroup \
  --name modelcard-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Deploy application
kubectl apply -f k8s/
```

## Monitoring and Observability

### Prometheus Metrics

```yaml
# ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: modelcard-generator
  namespace: modelcard-system
spec:
  selector:
    matchLabels:
      app: modelcard-generator
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Grafana Dashboard

Import the provided Grafana dashboard (`monitoring/grafana/dashboard.json`) to monitor:

- Request rate and latency
- Error rates
- Model card generation throughput
- Cache hit rates
- Resource utilization

### Logging

Configure structured logging:

```python
# config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'json',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

## Security

### Authentication

Configure authentication:

```python
# API Key authentication
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_api_key(credentials: HTTPCredentials = Security(security)):
    if credentials.credentials != expected_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials
```

### HTTPS/TLS

Configure TLS termination:

```yaml
# In Ingress
spec:
  tls:
  - hosts:
    - api.modelcard-generator.com
    secretName: tls-secret
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: modelcard-network-policy
  namespace: modelcard-system
spec:
  podSelector:
    matchLabels:
      app: modelcard-generator
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

## High Availability

### Multi-Region Deployment

Deploy across multiple regions for high availability:

```yaml
# Global load balancer configuration
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: modelcard-ssl-cert
spec:
  domains:
    - api.modelcard-generator.com
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: modelcard-global-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "modelcard-ip"
    networking.gke.io/managed-certificates: "modelcard-ssl-cert"
    kubernetes.io/ingress.class: "gce"
spec:
  rules:
  - host: api.modelcard-generator.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: modelcard-service
            port:
              number: 80
```

### Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: modelcard-hpa
  namespace: modelcard-system
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
```

## Backup and Disaster Recovery

### Database Backup

```bash
# PostgreSQL backup
kubectl exec -n modelcard-system postgres-0 -- pg_dump -U mcg modelcards > backup.sql

# Restore
kubectl exec -i -n modelcard-system postgres-0 -- psql -U mcg modelcards < backup.sql
```

### Configuration Backup

```bash
# Backup all configurations
kubectl get configmaps,secrets -n modelcard-system -o yaml > config-backup.yaml

# Restore
kubectl apply -f config-backup.yaml
```

## Performance Tuning

### Resource Optimization

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Caching

Configure Redis for caching:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: modelcard-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "125m"
          limits:
            memory: "512Mi"
            cpu: "250m"
```

## Troubleshooting

### Common Issues

1. **Pod crashes**: Check logs with `kubectl logs`
2. **Service unavailable**: Verify service endpoints
3. **Performance issues**: Check resource utilization
4. **Database connection**: Verify network policies

### Debug Commands

```bash
# Check pod status
kubectl get pods -n modelcard-system

# View logs
kubectl logs -f deployment/modelcard-generator -n modelcard-system

# Describe resources
kubectl describe pod POD_NAME -n modelcard-system

# Execute into pod
kubectl exec -it POD_NAME -n modelcard-system -- /bin/bash

# Port forward for debugging
kubectl port-forward service/modelcard-service 8080:80 -n modelcard-system
```

## Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/modelcard-generator \
  modelcard-generator=modelcard-generator:v2.0.0 \
  -n modelcard-system

# Check rollout status
kubectl rollout status deployment/modelcard-generator -n modelcard-system

# Rollback if needed
kubectl rollout undo deployment/modelcard-generator -n modelcard-system
```

### Health Checks

Implement comprehensive health checks:

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    # Check database connectivity
    # Check cache connectivity
    # Check external dependencies
    return {"status": "ready"}
```

## Next Steps

1. **Set up monitoring**: Configure Prometheus and Grafana
2. **Implement CI/CD**: Automate deployments
3. **Configure backups**: Set up regular backups
4. **Load testing**: Test under production load
5. **Documentation**: Document operational procedures
