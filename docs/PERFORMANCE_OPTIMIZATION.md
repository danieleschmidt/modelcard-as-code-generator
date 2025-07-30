# Performance Optimization Guide

This document outlines performance optimization strategies, benchmarking procedures, and monitoring approaches for the Model Card Generator.

## Overview

Performance optimization is critical for enterprise adoption, especially when processing large-scale model documentation workflows. This guide covers optimization strategies across all system components.

## Performance Targets

### Response Time SLAs
- **CLI Commands**: < 2 seconds for standard operations
- **Template Rendering**: < 500ms for standard templates
- **Validation**: < 1 second for comprehensive validation
- **Export Operations**: < 5 seconds for multi-format exports

### Throughput Targets
- **Batch Processing**: 100+ model cards per minute
- **Concurrent Users**: Support 50+ simultaneous operations
- **Memory Usage**: < 512MB peak memory for standard workflows
- **CPU Efficiency**: < 2 CPU cores for typical operations

## Code-Level Optimizations

### 1. Template Rendering Optimization

```python
# Efficient template caching
from functools import lru_cache
from jinja2 import Environment, FileSystemLoader

class OptimizedTemplateRenderer:
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader('templates'),
            cache_size=100,  # Cache compiled templates
            auto_reload=False  # Disable in production
        )
    
    @lru_cache(maxsize=50)
    def get_template(self, template_name: str):
        return self.env.get_template(template_name)
```

### 2. Data Processing Optimization

```python
# Efficient data processing with generators
def process_large_datasets(data_source):
    """Process large datasets without loading everything into memory."""
    for chunk in read_data_chunks(data_source, chunk_size=1000):
        yield process_chunk(chunk)

# Parallel processing for independent operations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def parallel_validation(cards: List[ModelCard]) -> List[ValidationResult]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(validate_card, card) for card in cards]
        return [future.result() for future in futures]
```

### 3. Memory Management

```python
# Context managers for resource cleanup
from contextlib import contextmanager

@contextmanager
def managed_resources():
    resources = []
    try:
        yield resources
    finally:
        for resource in resources:
            resource.cleanup()

# Lazy loading for large objects
class LazyModelCard:
    def __init__(self, path: str):
        self._path = path
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            self._data = load_model_card(self._path)
        return self._data
```

## Database Optimization

### Query Optimization
```sql
-- Index optimization for frequent queries
CREATE INDEX idx_model_cards_created_at ON model_cards(created_at);
CREATE INDEX idx_model_cards_status ON model_cards(status);
CREATE INDEX idx_model_cards_composite ON model_cards(status, created_at);

-- Efficient pagination
SELECT * FROM model_cards 
WHERE created_at < :cursor_date 
ORDER BY created_at DESC 
LIMIT 25;
```

### Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Optimized database configuration
engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Caching Strategies

### 1. Multi-Level Caching

```python
# Redis for distributed caching
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiry=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiry, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### 2. Template Compilation Caching

```python
# Pre-compile templates during startup
class TemplateCache:
    def __init__(self):
        self.compiled_templates = {}
        self._precompile_templates()
    
    def _precompile_templates(self):
        """Pre-compile all templates during initialization."""
        template_dir = Path("templates")
        for template_file in template_dir.glob("**/*.j2"):
            template_name = str(template_file.relative_to(template_dir))
            self.compiled_templates[template_name] = self.env.get_template(template_name)
```

## Asynchronous Processing

### 1. Async/Await for I/O Operations

```python
import asyncio
import aiohttp
from typing import List

async def fetch_model_metadata(urls: List[str]) -> List[dict]:
    """Fetch metadata from multiple sources concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_metadata(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch_single_metadata(session, url):
    async with session.get(url) as response:
        return await response.json()
```

### 2. Background Task Processing

```python
from celery import Celery

# Celery configuration for background tasks
app = Celery('modelcard_generator')
app.config_from_object('celeryconfig')

@app.task
def generate_model_card_async(config_data):
    """Generate model card in background."""
    generator = ModelCardGenerator(config_data)
    return generator.generate()

@app.task
def bulk_validation_task(card_paths):
    """Validate multiple cards in background."""
    return [validate_card(path) for path in card_paths]
```

## Monitoring and Profiling

### 1. Performance Monitoring

```python
import time
import psutil
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name: str):
    """Monitor CPU, memory, and execution time."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(f"{operation_name}: {duration:.2f}s, {memory_delta/1024/1024:.1f}MB")
```

### 2. Profiling Integration

```python
import cProfile
import pstats
from functools import wraps

def profile_performance(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 slowest functions
        
        return result
    return wrapper
```

## Benchmarking Framework

### 1. Automated Benchmarks

```python
import pytest
import time
from typing import Callable, Any

class PerformanceBenchmark:
    def __init__(self, name: str, target_time: float):
        self.name = name
        self.target_time = target_time
        self.results = []
    
    def benchmark(self, func: Callable, *args, **kwargs) -> float:
        """Run benchmark and return execution time."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        self.results.append(duration)
        
        assert duration <= self.target_time, f"{self.name} exceeded target time: {duration:.2f}s > {self.target_time}s"
        return duration

# Benchmark tests
def test_template_rendering_performance():
    benchmark = PerformanceBenchmark("Template Rendering", 0.5)
    
    generator = ModelCardGenerator()
    template_data = get_sample_template_data()
    
    benchmark.benchmark(generator.render_template, "huggingface", template_data)

def test_validation_performance():
    benchmark = PerformanceBenchmark("Card Validation", 1.0)
    
    validator = ModelCardValidator()
    card = load_sample_card()
    
    benchmark.benchmark(validator.validate, card)
```

### 2. Load Testing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

def load_test_concurrent_operations(operation_func, concurrency=10, iterations=100):
    """Load test with concurrent operations."""
    
    def run_operation():
        start_time = time.time()
        operation_func()
        return time.time() - start_time
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_operation) for _ in range(iterations)]
        times = [future.result() for future in as_completed(futures)]
    
    return {
        'min_time': min(times),
        'max_time': max(times),
        'avg_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'p95_time': statistics.quantiles(times, n=20)[18],  # 95th percentile
        'p99_time': statistics.quantiles(times, n=100)[98]  # 99th percentile
    }
```

## Infrastructure Optimization

### 1. Container Optimization

```dockerfile
# Multi-stage build for smaller images
FROM python:3.11-slim as builder

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --user build && python -m build

FROM python:3.11-slim as runtime

# Install only runtime dependencies
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Optimize for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY src/ ./src/

# Use non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

CMD ["python", "-m", "modelcard_generator.cli"]
```

### 2. Kubernetes Resource Optimization

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelcard-generator
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: modelcard-generator
        image: terragonlabs/modelcard-generator:latest
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: WORKER_PROCESSES
          value: "2"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 10
```

## Continuous Performance Monitoring

### 1. Performance Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Prometheus metrics
OPERATION_DURATION = Histogram('operation_duration_seconds', 'Operation duration', ['operation'])
ACTIVE_OPERATIONS = Gauge('active_operations', 'Number of active operations')
TOTAL_OPERATIONS = Counter('total_operations', 'Total operations', ['operation', 'status'])

def monitor_performance(operation_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ACTIVE_OPERATIONS.inc()
            
            with OPERATION_DURATION.labels(operation=operation_name).time():
                try:
                    result = func(*args, **kwargs)
                    TOTAL_OPERATIONS.labels(operation=operation_name, status='success').inc()
                    return result
                except Exception as e:
                    TOTAL_OPERATIONS.labels(operation=operation_name, status='error').inc()
                    raise
                finally:
                    ACTIVE_OPERATIONS.dec()
        return wrapper
    return decorator
```

### 2. Performance Alerting

```yaml
# Prometheus alerting rules
groups:
- name: modelcard_generator
  rules:
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, operation_duration_seconds) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }}s"

  - alert: HighErrorRate
    expr: rate(total_operations{status="error"}[5m]) / rate(total_operations[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }}"
```

## Performance Testing in CI/CD

### 1. Automated Performance Tests

```python
# Performance regression tests
@pytest.mark.performance
def test_template_rendering_regression():
    """Ensure template rendering doesn't regress."""
    
    # Load baseline performance data
    baseline = load_performance_baseline('template_rendering')
    
    # Run current performance test
    current_time = benchmark_template_rendering()
    
    # Allow 10% performance regression
    threshold = baseline * 1.1
    assert current_time <= threshold, f"Performance regression: {current_time}s > {threshold}s"

def benchmark_template_rendering():
    generator = ModelCardGenerator()
    data = get_large_sample_data()
    
    start_time = time.perf_counter()
    generator.render_template('huggingface', data)
    end_time = time.perf_counter()
    
    return end_time - start_time
```

### 2. Performance Report Generation

```python
def generate_performance_report(results: dict) -> str:
    """Generate performance report for CI/CD."""
    
    report = """
# Performance Test Results

## Summary
- **Template Rendering**: {template_time:.2f}s (target: <0.5s)
- **Validation**: {validation_time:.2f}s (target: <1.0s)
- **Export**: {export_time:.2f}s (target: <5.0s)

## Recommendations
{recommendations}

## Detailed Results
{detailed_results}
""".format(
        template_time=results['template_rendering'],
        validation_time=results['validation'],
        export_time=results['export'],
        recommendations=generate_recommendations(results),
        detailed_results=format_detailed_results(results)
    )
    
    return report
```

## Best Practices

### 1. Code Optimization
- Use generators for large data processing
- Implement lazy loading for expensive operations
- Cache frequently accessed data
- Use appropriate data structures (sets for membership tests, deques for queues)

### 2. Database Optimization
- Add indexes for frequently queried fields
- Use connection pooling
- Implement query result caching
- Optimize database schema design

### 3. Memory Management
- Use context managers for resource cleanup
- Implement memory profiling in development
- Monitor memory usage in production
- Use memory-efficient data structures

### 4. Monitoring
- Implement comprehensive performance monitoring
- Set up alerting for performance regressions
- Use distributed tracing for complex operations
- Regular performance testing in CI/CD

This performance optimization guide ensures the Model Card Generator maintains high performance standards while scaling to meet enterprise requirements.