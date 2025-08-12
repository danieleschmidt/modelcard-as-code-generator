# ModelCard Generator - API Reference

## Python SDK

### Core Classes

#### ModelCardGenerator

The main entry point for generating model cards programmatically.

```python
from modelcard_generator import ModelCardGenerator, CardConfig

class ModelCardGenerator:
    def __init__(self, config: Optional[CardConfig] = None):
        """
        Initialize the ModelCard Generator.
        
        Args:
            config: Configuration object for generation settings
        """
        
    def generate(
        self,
        eval_results: Optional[Union[str, Dict]] = None,
        training_logs: Optional[Union[str, List[str]]] = None,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> ModelCard:
        """
        Generate a model card from provided data sources.
        
        Args:
            eval_results: Path to evaluation results or dict
            training_logs: Path(s) to training logs
            model_path: Path to the model directory
            output_path: Where to save the generated card
            format: Output format ('huggingface', 'google', 'eu_cra')
            **kwargs: Additional parameters for specific formats
            
        Returns:
            ModelCard: Generated model card object
            
        Raises:
            ValidationError: If input data is invalid
            GenerationError: If card generation fails
        """
```

#### Enhanced Features (Production Extensions)

```python
from modelcard_generator.core.enhanced_generator import EnhancedModelCardGenerator

class EnhancedModelCardGenerator(ModelCardGenerator):
    def __init__(
        self,
        config: Optional[CardConfig] = None,
        enable_resilience: bool = True,
        enable_caching: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize enhanced generator with production features.
        
        Args:
            config: Base configuration
            enable_resilience: Enable fault tolerance patterns
            enable_caching: Enable intelligent caching
            enable_monitoring: Enable metrics collection
        """
        
    async def generate_batch(
        self,
        batch_specs: List[Dict[str, Any]],
        max_concurrency: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> List[ModelCard]:
        """
        Generate multiple model cards concurrently.
        
        Args:
            batch_specs: List of generation specifications
            max_concurrency: Maximum concurrent generations
            progress_callback: Optional progress reporting function
            
        Returns:
            List of generated model cards
        """
        
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about generation operations."""
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
```

### Configuration

#### CardConfig

```python
from modelcard_generator.core.models import CardConfig, CardFormat

@dataclass
class CardConfig:
    format: CardFormat = CardFormat.HUGGINGFACE
    template_path: Optional[str] = None
    output_dir: str = "model_cards"
    validate_output: bool = True
    include_plots: bool = True
    auto_infer_metadata: bool = True
    compliance_standards: List[str] = field(default_factory=list)
    custom_sections: Dict[str, Any] = field(default_factory=dict)
```

#### Advanced Configuration

```python
from modelcard_generator.core.config import ModelCardConfig

config = ModelCardConfig(
    # Logging configuration
    logging=LoggingConfig(
        level="INFO",
        structured=True,
        max_bytes=10*1024*1024
    ),
    
    # Security settings
    security=SecurityConfig(
        enable_scanning=True,
        max_file_size=100*1024*1024,
        scan_content=True
    ),
    
    # Validation rules
    validation=ValidationConfig(
        min_completeness_score=0.8,
        enforce_compliance=True,
        compliance_standards=["eu_cra", "gdpr"]
    ),
    
    # Performance tuning
    cache=CacheConfig(
        enabled=True,
        ttl_seconds=3600,
        max_size_mb=500
    )
)
```

### Resilience Patterns

#### Circuit Breaker and Fault Tolerance

```python
from modelcard_generator.core.resilience import (
    resilient_operation,
    AdaptiveTimeout,
    Bulkhead,
    GracefulDegradation
)

# Use resilient decorator
@resilient_operation(
    max_retries=3,
    backoff_multiplier=2.0,
    timeout_seconds=30.0
)
async def external_api_call():
    # Your API call logic
    pass

# Manual resilience patterns
async def example_usage():
    # Adaptive timeout based on historical performance
    timeout_manager = AdaptiveTimeout(
        initial_timeout=10.0,
        max_timeout=60.0
    )
    
    # Resource isolation
    bulkhead = Bulkhead(max_concurrent=5)
    
    # Graceful degradation
    degradation = GracefulDegradation({
        "high_quality": lambda: full_generation(),
        "basic": lambda: minimal_generation(),
        "fallback": lambda: template_only()
    })
```

### Intelligent Caching

#### Multi-layer Cache System

```python
from modelcard_generator.core.intelligent_cache import (
    IntelligentCache,
    cache_with_intelligence,
    initialize_intelligent_cache
)

# Initialize global cache
await initialize_intelligent_cache(
    memory_cache_mb=100,
    disk_cache_mb=1000,
    redis_url="redis://localhost:6379",
    enable_prefetching=True
)

# Use caching decorator
@cache_with_intelligence(
    cache_instance=global_cache,
    ttl_seconds=3600,
    cache_key_func=lambda model_id, version: f"{model_id}:{version}"
)
async def expensive_computation(model_id: str, version: str):
    # Expensive operation that benefits from caching
    return results

# Manual cache operations
cache = IntelligentCache()
await cache.start()

# Store and retrieve
await cache.put("key", data, ttl_seconds=3600)
result = await cache.get("key")

# Get comprehensive statistics
stats = cache.get_comprehensive_stats()
```

### Distributed Processing

#### Task Queue and Auto-scaling

```python
from modelcard_generator.core.distributed_processing import (
    DistributedTaskQueue,
    DistributedWorker,
    AutoScaler
)

# Set up distributed processing
task_queue = DistributedTaskQueue(
    redis_url="redis://localhost:6379",
    queue_name="modelcard_generation"
)

# Submit tasks
task_id = await task_queue.submit_task(
    "generate_modelcard",
    {
        "eval_results": "path/to/results.json",
        "model_path": "path/to/model",
        "format": "huggingface"
    },
    priority=1
)

# Check task status
status = await task_queue.get_task_status(task_id)
result = await task_queue.get_task_result(task_id)

# Auto-scaling worker pool
autoscaler = AutoScaler(
    min_workers=2,
    max_workers=10,
    scale_up_threshold=0.8,
    scale_down_threshold=0.2
)

await autoscaler.start()
```

### Monitoring and Metrics

#### Performance Monitoring

```python
from modelcard_generator.monitoring.enhanced_metrics import (
    SystemMonitor,
    PerformanceTracker,
    AlertManager
)

# System monitoring
monitor = SystemMonitor()
await monitor.start_monitoring()

# Performance tracking
tracker = PerformanceTracker()

# Track operation performance
async with tracker.track_operation("model_card_generation") as context:
    context.add_metadata({"format": "huggingface", "size": "large"})
    # Your generation logic here
    result = await generate_card()
    context.set_result_size(len(str(result)))

# Get performance metrics
metrics = tracker.get_metrics_summary()

# Alert management
alert_manager = AlertManager()
alert_manager.add_alert_rule(
    "high_error_rate",
    condition=lambda metrics: metrics.error_rate > 0.1,
    notification_channels=["email", "slack"]
)
```

### Security Scanning

#### Content Security and Compliance

```python
from modelcard_generator.security.advanced_scanner import (
    ContentSecurityScanner,
    ModelSecurityValidator,
    SecurityReportGenerator
)

# Content security scanning
scanner = ContentSecurityScanner()

# Scan model card content
scan_result = await scanner.scan_content(model_card_text)

if scan_result.has_issues():
    for issue in scan_result.issues:
        print(f"Security issue: {issue.type} - {issue.description}")

# Model security validation
validator = ModelSecurityValidator()
validation_result = await validator.validate_model(model_path)

# Generate comprehensive security report
report_generator = SecurityReportGenerator()
security_report = await report_generator.generate_report([
    scan_result,
    validation_result
])
```

## CLI Interface

### Basic Commands

```bash
# Generate model card
mcg generate --eval-results results.json --format huggingface --output MODEL_CARD.md

# Batch generation
mcg generate-batch --config batch_config.yaml --max-concurrency 5

# Validate existing card
mcg validate MODEL_CARD.md --standard huggingface

# Check for drift
mcg check-drift MODEL_CARD.md --new-data recent_results.json

# Security scanning
mcg scan-security MODEL_CARD.md --detailed-report

# Performance monitoring
mcg monitor --duration 3600 --export-metrics metrics.json
```

### Advanced Commands

```bash
# Enhanced generation with all features
mcg generate-enhanced \
    --eval-results results.json \
    --training-logs logs/ \
    --format huggingface \
    --enable-caching \
    --enable-resilience \
    --enable-monitoring \
    --output-dir ./cards/

# Distributed processing
mcg process-distributed \
    --task-specs tasks.json \
    --redis-url redis://localhost:6379 \
    --workers 5

# Cache management
mcg cache status
mcg cache clear --confirm
mcg cache warm --patterns patterns.yaml

# System diagnostics
mcg diagnose --check-all --export-report diagnosis.json
```

## REST API (Future Enhancement)

### Endpoints

```http
# Generate model card
POST /api/v1/generate
Content-Type: application/json

{
  "eval_results": {...},
  "training_logs": [...],
  "format": "huggingface",
  "config": {...}
}

# Get generation status
GET /api/v1/tasks/{task_id}/status

# Get generated card
GET /api/v1/tasks/{task_id}/result

# Health check
GET /health

# Metrics
GET /metrics
```

## Error Handling

### Exception Hierarchy

```python
from modelcard_generator.core.exceptions import (
    ModelCardError,
    ValidationError,
    GenerationError,
    ResourceError,
    ConfigurationError
)

try:
    card = generator.generate(eval_results="invalid.json")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Field: {e.field}")
    print(f"Suggestions: {e.suggestions}")
except GenerationError as e:
    print(f"Generation failed: {e.message}")
    print(f"Phase: {e.phase}")
    print(f"Recoverable: {e.recoverable}")
except ResourceError as e:
    print(f"Resource issue: {e.resource_type}")
    print(f"Error: {e.message}")
```

## Integration Examples

### MLflow Integration

```python
import mlflow
from modelcard_generator import ModelCardGenerator

# Generate card from MLflow run
with mlflow.start_run():
    # Your training code
    mlflow.log_metrics({"accuracy": 0.95})
    mlflow.log_artifacts("model/")
    
    # Generate model card
    generator = ModelCardGenerator()
    card = generator.generate_from_mlflow_run(
        run_id=mlflow.active_run().info.run_id,
        format="huggingface"
    )
```

### Weights & Biases Integration

```python
import wandb
from modelcard_generator.integrations.wandb import WandBCollector

# Generate from W&B run
collector = WandBCollector(api_key="your_key")
data = collector.collect_from_run(
    project="my-project",
    run_id="run_123"
)

generator = ModelCardGenerator()
card = generator.generate(**data)
```

### GitHub Actions Integration

```yaml
name: Generate Model Card
on:
  push:
    paths: ['models/**']

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: modelcard-generator/action@v1
        with:
          eval-results: 'results/metrics.json'
          training-logs: 'logs/'
          format: 'huggingface'
          output: 'MODEL_CARD.md'
```

## Performance Guidelines

### Optimization Best Practices

1. **Enable Caching**: Reduces computation for repeated operations
2. **Use Batch Processing**: More efficient for multiple cards
3. **Configure Timeouts**: Prevent hanging operations
4. **Monitor Memory**: Use profiling for large models
5. **Leverage Parallelization**: Process independent tasks concurrently

### Resource Management

```python
# Memory-efficient processing
config = ModelCardConfig(
    cache=CacheConfig(max_size_mb=200),  # Limit cache size
    validation=ValidationConfig(
        min_completeness_score=0.7  # Reduce validation overhead
    )
)

# Batch processing with memory management
async def process_large_batch(specs: List[Dict]):
    # Process in chunks to manage memory
    chunk_size = 10
    for i in range(0, len(specs), chunk_size):
        chunk = specs[i:i + chunk_size]
        results = await generator.generate_batch(
            chunk,
            max_concurrency=5
        )
        # Process results and free memory
        yield results
```

This API reference provides comprehensive documentation for all the enhanced features implemented during the autonomous SDLC execution, maintaining backward compatibility while exposing powerful new capabilities for production environments.