# ModelCard Generator - Usage Examples

## Quick Start Examples

### Basic Model Card Generation

```python
from modelcard_generator import ModelCardGenerator, CardConfig

# Simple generation
generator = ModelCardGenerator()
card = generator.generate(
    eval_results="path/to/evaluation_results.json",
    model_path="path/to/model/",
    format="huggingface"
)

# Save the card
with open("MODEL_CARD.md", "w") as f:
    f.write(str(card))
```

### Enhanced Production Generation

```python
from modelcard_generator.core.enhanced_generator import EnhancedModelCardGenerator
from modelcard_generator.core.config import ModelCardConfig

# Production configuration
config = ModelCardConfig(
    # Enable all production features
    cache=CacheConfig(enabled=True, max_size_mb=500),
    security=SecurityConfig(enable_scanning=True),
    monitoring=MonitoringConfig(enabled=True)
)

# Enhanced generator with resilience
generator = EnhancedModelCardGenerator(
    config=config,
    enable_resilience=True,
    enable_caching=True,
    enable_monitoring=True
)

# Generate with comprehensive statistics
card = await generator.generate(
    eval_results="results.json",
    training_logs=["training.log", "validation.log"],
    model_path="./model/",
    format="huggingface"
)

# Get detailed performance metrics
stats = generator.get_generation_statistics()
performance = generator.get_performance_report()

print(f"Generation time: {stats['total_time_seconds']:.2f}s")
print(f"Cache hit rate: {performance['cache_hit_rate']:.2%}")
```

## CLI Examples

### Basic CLI Usage

```bash
# Generate model card from evaluation results
mcg generate \
    --eval-results evaluation_results.json \
    --format huggingface \
    --output MODEL_CARD.md

# Include training logs
mcg generate \
    --eval-results results.json \
    --training-logs training.log validation.log \
    --format google \
    --output google_model_card.md

# Validate existing card
mcg validate MODEL_CARD.md --standard huggingface

# Check for drift between cards
mcg check-drift old_card.md --new-data new_results.json
```

### Advanced CLI Features

```bash
# Enhanced generation with all features
mcg generate-enhanced \
    --eval-results results.json \
    --training-logs logs/ \
    --model-path ./model/ \
    --format eu_cra \
    --enable-caching \
    --enable-resilience \
    --enable-monitoring \
    --config production.yaml \
    --output MODEL_CARD.md

# Batch processing
mcg generate-batch \
    --config-dir batch_configs/ \
    --max-concurrency 5 \
    --output-dir generated_cards/

# Distributed processing
mcg process-distributed \
    --task-specs distributed_tasks.json \
    --redis-url redis://localhost:6379 \
    --workers 3

# System monitoring
mcg monitor \
    --duration 3600 \
    --export-metrics metrics.json \
    --alert-thresholds alerts.yaml

# Cache management
mcg cache status
mcg cache warm --patterns cache_patterns.yaml
mcg cache clear --confirm
```

## Real-World Scenarios

### Scenario 1: MLOps Pipeline Integration

```python
"""
Integration with MLflow experiment tracking
"""
import mlflow
import mlflow.sklearn
from modelcard_generator import ModelCardGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json

# Training phase
with mlflow.start_run() as run:
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Evaluate and log metrics
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    
    # Log evaluation metrics
    mlflow.log_metrics({
        "accuracy": report['accuracy'],
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall'],
        "f1": report['macro avg']['f1-score']
    })
    
    # Save evaluation results for model card
    eval_results = {
        "model_name": "fraud_detection_rf",
        "model_version": "1.0.0",
        "dataset_name": "fraud_transactions",
        "metrics": report,
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42
        },
        "training_data": {
            "size": len(X_train),
            "features": list(X_train.columns),
            "target_distribution": dict(pd.Series(y_train).value_counts())
        }
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

# Model card generation
generator = ModelCardGenerator()
card = generator.generate(
    eval_results="evaluation_results.json",
    model_path=f"mlruns/{run.info.experiment_id}/{run.info.run_id}/artifacts/model",
    format="huggingface"
)

# Save to repository
with open("MODEL_CARD.md", "w") as f:
    f.write(str(card))
```

### Scenario 2: Regulatory Compliance (EU AI Act)

```python
"""
Generate EU AI Act compliant model card
"""
from modelcard_generator.core.enhanced_generator import EnhancedModelCardGenerator
from modelcard_generator.core.config import ModelCardConfig, ValidationConfig

# EU AI Act specific configuration
config = ModelCardConfig(
    validation=ValidationConfig(
        enforce_compliance=True,
        compliance_standards=["eu_ai_act", "gdpr"],
        min_completeness_score=0.95
    ),
    security=SecurityConfig(
        enable_scanning=True,
        scan_content=True,
        allow_external_urls=False
    )
)

generator = EnhancedModelCardGenerator(config=config)

# High-risk AI system data
eval_data = {
    "model_name": "credit_scoring_system",
    "model_type": "high_risk_ai_system",
    "intended_use": {
        "purpose": "Credit risk assessment for loan applications",
        "users": ["Financial institutions", "Loan officers"],
        "use_cases": ["Personal loans", "Business loans", "Mortgage applications"]
    },
    "risk_assessment": {
        "risk_category": "high_risk",
        "prohibited_uses": [
            "Social scoring",
            "Predictive policing",
            "Emotion recognition in workplace"
        ],
        "mitigation_measures": [
            "Human oversight required",
            "Bias monitoring implemented",
            "Regular model auditing"
        ]
    },
    "data_governance": {
        "data_sources": ["Credit history", "Income verification", "Employment records"],
        "data_protection": "GDPR compliant",
        "data_retention": "7 years as per financial regulations",
        "data_minimization": "Only relevant features used"
    },
    "fairness_assessment": {
        "protected_attributes": ["age", "gender", "ethnicity", "marital_status"],
        "bias_metrics": {
            "demographic_parity": 0.95,
            "equalized_odds": 0.92,
            "calibration": 0.98
        },
        "fairness_constraints": "Disparate impact < 0.8 threshold"
    }
}

# Generate compliant model card
card = await generator.generate(
    eval_results=eval_data,
    format="eu_cra",
    compliance_check=True
)

# Validate compliance
validation_result = await generator.validate_compliance(card, "eu_ai_act")
if not validation_result.is_compliant:
    print("Compliance issues found:")
    for issue in validation_result.issues:
        print(f"- {issue}")
```

### Scenario 3: Distributed Model Card Generation

```python
"""
Large-scale batch processing with distributed workers
"""
import asyncio
from modelcard_generator.core.distributed_processing import (
    DistributedTaskQueue,
    AutoScaler,
    LoadBalancer
)

async def distributed_generation_example():
    # Initialize distributed components
    task_queue = DistributedTaskQueue(
        redis_url="redis://redis-cluster:6379",
        queue_name="model_cards"
    )
    
    autoscaler = AutoScaler(
        min_workers=3,
        max_workers=20,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3
    )
    
    load_balancer = LoadBalancer(
        strategy="least_connections"
    )
    
    # Start distributed system
    await task_queue.start()
    await autoscaler.start()
    await load_balancer.start()
    
    # Submit batch of model card generation tasks
    model_specs = [
        {
            "model_id": f"model_{i}",
            "eval_results": f"results/model_{i}/eval.json",
            "training_logs": f"logs/model_{i}/",
            "format": "huggingface" if i % 2 == 0 else "google"
        }
        for i in range(100)  # 100 models to process
    ]
    
    # Submit all tasks
    task_ids = []
    for spec in model_specs:
        task_id = await task_queue.submit_task(
            "generate_modelcard",
            spec,
            priority=1
        )
        task_ids.append(task_id)
        print(f"Submitted task {task_id} for {spec['model_id']}")
    
    # Monitor progress
    completed = 0
    failed = 0
    
    while completed + failed < len(task_ids):
        await asyncio.sleep(5)  # Check every 5 seconds
        
        for task_id in task_ids:
            status = await task_queue.get_task_status(task_id)
            if status.state == "completed" and task_id not in processed:
                result = await task_queue.get_task_result(task_id)
                completed += 1
                print(f"✓ Task {task_id} completed")
            elif status.state == "failed":
                failed += 1
                print(f"✗ Task {task_id} failed: {status.error}")
        
        # Get system metrics
        metrics = await autoscaler.get_metrics()
        print(f"Workers: {metrics['active_workers']}, Queue: {metrics['queue_size']}")
    
    print(f"Batch completed: {completed} successful, {failed} failed")

# Run the distributed generation
asyncio.run(distributed_generation_example())
```

### Scenario 4: Real-time Monitoring Dashboard

```python
"""
Real-time monitoring and alerting system
"""
from modelcard_generator.monitoring.enhanced_metrics import (
    SystemMonitor,
    PerformanceTracker,
    AlertManager
)
import asyncio
import json

async def monitoring_dashboard():
    # Initialize monitoring components
    system_monitor = SystemMonitor()
    performance_tracker = PerformanceTracker()
    alert_manager = AlertManager()
    
    # Configure alerts
    alert_manager.add_alert_rule(
        "high_error_rate",
        condition=lambda m: m.error_rate > 0.05,
        notification_channels=["email", "slack"],
        cooldown_minutes=15
    )
    
    alert_manager.add_alert_rule(
        "high_memory_usage",
        condition=lambda m: m.memory_usage_percent > 80,
        notification_channels=["email"],
        cooldown_minutes=5
    )
    
    alert_manager.add_alert_rule(
        "slow_generation_time",
        condition=lambda m: m.avg_generation_time > 60,
        notification_channels=["slack"],
        cooldown_minutes=30
    )
    
    # Start monitoring
    await system_monitor.start_monitoring()
    await performance_tracker.start()
    await alert_manager.start()
    
    # Real-time dashboard loop
    try:
        while True:
            # Collect current metrics
            system_metrics = await system_monitor.get_current_metrics()
            perf_metrics = performance_tracker.get_metrics_summary()
            
            # Create dashboard data
            dashboard = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "disk_usage": system_metrics.disk_usage,
                    "network_io": system_metrics.network_io
                },
                "application": {
                    "total_requests": perf_metrics.total_requests,
                    "success_rate": perf_metrics.success_rate,
                    "avg_response_time": perf_metrics.avg_response_time,
                    "active_workers": perf_metrics.active_workers
                },
                "generation": {
                    "cards_generated": perf_metrics.cards_generated,
                    "avg_generation_time": perf_metrics.avg_generation_time,
                    "cache_hit_rate": perf_metrics.cache_hit_rate,
                    "error_rate": perf_metrics.error_rate
                }
            }
            
            # Update dashboard (could be WebSocket, file, or external system)
            with open("/tmp/dashboard.json", "w") as f:
                json.dump(dashboard, f, indent=2)
            
            # Check for alerts
            await alert_manager.check_alerts(perf_metrics)
            
            # Display key metrics
            print(f"🎯 Success Rate: {perf_metrics.success_rate:.1%}")
            print(f"⏱️  Avg Generation Time: {perf_metrics.avg_generation_time:.1f}s")
            print(f"💾 Cache Hit Rate: {perf_metrics.cache_hit_rate:.1%}")
            print(f"🔧 Active Workers: {perf_metrics.active_workers}")
            print("-" * 50)
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("Stopping monitoring dashboard...")
    finally:
        await system_monitor.stop()
        await performance_tracker.stop()
        await alert_manager.stop()

# Run monitoring dashboard
asyncio.run(monitoring_dashboard())
```

## Integration Patterns

### GitHub Actions Workflow

```yaml
name: Automated Model Card Generation
on:
  push:
    paths: 
      - 'models/**'
      - 'experiments/**'

jobs:
  generate-cards:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install ModelCard Generator
        run: |
          pip install modelcard-generator[all]
      
      - name: Generate Model Cards
        run: |
          # Generate cards for all models
          find models/ -name "eval_results.json" | while read eval_file; do
            model_dir=$(dirname "$eval_file")
            model_name=$(basename "$model_dir")
            
            mcg generate-enhanced \
              --eval-results "$eval_file" \
              --model-path "$model_dir" \
              --format huggingface \
              --enable-caching \
              --enable-monitoring \
              --output "$model_dir/MODEL_CARD.md"
          done
      
      - name: Validate Generated Cards
        run: |
          find models/ -name "MODEL_CARD.md" | while read card; do
            mcg validate "$card" --standard huggingface
          done
      
      - name: Security Scan
        run: |
          find models/ -name "MODEL_CARD.md" | while read card; do
            mcg scan-security "$card" --detailed-report
          done
      
      - name: Commit Generated Cards
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add models/*/MODEL_CARD.md
          if git diff --staged --quiet; then
            echo "No changes to commit"
          else
            git commit -m "🤖 Update model cards"
            git push
          fi
      
      - name: Create Summary
        run: |
          echo "## Model Card Generation Summary" >> $GITHUB_STEP_SUMMARY
          echo "Generated cards for $(find models/ -name 'MODEL_CARD.md' | wc -l) models" >> $GITHUB_STEP_SUMMARY
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  mcg-dev:
    build: .
    volumes:
      - ./:/workspace
      - mcg-cache:/app/cache
    environment:
      - MCG_LOG_LEVEL=DEBUG
      - MCG_ENABLE_CACHING=true
      - MCG_REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    ports:
      - "8080:8080"
      - "9090:9090"  # metrics
    command: ["mcg", "serve", "--host", "0.0.0.0", "--port", "8080"]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  jupyter:
    image: jupyter/scipy-notebook
    ports:
      - "8888:8888"
    volumes:
      - ./:/workspace
      - mcg-cache:/workspace/cache
    environment:
      - JUPYTER_ENABLE_LAB=yes

volumes:
  mcg-cache:
  redis-data:
```

### Pre-commit Hook Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: generate-model-cards
        name: Generate Model Cards
        entry: mcg
        language: system
        args: 
          - generate-batch
          - --config-dir
          - models/
          - --format
          - huggingface
        files: '^models/.*/(eval_results\.json|config\.yaml)$'
      
      - id: validate-model-cards
        name: Validate Model Cards
        entry: mcg
        language: system
        args:
          - validate-batch
          - --directory
          - models/
          - --standard
          - huggingface
        files: '^models/.*/MODEL_CARD\.md$'
      
      - id: security-scan
        name: Security Scan Model Cards
        entry: mcg
        language: system
        args:
          - scan-security-batch
          - --directory
          - models/
          - --fail-on-high
        files: '^models/.*/MODEL_CARD\.md$'
```

## Testing Examples

### Unit Testing with Production Features

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from modelcard_generator.core.enhanced_generator import EnhancedModelCardGenerator
from modelcard_generator.core.intelligent_cache import IntelligentCache

class TestEnhancedGeneration:
    
    @pytest.fixture
    async def generator(self):
        """Create enhanced generator for testing."""
        generator = EnhancedModelCardGenerator(
            enable_resilience=True,
            enable_caching=True,
            enable_monitoring=True
        )
        await generator.initialize()
        yield generator
        await generator.cleanup()
    
    @pytest.mark.asyncio
    async def test_resilient_generation(self, generator):
        """Test generation with resilience patterns."""
        # Mock a failing operation that succeeds on retry
        call_count = 0
        def mock_external_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"status": "success"}
        
        # This should succeed despite initial failures
        result = await generator._resilient_external_call(mock_external_call)
        assert result["status"] == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, generator):
        """Test intelligent caching system."""
        test_data = {"model": "test", "metrics": {"accuracy": 0.95}}
        
        # First call should miss cache
        result1 = await generator.generate(**test_data)
        stats1 = generator.cache.get_comprehensive_stats()
        
        # Second call should hit cache
        result2 = await generator.generate(**test_data)
        stats2 = generator.cache.get_comprehensive_stats()
        
        assert str(result1) == str(result2)  # Same result
        assert stats2["global"]["hit_rate"] > stats1["global"]["hit_rate"]
    
    def test_performance_monitoring(self, generator):
        """Test performance metrics collection."""
        # Generate some activity
        for i in range(10):
            generator.generate(eval_results={"test": i})
        
        stats = generator.get_generation_statistics()
        performance = generator.get_performance_report()
        
        assert stats["total_generations"] == 10
        assert "avg_generation_time" in performance
        assert "cache_hit_rate" in performance

@pytest.mark.integration
class TestDistributedProcessing:
    
    @pytest.fixture
    async def distributed_setup(self):
        """Set up distributed processing components."""
        from modelcard_generator.core.distributed_processing import (
            DistributedTaskQueue,
            AutoScaler
        )
        
        queue = DistributedTaskQueue(redis_url="redis://localhost:6379")
        scaler = AutoScaler(min_workers=1, max_workers=3)
        
        await queue.start()
        await scaler.start()
        
        yield queue, scaler
        
        await queue.stop()
        await scaler.stop()
    
    @pytest.mark.asyncio
    async def test_task_distribution(self, distributed_setup):
        """Test distributed task processing."""
        queue, scaler = distributed_setup
        
        # Submit multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await queue.submit_task(
                "test_task",
                {"data": f"test_{i}"},
                priority=1
            )
            task_ids.append(task_id)
        
        # Wait for completion
        completed_tasks = 0
        max_wait = 30  # seconds
        wait_time = 0
        
        while completed_tasks < 5 and wait_time < max_wait:
            await asyncio.sleep(1)
            wait_time += 1
            
            for task_id in task_ids:
                status = await queue.get_task_status(task_id)
                if status.state == "completed":
                    completed_tasks += 1
        
        assert completed_tasks == 5
```

## Performance Benchmarks

### Benchmarking Script

```python
"""
Performance benchmarking for ModelCard Generator
"""
import time
import asyncio
import statistics
from typing import List, Dict, Any
from modelcard_generator.core.enhanced_generator import EnhancedModelCardGenerator

async def benchmark_generation_performance():
    """Benchmark model card generation performance."""
    
    test_cases = [
        {"size": "small", "metrics_count": 5, "logs_lines": 100},
        {"size": "medium", "metrics_count": 20, "logs_lines": 1000},
        {"size": "large", "metrics_count": 50, "logs_lines": 10000},
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"Benchmarking {case['size']} model card generation...")
        
        # Create test data
        eval_data = create_test_evaluation_data(case["metrics_count"])
        training_logs = create_test_logs(case["logs_lines"])
        
        # Run multiple iterations
        times = []
        cache_hits = []
        
        generator = EnhancedModelCardGenerator(
            enable_caching=True,
            enable_monitoring=True
        )
        
        for i in range(10):  # 10 iterations
            start_time = time.time()
            
            card = await generator.generate(
                eval_results=eval_data,
                training_logs=training_logs,
                format="huggingface"
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Get cache statistics
            cache_stats = generator.cache.get_comprehensive_stats()
            cache_hits.append(cache_stats["global"]["hit_rate"])
        
        # Calculate statistics
        results[case["size"]] = {
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "final_cache_hit_rate": cache_hits[-1] if cache_hits else 0
        }
        
        await generator.cleanup()
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    
    for size, stats in results.items():
        print(f"\n{size.upper()} Model Cards:")
        print(f"  Average time: {stats['avg_time']:.2f}s")
        print(f"  Min time:     {stats['min_time']:.2f}s")
        print(f"  Max time:     {stats['max_time']:.2f}s")
        print(f"  Std dev:      {stats['std_dev']:.2f}s")
        print(f"  Cache hit:    {stats['final_cache_hit_rate']:.1%}")
    
    return results

def create_test_evaluation_data(metrics_count: int) -> Dict[str, Any]:
    """Create test evaluation data."""
    import random
    
    metrics = {}
    for i in range(metrics_count):
        metrics[f"metric_{i}"] = random.uniform(0.1, 0.99)
    
    return {
        "model_name": "benchmark_model",
        "metrics": metrics,
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    }

def create_test_logs(line_count: int) -> List[str]:
    """Create test training logs."""
    logs = []
    for i in range(line_count):
        logs.append(f"Epoch {i//100}: loss=0.{random.randint(100,999)}")
    return logs

if __name__ == "__main__":
    asyncio.run(benchmark_generation_performance())
```

These usage examples demonstrate the full capabilities of the enhanced ModelCard Generator, from basic usage to advanced production scenarios with resilience patterns, distributed processing, and comprehensive monitoring.