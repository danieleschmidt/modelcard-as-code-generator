"""Distributed processing and scaling capabilities for model card generation."""

import asyncio
import json
import multiprocessing as mp
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aioredis

from ..core.exceptions import ResourceError
from ..core.logging_config import get_logger
from ..monitoring.enhanced_metrics import performance_tracker

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get task execution duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


@dataclass
class WorkerStats:
    """Statistics for a distributed worker."""
    worker_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration: float = 0.0
    last_activity: Optional[datetime] = None
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate worker success rate."""
        return self.completed_tasks / max(1, self.total_tasks)


class DistributedTaskQueue:
    """Redis-backed distributed task queue."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.queue_name = "modelcard_tasks"
        self.result_prefix = "result:"
        self.worker_prefix = "worker:"
        self.is_connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            self.is_connected = True
            logger.info("Connected to Redis for distributed processing")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            raise ResourceError("redis_connection", f"Cannot connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.is_connected = False

    async def enqueue_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        priority: int = 0,
        delay_seconds: int = 0
    ) -> str:
        """Enqueue a task for distributed processing."""
        if not self.is_connected:
            await self.connect()

        task_id = str(uuid.uuid4())
        task_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "task_data": task_data,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "delay_until": (datetime.now() + timedelta(seconds=delay_seconds)).isoformat() if delay_seconds > 0 else None
        }

        # Store task result placeholder
        await self.redis.setex(
            f"{self.result_prefix}{task_id}",
            3600,  # 1 hour TTL
            json.dumps(asdict(TaskResult(task_id=task_id, status=TaskStatus.PENDING)))
        )

        # Add to priority queue
        if delay_seconds > 0:
            # Delayed task
            await self.redis.zadd(
                f"{self.queue_name}:delayed",
                {json.dumps(task_payload): time.time() + delay_seconds}
            )
        else:
            # Immediate task
            await self.redis.zadd(
                self.queue_name,
                {json.dumps(task_payload): -priority}  # Negative for high priority first
            )

        logger.info(f"Enqueued task {task_id} of type {task_type}")
        return task_id

    async def dequeue_task(self, worker_id: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Dequeue a task for processing."""
        if not self.is_connected:
            await self.connect()

        # First, check for delayed tasks that are ready
        await self._process_delayed_tasks()

        # Get next task from priority queue
        result = await self.redis.bzpopmin(self.queue_name, timeout=timeout)
        if not result:
            return None

        queue_name, task_json, score = result
        task_data = json.loads(task_json.decode())

        # Update task status to running
        task_result = TaskResult(
            task_id=task_data["task_id"],
            status=TaskStatus.RUNNING,
            started_at=datetime.now(),
            worker_id=worker_id
        )

        await self.redis.setex(
            f"{self.result_prefix}{task_data['task_id']}",
            3600,
            json.dumps(asdict(task_result))
        )

        return task_data

    async def _process_delayed_tasks(self) -> None:
        """Move ready delayed tasks to main queue."""
        current_time = time.time()

        # Get tasks that are ready to run
        ready_tasks = await self.redis.zrangebyscore(
            f"{self.queue_name}:delayed",
            0,
            current_time,
            withscores=True
        )

        if ready_tasks:
            # Move to main queue
            pipe = self.redis.pipeline()
            for task_json, score in ready_tasks:
                task_data = json.loads(task_json.decode())
                pipe.zadd(self.queue_name, {task_json: -task_data["priority"]})
                pipe.zrem(f"{self.queue_name}:delayed", task_json)

            await pipe.execute()

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
        error: Optional[str] = None
    ) -> None:
        """Mark task as completed."""
        status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED

        task_result = TaskResult(
            task_id=task_id,
            status=status,
            result=result,
            error=error,
            completed_at=datetime.now()
        )

        # Get existing result to preserve metadata
        existing = await self.redis.get(f"{self.result_prefix}{task_id}")
        if existing:
            existing_result = json.loads(existing)
            task_result.started_at = datetime.fromisoformat(existing_result.get("started_at")) if existing_result.get("started_at") else None
            task_result.worker_id = existing_result.get("worker_id")

        await self.redis.setex(
            f"{self.result_prefix}{task_id}",
            3600,
            json.dumps(asdict(task_result))
        )

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID."""
        result_json = await self.redis.get(f"{self.result_prefix}{task_id}")
        if not result_json:
            return None

        result_data = json.loads(result_json)

        # Convert datetime strings back to datetime objects
        if result_data.get("started_at"):
            result_data["started_at"] = datetime.fromisoformat(result_data["started_at"])
        if result_data.get("completed_at"):
            result_data["completed_at"] = datetime.fromisoformat(result_data["completed_at"])

        return TaskResult(**result_data)

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending_tasks = await self.redis.zcard(self.queue_name)
        delayed_tasks = await self.redis.zcard(f"{self.queue_name}:delayed")

        # Get worker count
        workers = await self.redis.keys(f"{self.worker_prefix}*")
        active_workers = len(workers) if workers else 0

        return {
            "pending_tasks": pending_tasks,
            "delayed_tasks": delayed_tasks,
            "active_workers": active_workers,
            "queue_name": self.queue_name
        }


class DistributedWorker:
    """Distributed worker for processing model card tasks."""

    def __init__(
        self,
        worker_id: str,
        queue: DistributedTaskQueue,
        max_concurrent_tasks: int = 4
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_handlers: Dict[str, Callable] = {}
        self.running = False
        self.stats = WorkerStats(worker_id=worker_id)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a task handler."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def start(self) -> None:
        """Start the worker."""
        self.running = True

        # Register worker in Redis
        await self._register_worker()

        logger.info(f"Starting distributed worker {self.worker_id}")

        # Start worker loops
        tasks = []
        for i in range(self.max_concurrent_tasks):
            task = asyncio.create_task(self._worker_loop(f"{self.worker_id}-{i}"))
            tasks.append(task)

        # Start stats reporting
        stats_task = asyncio.create_task(self._stats_loop())
        tasks.append(stats_task)

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            await self._unregister_worker()

    async def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info(f"Stopping distributed worker {self.worker_id}")

    async def _worker_loop(self, loop_id: str) -> None:
        """Main worker processing loop."""
        while self.running:
            try:
                async with self.semaphore:
                    # Get next task
                    task_data = await self.queue.dequeue_task(self.worker_id, timeout=5)
                    if not task_data:
                        continue

                    await self._process_task(task_data)

            except Exception as e:
                logger.error(f"Error in worker loop {loop_id}: {e}")
                await asyncio.sleep(1)

    async def _process_task(self, task_data: Dict[str, Any]) -> None:
        """Process a single task."""
        task_id = task_data["task_id"]
        task_type = task_data["task_type"]

        start_time = time.time()
        self.stats.total_tasks += 1
        self.stats.last_activity = datetime.now()

        try:
            # Get handler
            handler = self.task_handlers.get(task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task_type}")

            # Execute task
            async with performance_tracker.track_operation(f"task_{task_type}"):
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(task_data["task_data"])
                else:
                    result = handler(task_data["task_data"])

            # Complete task
            await self.queue.complete_task(task_id, result=result)

            self.stats.completed_tasks += 1
            duration = time.time() - start_time
            self.stats.avg_task_duration = (
                (self.stats.avg_task_duration * (self.stats.completed_tasks - 1) + duration) /
                self.stats.completed_tasks
            )

            logger.info(f"Completed task {task_id} in {duration:.2f}s")

        except Exception as e:
            # Task failed
            await self.queue.complete_task(task_id, error=str(e))
            self.stats.failed_tasks += 1
            logger.error(f"Task {task_id} failed: {e}")

    async def _register_worker(self) -> None:
        """Register worker in Redis."""
        await self.queue.redis.setex(
            f"{self.queue.worker_prefix}{self.worker_id}",
            30,  # 30 second TTL
            json.dumps(asdict(self.stats))
        )

    async def _unregister_worker(self) -> None:
        """Unregister worker from Redis."""
        await self.queue.redis.delete(f"{self.queue.worker_prefix}{self.worker_id}")

    async def _stats_loop(self) -> None:
        """Periodically update worker stats in Redis."""
        while self.running:
            try:
                await self._register_worker()  # Refresh TTL and stats
                await asyncio.sleep(15)  # Update every 15 seconds
            except Exception as e:
                logger.error(f"Error updating worker stats: {e}")
                await asyncio.sleep(5)


class AutoScaler:
    """Automatic scaling for distributed processing."""

    def __init__(self, queue: DistributedTaskQueue):
        self.queue = queue
        self.min_workers = 1
        self.max_workers = 10
        self.target_queue_length = 10
        self.scale_up_threshold = 20
        self.scale_down_threshold = 5
        self.workers: List[DistributedWorker] = []
        self.worker_processes: List[mp.Process] = []

    async def start_autoscaling(self) -> None:
        """Start automatic scaling."""
        logger.info("Starting autoscaler")

        while True:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Autoscaling error: {e}")
                await asyncio.sleep(10)

    async def _evaluate_scaling(self) -> None:
        """Evaluate if scaling is needed."""
        stats = await self.queue.get_queue_stats()
        pending_tasks = stats["pending_tasks"]
        active_workers = stats["active_workers"]

        logger.debug(f"Scaling evaluation: {pending_tasks} pending tasks, {active_workers} workers")

        # Scale up decision
        if pending_tasks > self.scale_up_threshold and active_workers < self.max_workers:
            target_workers = min(
                self.max_workers,
                active_workers + max(1, pending_tasks // self.target_queue_length)
            )
            await self._scale_up(target_workers - active_workers)

        # Scale down decision
        elif pending_tasks < self.scale_down_threshold and active_workers > self.min_workers:
            target_workers = max(
                self.min_workers,
                active_workers - 1
            )
            if target_workers < active_workers:
                await self._scale_down(active_workers - target_workers)

    async def _scale_up(self, count: int) -> None:
        """Scale up workers."""
        logger.info(f"Scaling up by {count} workers")

        for i in range(count):
            worker_id = f"autoscale-worker-{int(time.time())}-{i}"
            # In a real implementation, you would start worker processes
            # For now, just log the scaling action
            logger.info(f"Would start worker: {worker_id}")

    async def _scale_down(self, count: int) -> None:
        """Scale down workers."""
        logger.info(f"Scaling down by {count} workers")
        # Implementation would gracefully terminate worker processes


class LoadBalancer:
    """Intelligent load balancing for distributed tasks."""

    def __init__(self, queue: DistributedTaskQueue):
        self.queue = queue
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.routing_strategies = {
            "round_robin": self._round_robin_routing,
            "least_loaded": self._least_loaded_routing,
            "performance_based": self._performance_based_routing
        }
        self.current_strategy = "performance_based"
        self.round_robin_counter = 0

    async def route_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        routing_strategy: Optional[str] = None
    ) -> str:
        """Route task to optimal worker."""
        strategy = routing_strategy or self.current_strategy

        # Update worker stats
        await self._update_worker_stats()

        # Apply routing strategy
        router = self.routing_strategies.get(strategy, self._round_robin_routing)
        selected_queue = await router(task_type, task_data)

        # Enqueue task to selected queue/worker
        return await self.queue.enqueue_task(task_type, task_data)

    async def _update_worker_stats(self) -> None:
        """Update worker statistics from Redis."""
        worker_keys = await self.queue.redis.keys(f"{self.queue.worker_prefix}*")

        self.worker_stats = {}
        for key in worker_keys:
            worker_data = await self.queue.redis.get(key)
            if worker_data:
                stats_data = json.loads(worker_data)
                worker_id = key.decode().split(":")[-1]
                self.worker_stats[worker_id] = WorkerStats(**stats_data)

    async def _round_robin_routing(self, task_type: str, task_data: Dict[str, Any]) -> str:
        """Simple round-robin routing."""
        worker_ids = list(self.worker_stats.keys())
        if not worker_ids:
            return "default"

        selected = worker_ids[self.round_robin_counter % len(worker_ids)]
        self.round_robin_counter += 1
        return selected

    async def _least_loaded_routing(self, task_type: str, task_data: Dict[str, Any]) -> str:
        """Route to least loaded worker."""
        if not self.worker_stats:
            return "default"

        # Find worker with lowest active task count (approximation)
        best_worker = min(
            self.worker_stats.items(),
            key=lambda x: x[1].total_tasks - x[1].completed_tasks - x[1].failed_tasks
        )
        return best_worker[0]

    async def _performance_based_routing(self, task_type: str, task_data: Dict[str, Any]) -> str:
        """Route based on worker performance."""
        if not self.worker_stats:
            return "default"

        # Score workers based on success rate and speed
        best_worker = max(
            self.worker_stats.items(),
            key=lambda x: x[1].success_rate / max(0.1, x[1].avg_task_duration)
        )
        return best_worker[0]


# Global instances for easy access
default_queue = DistributedTaskQueue()
default_load_balancer = LoadBalancer(default_queue)


async def initialize_distributed_processing(redis_url: str = "redis://localhost:6379") -> None:
    """Initialize distributed processing components."""
    global default_queue, default_load_balancer

    default_queue = DistributedTaskQueue(redis_url)
    await default_queue.connect()

    default_load_balancer = LoadBalancer(default_queue)

    logger.info("Distributed processing initialized")


async def shutdown_distributed_processing() -> None:
    """Shutdown distributed processing components."""
    await default_queue.disconnect()
    logger.info("Distributed processing shutdown")


def get_distributed_stats() -> Dict[str, Any]:
    """Get comprehensive distributed processing statistics."""
    return {
        "queue_connected": default_queue.is_connected,
        "redis_url": default_queue.redis_url,
        "queue_name": default_queue.queue_name
    }
