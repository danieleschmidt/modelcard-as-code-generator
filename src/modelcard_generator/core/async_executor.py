"""Async execution engine with advanced concurrency patterns."""

import asyncio
import time
from asyncio import Queue, Semaphore
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Generic, Union, Tuple
from enum import Enum
from functools import wraps

from .logging_config import get_logger
from .exceptions import ResourceError
from .performance_monitor import performance_tracker

logger = get_logger(__name__)

T = TypeVar('T')
P = TypeVar('P')


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskInfo(Generic[T]):
    """Information about an async task."""
    id: str
    coroutine: Coroutine[Any, Any, T]
    priority: TaskPriority
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def wait_time(self) -> Optional[float]:
        """Get time waited in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return None


class TaskResult(Generic[T]):
    """Result of task execution."""
    
    def __init__(self, task_id: str, success: bool, result: Optional[T] = None, error: Optional[Exception] = None):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.completed_at = time.time()


class WorkerPool:
    """Pool of async workers for task execution."""
    
    def __init__(self, max_workers: int = 10, worker_timeout: float = 300.0):
        self.max_workers = max_workers
        self.worker_timeout = worker_timeout
        self.task_queue: Queue[TaskInfo] = Queue()
        self.workers: List[asyncio.Task] = []
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.worker_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_runtime": 0.0,
            "avg_task_time": 0.0
        })
        self._shutdown = False
        self._semaphore = Semaphore(max_workers)
    
    async def start(self) -> None:
        """Start the worker pool."""
        logger.info(f"Starting worker pool with {self.max_workers} workers")
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
    
    async def stop(self) -> None:
        """Stop the worker pool gracefully."""
        logger.info("Stopping worker pool")
        self._shutdown = True
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Cancel any remaining active tasks
        for task_info in self.active_tasks.values():
            if hasattr(task_info.coroutine, 'cancel'):
                try:
                    task_info.coroutine.cancel()
                except Exception:
                    pass
    
    async def submit_task(self, task_info: TaskInfo) -> None:
        """Submit a task to the worker pool."""
        if self._shutdown:
            raise ResourceError("worker_pool", "Worker pool is shutting down")
        
        await self.task_queue.put(task_info)
        logger.debug(f"Task {task_info.id} submitted to queue")
    
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks."""
        logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown:
            try:
                # Get task from queue with timeout
                task_info = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                await self._process_task(worker_id, task_info)
                
            except asyncio.TimeoutError:
                # No task available, continue loop
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _process_task(self, worker_id: int, task_info: TaskInfo) -> None:
        """Process a single task."""
        task_start = time.time()
        task_info.started_at = task_start
        
        # Add to active tasks
        self.active_tasks[task_info.id] = task_info
        
        try:
            # Execute task with timeout
            if task_info.timeout:
                result = await asyncio.wait_for(task_info.coroutine, timeout=task_info.timeout)
            else:
                result = await task_info.coroutine
            
            # Task completed successfully
            task_info.completed_at = time.time()
            task_result = TaskResult(task_info.id, True, result)
            
            # Update stats
            duration = task_info.duration or 0
            stats = self.worker_stats[worker_id]
            stats["tasks_completed"] += 1
            stats["total_runtime"] += duration
            stats["avg_task_time"] = stats["total_runtime"] / stats["tasks_completed"]
            
            logger.debug(f"Task {task_info.id} completed by worker {worker_id} in {duration:.2f}s")
            
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_info.id} timed out after {task_info.timeout}s")
            task_result = TaskResult(task_info.id, False, error=ResourceError("timeout", "Task execution timeout"))
            self.worker_stats[worker_id]["tasks_failed"] += 1
            
        except Exception as e:
            logger.error(f"Task {task_info.id} failed: {e}")
            task_result = TaskResult(task_info.id, False, error=e)
            self.worker_stats[worker_id]["tasks_failed"] += 1
        
        finally:
            # Remove from active tasks and add to completed
            self.active_tasks.pop(task_info.id, None)
            self.completed_tasks.append(task_result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        total_completed = sum(stats["tasks_completed"] for stats in self.worker_stats.values())
        total_failed = sum(stats["tasks_failed"] for stats in self.worker_stats.values())
        
        return {
            "max_workers": self.max_workers,
            "active_workers": len([w for w in self.workers if not w.done()]),
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "total_completed": total_completed,
            "total_failed": total_failed,
            "success_rate": total_completed / max(1, total_completed + total_failed),
            "worker_stats": dict(self.worker_stats)
        }


class PriorityAsyncExecutor:
    """High-level async executor with priority queues and advanced features."""
    
    def __init__(self, max_concurrent: int = 10, enable_metrics: bool = True):
        self.max_concurrent = max_concurrent
        self.enable_metrics = enable_metrics
        
        # Priority queues
        self.priority_queues: Dict[TaskPriority, Queue] = {
            priority: Queue() for priority in TaskPriority
        }
        
        self.worker_pool = WorkerPool(max_concurrent)
        self._task_counter = 0
        self._results: Dict[str, TaskResult] = {}
        self._pending_tasks: Dict[str, asyncio.Future] = {}
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the executor."""
        if self._running:
            return
        
        self._running = True
        await self.worker_pool.start()
        
        # Start priority dispatcher
        self._dispatcher_task = asyncio.create_task(self._priority_dispatcher())
        
        logger.info("Priority async executor started")
    
    async def stop(self) -> None:
        """Stop the executor gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop dispatcher
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        
        # Stop worker pool
        await self.worker_pool.stop()
        
        # Cancel pending tasks
        for future in self._pending_tasks.values():
            if not future.done():
                future.cancel()
        
        logger.info("Priority async executor stopped")
    
    async def submit(
        self, 
        coro: Coroutine[Any, Any, T], 
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a coroutine for execution."""
        if not self._running:
            await self.start()
        
        self._task_counter += 1
        task_id = f"task_{self._task_counter}_{int(time.time() * 1000)}"
        
        task_info = TaskInfo(
            id=task_id,
            coroutine=coro,
            priority=priority,
            created_at=time.time(),
            timeout=timeout,
            metadata=metadata or {}
        )
        
        # Add to priority queue
        await self.priority_queues[priority].put(task_info)
        
        # Create future for result
        future: asyncio.Future[T] = asyncio.Future()
        self._pending_tasks[task_id] = future
        
        logger.debug(f"Task {task_id} submitted with priority {priority.name}")
        return task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> T:
        """Get result of a submitted task."""
        if task_id not in self._pending_tasks:
            if task_id in self._results:
                result = self._results[task_id]
                if result.success:
                    return result.result
                else:
                    raise result.error
            else:
                raise ValueError(f"Unknown task ID: {task_id}")
        
        future = self._pending_tasks[task_id]
        
        try:
            if timeout:
                return await asyncio.wait_for(future, timeout=timeout)
            else:
                return await future
        except asyncio.TimeoutError:
            raise ResourceError("timeout", f"Task {task_id} result timeout")
    
    async def wait_for_completion(
        self, 
        task_ids: List[str], 
        timeout: Optional[float] = None, 
        return_when: str = 'ALL_COMPLETED'
    ) -> Tuple[List[T], List[Exception]]:
        """Wait for multiple tasks to complete."""
        futures = [self._pending_tasks[tid] for tid in task_ids if tid in self._pending_tasks]
        
        if not futures:
            return [], []
        
        try:
            if return_when == 'ALL_COMPLETED':
                results = await asyncio.gather(*futures, return_exceptions=True)
            else:  # FIRST_COMPLETED
                done, pending = await asyncio.wait(futures, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                results = [f.result() for f in done]
                # Cancel pending
                for f in pending:
                    f.cancel()
            
            # Separate successful results from exceptions
            successful = [r for r in results if not isinstance(r, Exception)]
            failed = [r for r in results if isinstance(r, Exception)]
            
            return successful, failed
            
        except asyncio.TimeoutError:
            raise ResourceError("timeout", "Wait for completion timeout")
    
    async def _priority_dispatcher(self) -> None:
        """Dispatcher that feeds tasks to workers based on priority."""
        while self._running:
            task_info = None
            
            # Check priority queues in order (highest first)
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                queue = self.priority_queues[priority]
                try:
                    task_info = await asyncio.wait_for(queue.get(), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    continue
            
            if task_info:
                # Submit to worker pool
                await self.worker_pool.submit_task(task_info)
                
                # Set up result handling
                asyncio.create_task(self._handle_task_result(task_info.id))
    
    async def _handle_task_result(self, task_id: str) -> None:
        """Handle task result and notify waiting futures."""
        # Wait for task to complete
        while task_id not in [r.task_id for r in self.worker_pool.completed_tasks]:
            await asyncio.sleep(0.1)
        
        # Find result
        result = next((r for r in self.worker_pool.completed_tasks if r.task_id == task_id), None)
        
        if result and task_id in self._pending_tasks:
            future = self._pending_tasks.pop(task_id)
            
            if result.success:
                future.set_result(result.result)
            else:
                future.set_exception(result.error or Exception("Unknown task error"))
            
            # Store result for later retrieval
            self._results[task_id] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive executor statistics."""
        queue_stats = {
            priority.name: queue.qsize() 
            for priority, queue in self.priority_queues.items()
        }
        
        return {
            "running": self._running,
            "max_concurrent": self.max_concurrent,
            "total_submitted": self._task_counter,
            "pending_tasks": len(self._pending_tasks),
            "queue_sizes": queue_stats,
            "worker_pool": self.worker_pool.get_stats()
        }


class BatchProcessor:
    """Process batches of tasks with different strategies."""
    
    def __init__(self, executor: PriorityAsyncExecutor):
        self.executor = executor
    
    async def process_batch_sequential(self, coroutines: List[Coroutine], **kwargs) -> List[Any]:
        """Process batch sequentially."""
        results = []
        for coro in coroutines:
            task_id = await self.executor.submit(coro, **kwargs)
            result = await self.executor.get_result(task_id)
            results.append(result)
        return results
    
    async def process_batch_parallel(self, coroutines: List[Coroutine], **kwargs) -> List[Any]:
        """Process batch in parallel."""
        task_ids = []
        for coro in coroutines:
            task_id = await self.executor.submit(coro, **kwargs)
            task_ids.append(task_id)
        
        results, errors = await self.executor.wait_for_completion(task_ids)
        
        # Re-raise first error if any
        if errors:
            raise errors[0]
        
        return results
    
    async def process_batch_chunked(
        self, 
        coroutines: List[Coroutine], 
        chunk_size: int = 10, 
        delay_between_chunks: float = 0.1,
        **kwargs
    ) -> List[Any]:
        """Process batch in chunks."""
        results = []
        
        for i in range(0, len(coroutines), chunk_size):
            chunk = coroutines[i:i + chunk_size]
            chunk_results = await self.process_batch_parallel(chunk, **kwargs)
            results.extend(chunk_results)
            
            # Delay between chunks
            if i + chunk_size < len(coroutines) and delay_between_chunks > 0:
                await asyncio.sleep(delay_between_chunks)
        
        return results


# Context managers and decorators
@asynccontextmanager
async def managed_executor(max_concurrent: int = 10):
    """Context manager for automatic executor lifecycle management."""
    executor = PriorityAsyncExecutor(max_concurrent)
    try:
        await executor.start()
        yield executor
    finally:
        await executor.stop()


def async_task(priority: TaskPriority = TaskPriority.NORMAL, timeout: Optional[float] = None):
    """Decorator to mark async functions as tasks."""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # This would need integration with a global executor instance
            return await func(*args, **kwargs)
        
        wrapper._task_priority = priority
        wrapper._task_timeout = timeout
        return wrapper
    
    return decorator


# Global executor instance
global_executor: Optional[PriorityAsyncExecutor] = None


async def get_global_executor() -> PriorityAsyncExecutor:
    """Get or create global executor instance."""
    global global_executor
    
    if global_executor is None:
        global_executor = PriorityAsyncExecutor(max_concurrent=10)
        await global_executor.start()
    
    return global_executor


async def shutdown_global_executor() -> None:
    """Shutdown global executor."""
    global global_executor
    
    if global_executor:
        await global_executor.stop()
        global_executor = None