"""Intelligent multi-layer caching system with predictive prefetching."""

import asyncio
import hashlib
import json
import pickle
import threading
import time
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import aioredis

from ..core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CacheLevel(Enum):
    """Cache level enumeration."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    compression_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(1, total)


class AccessPattern:
    """Track and analyze cache access patterns."""

    def __init__(self, max_history: int = 1000):
        self.access_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.key_frequencies = defaultdict(int)
        self.temporal_patterns = defaultdict(list)

    def record_access(self, key: str, hit: bool, timestamp: Optional[datetime] = None) -> None:
        """Record a cache access."""
        if timestamp is None:
            timestamp = datetime.now()

        access_record = {
            "key": key,
            "hit": hit,
            "timestamp": timestamp,
            "hour": timestamp.hour,
            "weekday": timestamp.weekday()
        }

        self.access_history.append(access_record)
        self.key_frequencies[key] += 1
        self.temporal_patterns[f"{timestamp.weekday()}_{timestamp.hour}"].append(key)

        # Maintain history size
        if len(self.access_history) > self.max_history:
            old_record = self.access_history.pop(0)
            self.key_frequencies[old_record["key"]] -= 1
            if self.key_frequencies[old_record["key"]] <= 0:
                del self.key_frequencies[old_record["key"]]

    def predict_next_accesses(self, current_time: Optional[datetime] = None) -> List[str]:
        """Predict likely next cache accesses."""
        if current_time is None:
            current_time = datetime.now()

        # Time-based prediction
        pattern_key = f"{current_time.weekday()}_{current_time.hour}"
        temporal_keys = self.temporal_patterns.get(pattern_key, [])

        # Frequency-based prediction
        frequent_keys = sorted(
            self.key_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Combine predictions (remove duplicates while preserving order)
        predictions = []
        seen = set()

        # Prioritize temporal patterns
        for key in temporal_keys[-5:]:  # Recent keys from this time pattern
            if key not in seen:
                predictions.append(key)
                seen.add(key)

        # Add frequent keys
        for key, freq in frequent_keys:
            if key not in seen and len(predictions) < 15:
                predictions.append(key)
                seen.add(key)

        return predictions


class InMemoryCache:
    """High-performance in-memory LRU cache with compression."""

    def __init__(self, max_size_mb: int = 100, compression_threshold: int = 1024):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression_threshold = compression_threshold
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.entries:
                entry = self.entries[key]

                # Check expiration
                if entry.is_expired:
                    del self.entries[key]
                    self.stats.misses += 1
                    return None

                # Move to end (most recently used)
                self.entries.move_to_end(key)
                entry.accessed_at = datetime.now()
                entry.access_count += 1

                self.stats.hits += 1

                # Decompress if needed
                value = entry.value
                if entry.compression_enabled:
                    value = pickle.loads(zlib.decompress(value))

                return value
            else:
                self.stats.misses += 1
                return None

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put item in cache."""
        with self._lock:
            # Serialize and optionally compress
            serialized_value = value
            compressed = False

            try:
                pickled = pickle.dumps(value)
                size_bytes = len(pickled)

                if size_bytes > self.compression_threshold:
                    compressed_data = zlib.compress(pickled)
                    if len(compressed_data) < size_bytes:
                        serialized_value = compressed_data
                        compressed = True
                        size_bytes = len(compressed_data)
                    else:
                        serialized_value = pickled
                else:
                    serialized_value = pickled

            except Exception as e:
                logger.warning(f"Failed to serialize cache entry {key}: {e}")
                return

            # Remove existing entry if present
            if key in self.entries:
                old_entry = self.entries[key]
                self.stats.total_size_bytes -= old_entry.size_bytes
                del self.entries[key]

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=serialized_value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                compression_enabled=compressed
            )

            # Evict if necessary
            while (self.stats.total_size_bytes + size_bytes > self.max_size_bytes and
                   len(self.entries) > 0):
                self._evict_lru()

            # Add new entry
            self.entries[key] = entry
            self.stats.total_size_bytes += size_bytes
            self.stats.entry_count = len(self.entries)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.entries:
            key, entry = self.entries.popitem(last=False)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.evictions += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            self.stats = CacheStats()

    def size_info(self) -> Dict[str, Any]:
        """Get cache size information."""
        with self._lock:
            return {
                "entry_count": len(self.entries),
                "size_bytes": self.stats.total_size_bytes,
                "size_mb": self.stats.total_size_bytes / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "utilization": self.stats.total_size_bytes / max(1, self.max_size_bytes)
            }


class DiskCache:
    """Persistent disk-based cache with indexing."""

    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self.stats = CacheStats()
        self._lock = threading.Lock()

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    self.index = json.load(f)

                # Update stats
                self.stats.entry_count = len(self.index)
                self.stats.total_size_bytes = sum(
                    entry.get("size_bytes", 0) for entry in self.index.values()
                )
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        with self._lock:
            if key not in self.index:
                self.stats.misses += 1
                return None

            entry_info = self.index[key]

            # Check expiration
            if entry_info.get("ttl_seconds"):
                created_at = datetime.fromisoformat(entry_info["created_at"])
                if (datetime.now() - created_at).total_seconds() > entry_info["ttl_seconds"]:
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None

            # Load from disk
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                self._remove_entry(key)
                self.stats.misses += 1
                return None

            try:
                with open(cache_path, "rb") as f:
                    data = f.read()

                # Decompress if needed
                if entry_info.get("compressed", False):
                    data = zlib.decompress(data)

                value = pickle.loads(data)

                # Update access info
                entry_info["accessed_at"] = datetime.now().isoformat()
                entry_info["access_count"] = entry_info.get("access_count", 0) + 1

                self.stats.hits += 1
                self._save_index()

                return value

            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                self._remove_entry(key)
                self.stats.misses += 1
                return None

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put item in disk cache."""
        with self._lock:
            try:
                # Serialize
                pickled_data = pickle.dumps(value)
                compressed = False

                # Compress if beneficial
                if len(pickled_data) > 1024:
                    compressed_data = zlib.compress(pickled_data)
                    if len(compressed_data) < len(pickled_data):
                        data_to_save = compressed_data
                        compressed = True
                    else:
                        data_to_save = pickled_data
                else:
                    data_to_save = pickled_data

                # Check if we need to evict
                data_size = len(data_to_save)
                while (self.stats.total_size_bytes + data_size > self.max_size_bytes and
                       len(self.index) > 0):
                    self._evict_lru()

                # Save to disk
                cache_path = self._get_cache_path(key)
                with open(cache_path, "wb") as f:
                    f.write(data_to_save)

                # Update index
                if key in self.index:
                    self.stats.total_size_bytes -= self.index[key].get("size_bytes", 0)

                self.index[key] = {
                    "created_at": datetime.now().isoformat(),
                    "accessed_at": datetime.now().isoformat(),
                    "access_count": 1,
                    "size_bytes": data_size,
                    "ttl_seconds": ttl_seconds,
                    "compressed": compressed,
                    "cache_path": str(cache_path)
                }

                self.stats.total_size_bytes += data_size
                self.stats.entry_count = len(self.index)
                self._save_index()

            except Exception as e:
                logger.error(f"Failed to save cache entry {key}: {e}")

    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        if key in self.index:
            entry_info = self.index[key]
            cache_path = Path(entry_info["cache_path"])

            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_path}: {e}")

            self.stats.total_size_bytes -= entry_info.get("size_bytes", 0)
            del self.index[key]
            self.stats.entry_count = len(self.index)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.index:
            return

        # Find LRU entry
        lru_key = min(
            self.index.keys(),
            key=lambda k: self.index[k].get("accessed_at", "1900-01-01")
        )

        self._remove_entry(lru_key)
        self.stats.evictions += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for key in list(self.index.keys()):
                self._remove_entry(key)
            self.stats = CacheStats()
            self._save_index()


class DistributedCache:
    """Redis-based distributed cache."""

    def __init__(self, redis_url: str = "redis://localhost:6379", key_prefix: str = "mcg:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis: Optional[aioredis.Redis] = None
        self.stats = CacheStats()
        self.is_connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            self.is_connected = True
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.is_connected = False

    async def get(self, key: str) -> Optional[Any]:
        """Get item from distributed cache."""
        if not self.is_connected:
            self.stats.misses += 1
            return None

        try:
            prefixed_key = f"{self.key_prefix}{key}"
            data = await self.redis.get(prefixed_key)

            if data is None:
                self.stats.misses += 1
                return None

            # Deserialize
            value = pickle.loads(data)
            self.stats.hits += 1
            return value

        except Exception as e:
            logger.warning(f"Failed to get from distributed cache: {e}")
            self.stats.misses += 1
            return None

    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put item in distributed cache."""
        if not self.is_connected:
            return

        try:
            prefixed_key = f"{self.key_prefix}{key}"
            data = pickle.dumps(value)

            if ttl_seconds:
                await self.redis.setex(prefixed_key, ttl_seconds, data)
            else:
                await self.redis.set(prefixed_key, data)

        except Exception as e:
            logger.warning(f"Failed to put in distributed cache: {e}")

    async def clear(self) -> None:
        """Clear distributed cache entries with our prefix."""
        if not self.is_connected:
            return

        try:
            keys = await self.redis.keys(f"{self.key_prefix}*")
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Failed to clear distributed cache: {e}")


class IntelligentCache:
    """Multi-level intelligent cache with predictive prefetching."""

    def __init__(
        self,
        memory_cache_mb: int = 100,
        disk_cache_mb: int = 1000,
        redis_url: Optional[str] = None,
        enable_prefetching: bool = True
    ):
        # Initialize cache layers
        self.memory_cache = InMemoryCache(memory_cache_mb)
        self.disk_cache = DiskCache(max_size_mb=disk_cache_mb)
        self.distributed_cache = DistributedCache(redis_url) if redis_url else None

        # Access pattern analysis
        self.access_pattern = AccessPattern()
        self.enable_prefetching = enable_prefetching

        # Prefetching
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_task: Optional[asyncio.Task] = None

        # Stats
        self.global_stats = CacheStats()

    async def start(self) -> None:
        """Start cache services."""
        if self.distributed_cache:
            await self.distributed_cache.connect()

        if self.enable_prefetching:
            self.prefetch_task = asyncio.create_task(self._prefetch_loop())

        logger.info("Intelligent cache system started")

    async def stop(self) -> None:
        """Stop cache services."""
        if self.prefetch_task:
            self.prefetch_task.cancel()
            try:
                await self.prefetch_task
            except asyncio.CancelledError:
                pass

        logger.info("Intelligent cache system stopped")

    async def get(self, key: str) -> Optional[Any]:
        """Get item with intelligent multi-level lookup."""
        start_time = time.time()

        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self.access_pattern.record_access(key, hit=True)
            self._update_global_stats(hit=True, access_time_ms=(time.time() - start_time) * 1000)
            return value

        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            self.access_pattern.record_access(key, hit=True)
            self._update_global_stats(hit=True, access_time_ms=(time.time() - start_time) * 1000)
            return value

        # Try distributed cache
        if self.distributed_cache:
            value = await self.distributed_cache.get(key)
            if value is not None:
                # Promote to higher levels
                self.memory_cache.put(key, value)
                self.disk_cache.put(key, value)
                self.access_pattern.record_access(key, hit=True)
                self._update_global_stats(hit=True, access_time_ms=(time.time() - start_time) * 1000)
                return value

        # Cache miss
        self.access_pattern.record_access(key, hit=False)
        self._update_global_stats(hit=False, access_time_ms=(time.time() - start_time) * 1000)

        # Trigger predictive prefetching
        if self.enable_prefetching:
            await self._trigger_prefetch(key)

        return None

    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put item in all cache levels."""
        # Store in all levels
        self.memory_cache.put(key, value, ttl_seconds)
        self.disk_cache.put(key, value, ttl_seconds)

        if self.distributed_cache:
            await self.distributed_cache.put(key, value, ttl_seconds)

    def _update_global_stats(self, hit: bool, access_time_ms: float) -> None:
        """Update global cache statistics."""
        if hit:
            self.global_stats.hits += 1
        else:
            self.global_stats.misses += 1

        # Update average access time
        total_accesses = self.global_stats.hits + self.global_stats.misses
        self.global_stats.avg_access_time_ms = (
            (self.global_stats.avg_access_time_ms * (total_accesses - 1) + access_time_ms) /
            total_accesses
        )

    async def _trigger_prefetch(self, missed_key: str) -> None:
        """Trigger predictive prefetching based on access patterns."""
        predictions = self.access_pattern.predict_next_accesses()

        # Add predictions to prefetch queue
        for predicted_key in predictions[:5]:  # Limit prefetch candidates
            if predicted_key != missed_key:
                try:
                    self.prefetch_queue.put_nowait(predicted_key)
                except asyncio.QueueFull:
                    break

    async def _prefetch_loop(self) -> None:
        """Background prefetching loop."""
        while True:
            try:
                # Wait for prefetch candidates
                key = await asyncio.wait_for(self.prefetch_queue.get(), timeout=30.0)

                # Check if already cached
                if self.memory_cache.get(key) is not None:
                    continue

                logger.debug(f"Prefetching key: {key}")
                # In a real implementation, this would trigger data source loading
                # For now, just log the prefetch attempt

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in prefetch loop: {e}")
                await asyncio.sleep(1)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "global": {
                "hit_rate": self.global_stats.hit_rate,
                "avg_access_time_ms": self.global_stats.avg_access_time_ms,
                "total_accesses": self.global_stats.hits + self.global_stats.misses
            },
            "memory_cache": {
                "stats": self.memory_cache.stats.__dict__,
                "size_info": self.memory_cache.size_info()
            },
            "disk_cache": {
                "stats": self.disk_cache.stats.__dict__,
                "entry_count": len(self.disk_cache.index),
                "size_mb": self.disk_cache.stats.total_size_bytes / 1024 / 1024
            },
            "distributed_cache": {
                "connected": self.distributed_cache.is_connected if self.distributed_cache else False,
                "stats": self.distributed_cache.stats.__dict__ if self.distributed_cache else {}
            },
            "access_patterns": {
                "total_patterns": len(self.access_pattern.access_history),
                "unique_keys": len(self.access_pattern.key_frequencies),
                "temporal_patterns": len(self.access_pattern.temporal_patterns)
            }
        }


def cache_with_intelligence(
    cache_instance: IntelligentCache,
    ttl_seconds: Optional[int] = None,
    cache_key_func: Optional[Callable[..., str]] = None
) -> Callable:
    """Decorator for intelligent caching of function results."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try cache first
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await cache_instance.put(cache_key, result, ttl_seconds)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # For sync functions, we need to handle cache differently
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try memory and disk cache (sync only)
            cached_result = cache_instance.memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            cached_result = cache_instance.disk_cache.get(cache_key)
            if cached_result is not None:
                cache_instance.memory_cache.put(cache_key, cached_result, ttl_seconds)
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.memory_cache.put(cache_key, result, ttl_seconds)
            cache_instance.disk_cache.put(cache_key, result, ttl_seconds)

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global intelligent cache instance
global_cache = IntelligentCache()


async def initialize_intelligent_cache(**kwargs) -> None:
    """Initialize global intelligent cache."""
    global global_cache
    global_cache = IntelligentCache(**kwargs)
    await global_cache.start()


async def shutdown_intelligent_cache() -> None:
    """Shutdown global intelligent cache."""
    await global_cache.stop()
