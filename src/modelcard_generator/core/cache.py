"""Intelligent caching system for model card generator."""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable, TypeVar, Union
from datetime import datetime, timedelta
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from .exceptions import ResourceError
from .logging_config import get_logger
from .config import get_config

logger = get_logger(__name__)
T = TypeVar('T')


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    tags: list[str]
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class IntelligentCache:
    """High-performance intelligent cache with adaptive eviction."""
    
    def __init__(
        self,
        max_size_mb: int = 500,
        ttl_seconds: int = 3600,
        cache_dir: Optional[str] = None,
        persistent: bool = True,
        compression: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.persistent = persistent
        self.compression = compression
        
        # Thread-safe storage
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = RLock()
        self._stats_lock = Lock()
        
        # Cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".modelcard" / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0,
            "entries": 0
        }
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache")
        
        # Load persistent cache
        if self.persistent:
            self._load_persistent_cache()
        
        logger.info(f"Intelligent cache initialized: {max_size_mb}MB, TTL: {ttl_seconds}s")
    
    def get(self, key: str, default: T = None) -> Optional[T]:
        """Get value from cache with intelligent prefetching."""
        start_time = time.time()
        
        try:
            with self._cache_lock:
                # Check memory cache
                if key in self._memory_cache:
                    entry = self._memory_cache[key]
                    
                    # Check expiration
                    if entry.is_expired:
                        del self._memory_cache[key]
                        self._update_stats("miss")
                        logger.debug(f"Cache expired for key: {key}")
                        return default
                    
                    # Update access info
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    self._update_stats("hit")
                    
                    duration_ms = (time.time() - start_time) * 1000
                    logger.debug(f"Cache hit for {key}: {duration_ms:.2f}ms")
                    
                    return entry.value
                
                # Check persistent cache
                if self.persistent:
                    persistent_value = self._load_from_disk(key)
                    if persistent_value is not None:
                        # Add back to memory cache
                        self.put(key, persistent_value, ttl_seconds=self.default_ttl)
                        self._update_stats("hit")
                        
                        duration_ms = (time.time() - start_time) * 1000
                        logger.debug(f"Cache hit from disk for {key}: {duration_ms:.2f}ms")
                        
                        return persistent_value
                
                self._update_stats("miss")
                
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Cache miss for {key}: {duration_ms:.2f}ms")
                
                return default
                
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            return default
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[list[str]] = None
    ) -> None:
        """Put value in cache with intelligent eviction."""
        try:
            ttl = ttl_seconds or self.default_ttl
            tags = tags or []
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            with self._cache_lock:
                # Check if we need to evict
                self._ensure_capacity(size_bytes)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    size_bytes=size_bytes,
                    tags=tags,
                    ttl_seconds=ttl
                )
                # Store in memory
                old_entry = self._memory_cache.get(key)
                self._memory_cache[key] = entry
                
                # Update stats
                if old_entry:\n                    self._stats[\"size_bytes\"] -= old_entry.size_bytes\n                else:\n                    self._stats[\"entries\"] += 1\n                \n                self._stats[\"size_bytes\"] += size_bytes\n                \n                # Store persistently\n                if self.persistent:\n                    self._executor.submit(self._save_to_disk, key, value, entry)\n                \n                logger.debug(f\"Cached {key}: {size_bytes} bytes, TTL: {ttl}s\")\n                \n        except Exception as e:\n            logger.error(f\"Cache put error for {key}: {e}\")\n    \n    def delete(self, key: str) -> bool:\n        \"\"\"Delete key from cache.\"\"\"\n        try:\n            with self._cache_lock:\n                if key in self._memory_cache:\n                    entry = self._memory_cache[key]\n                    del self._memory_cache[key]\n                    \n                    self._stats[\"size_bytes\"] -= entry.size_bytes\n                    self._stats[\"entries\"] -= 1\n                    \n                    # Delete from disk\n                    if self.persistent:\n                        self._executor.submit(self._delete_from_disk, key)\n                    \n                    logger.debug(f\"Deleted cache entry: {key}\")\n                    return True\n            \n            return False\n            \n        except Exception as e:\n            logger.error(f\"Cache delete error for {key}: {e}\")\n            return False\n    \n    def clear(self, tags: Optional[list[str]] = None) -> int:\n        \"\"\"Clear cache entries, optionally by tags.\"\"\"\n        try:\n            cleared = 0\n            \n            with self._cache_lock:\n                if tags:\n                    # Clear by tags\n                    to_delete = []\n                    for key, entry in self._memory_cache.items():\n                        if any(tag in entry.tags for tag in tags):\n                            to_delete.append(key)\n                    \n                    for key in to_delete:\n                        if self.delete(key):\n                            cleared += 1\n                else:\n                    # Clear all\n                    cleared = len(self._memory_cache)\n                    self._memory_cache.clear()\n                    self._stats[\"size_bytes\"] = 0\n                    self._stats[\"entries\"] = 0\n                    \n                    # Clear disk cache\n                    if self.persistent:\n                        self._executor.submit(self._clear_disk_cache)\n            \n            logger.info(f\"Cleared {cleared} cache entries\")\n            return cleared\n            \n        except Exception as e:\n            logger.error(f\"Cache clear error: {e}\")\n            return 0\n    \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get cache statistics.\"\"\"\n        with self._stats_lock:\n            stats = self._stats.copy()\n        \n        with self._cache_lock:\n            stats[\"hit_rate\"] = stats[\"hits\"] / (stats[\"hits\"] + stats[\"misses\"]) if (stats[\"hits\"] + stats[\"misses\"]) > 0 else 0\n            stats[\"size_mb\"] = stats[\"size_bytes\"] / (1024 * 1024)\n            stats[\"utilization\"] = stats[\"size_bytes\"] / self.max_size_bytes\n            stats[\"active_entries\"] = len(self._memory_cache)\n        \n        return stats\n    \n    def _ensure_capacity(self, required_bytes: int) -> None:\n        \"\"\"Ensure cache has capacity for new entry.\"\"\"\n        if self._stats[\"size_bytes\"] + required_bytes <= self.max_size_bytes:\n            return\n        \n        # Calculate how much to evict\n        bytes_to_evict = (self._stats[\"size_bytes\"] + required_bytes) - self.max_size_bytes\n        bytes_to_evict = max(bytes_to_evict, self.max_size_bytes * 0.1)  # Evict at least 10%\n        \n        evicted = self._evict_entries(bytes_to_evict)\n        logger.debug(f\"Evicted {evicted} entries to free {bytes_to_evict} bytes\")\n    \n    def _evict_entries(self, target_bytes: int) -> int:\n        \"\"\"Evict entries using adaptive algorithm.\"\"\"\n        evicted_count = 0\n        evicted_bytes = 0\n        \n        # Sort entries by eviction priority (LFU + LRU + size)\n        entries = list(self._memory_cache.items())\n        entries.sort(key=lambda x: self._eviction_score(x[1]))\n        \n        for key, entry in entries:\n            if evicted_bytes >= target_bytes:\n                break\n            \n            del self._memory_cache[key]\n            evicted_bytes += entry.size_bytes\n            evicted_count += 1\n            \n            # Delete from disk\n            if self.persistent:\n                self._executor.submit(self._delete_from_disk, key)\n        \n        self._stats[\"size_bytes\"] -= evicted_bytes\n        self._stats[\"entries\"] -= evicted_count\n        self._stats[\"evictions\"] += evicted_count\n        \n        return evicted_count\n    \n    def _eviction_score(self, entry: CacheEntry) -> float:\n        \"\"\"Calculate eviction score (lower = evict first).\"\"\"\n        # Factors: frequency, recency, size, age\n        frequency_score = 1.0 / max(entry.access_count, 1)\n        recency_score = (datetime.now() - entry.last_accessed).total_seconds() / 3600  # hours\n        size_score = entry.size_bytes / (1024 * 1024)  # MB\n        age_score = entry.age_seconds / 3600  # hours\n        \n        # Weighted combination\n        return frequency_score * 0.3 + recency_score * 0.3 + size_score * 0.2 + age_score * 0.2\n    \n    def _calculate_size(self, value: Any) -> int:\n        \"\"\"Calculate size of value in bytes.\"\"\"\n        try:\n            if hasattr(value, '__sizeof__'):\n                return value.__sizeof__()\n            else:\n                # Serialize to estimate size\n                serialized = pickle.dumps(value)\n                return len(serialized)\n        except Exception:\n            # Fallback estimate\n            return 1024\n    \n    def _update_stats(self, event: str) -> None:\n        \"\"\"Update cache statistics.\"\"\"\n        with self._stats_lock:\n            if event in self._stats:\n                self._stats[event] += 1\n    \n    def _load_persistent_cache(self) -> None:\n        \"\"\"Load cache from disk.\"\"\"\n        try:\n            index_file = self.cache_dir / \"index.json\"\n            if not index_file.exists():\n                return\n            \n            with open(index_file, 'r') as f:\n                index = json.load(f)\n            \n            loaded = 0\n            for key, entry_data in index.items():\n                try:\n                    entry = CacheEntry.from_dict(entry_data)\n                    if not entry.is_expired:\n                        value = self._load_from_disk(key)\n                        if value is not None:\n                            self._memory_cache[key] = entry\n                            self._stats[\"size_bytes\"] += entry.size_bytes\n                            self._stats[\"entries\"] += 1\n                            loaded += 1\n                except Exception as e:\n                    logger.warning(f\"Failed to load cache entry {key}: {e}\")\n            \n            if loaded > 0:\n                logger.info(f\"Loaded {loaded} entries from persistent cache\")\n                \n        except Exception as e:\n            logger.warning(f\"Failed to load persistent cache: {e}\")\n    \n    def _save_to_disk(self, key: str, value: Any, entry: CacheEntry) -> None:\n        \"\"\"Save entry to disk.\"\"\"\n        try:\n            # Save value\n            cache_file = self.cache_dir / f\"{self._hash_key(key)}.cache\"\n            with open(cache_file, 'wb') as f:\n                if self.compression:\n                    import gzip\n                    with gzip.open(f, 'wb') as gz:\n                        pickle.dump(value, gz)\n                else:\n                    pickle.dump(value, f)\n            \n            # Update index\n            self._update_index(key, entry)\n            \n        except Exception as e:\n            logger.warning(f\"Failed to save cache entry {key}: {e}\")\n    \n    def _load_from_disk(self, key: str) -> Optional[Any]:\n        \"\"\"Load entry from disk.\"\"\"\n        try:\n            cache_file = self.cache_dir / f\"{self._hash_key(key)}.cache\"\n            if not cache_file.exists():\n                return None\n            \n            with open(cache_file, 'rb') as f:\n                if self.compression:\n                    import gzip\n                    with gzip.open(f, 'rb') as gz:\n                        return pickle.load(gz)\n                else:\n                    return pickle.load(f)\n                    \n        except Exception as e:\n            logger.warning(f\"Failed to load cache entry {key}: {e}\")\n            return None\n    \n    def _delete_from_disk(self, key: str) -> None:\n        \"\"\"Delete entry from disk.\"\"\"\n        try:\n            cache_file = self.cache_dir / f\"{self._hash_key(key)}.cache\"\n            if cache_file.exists():\n                cache_file.unlink()\n        except Exception as e:\n            logger.warning(f\"Failed to delete cache file {key}: {e}\")\n    \n    def _clear_disk_cache(self) -> None:\n        \"\"\"Clear all disk cache files.\"\"\"\n        try:\n            for cache_file in self.cache_dir.glob(\"*.cache\"):\n                cache_file.unlink()\n            \n            index_file = self.cache_dir / \"index.json\"\n            if index_file.exists():\n                index_file.unlink()\n                \n        except Exception as e:\n            logger.warning(f\"Failed to clear disk cache: {e}\")\n    \n    def _update_index(self, key: str, entry: CacheEntry) -> None:\n        \"\"\"Update cache index.\"\"\"\n        try:\n            index_file = self.cache_dir / \"index.json\"\n            \n            # Load existing index\n            index = {}\n            if index_file.exists():\n                with open(index_file, 'r') as f:\n                    index = json.load(f)\n            \n            # Update entry\n            index[key] = entry.to_dict()\n            \n            # Save index\n            with open(index_file, 'w') as f:\n                json.dump(index, f, indent=2)\n                \n        except Exception as e:\n            logger.warning(f\"Failed to update cache index: {e}\")\n    \n    def _hash_key(self, key: str) -> str:\n        \"\"\"Generate hash for cache key.\"\"\"\n        return hashlib.sha256(key.encode()).hexdigest()[:16]\n    \n    def __del__(self):\n        \"\"\"Cleanup on destruction.\"\"\"\n        try:\n            if hasattr(self, '_executor'):\n                self._executor.shutdown(wait=True)\n        except Exception:\n            pass\n\n\nclass CacheManager:\n    \"\"\"Global cache manager with multiple cache instances.\"\"\"\n    \n    def __init__(self):\n        self._caches: Dict[str, IntelligentCache] = {}\n        self._cache_lock = Lock()\n    \n    def get_cache(self, name: str = \"default\") -> IntelligentCache:\n        \"\"\"Get or create cache instance.\"\"\"\n        with self._cache_lock:\n            if name not in self._caches:\n                config = get_config()\n                self._caches[name] = IntelligentCache(\n                    max_size_mb=config.cache.max_size_mb,\n                    ttl_seconds=config.cache.ttl_seconds,\n                    cache_dir=str(Path(config.cache.cache_dir) / name),\n                    persistent=True,\n                    compression=True\n                )\n            return self._caches[name]\n    \n    def clear_all(self) -> None:\n        \"\"\"Clear all caches.\"\"\"\n        with self._cache_lock:\n            for cache in self._caches.values():\n                cache.clear()\n    \n    def get_global_stats(self) -> Dict[str, Any]:\n        \"\"\"Get statistics for all caches.\"\"\"\n        stats = {}\n        with self._cache_lock:\n            for name, cache in self._caches.items():\n                stats[name] = cache.get_stats()\n        return stats\n\n\n# Global cache manager\ncache_manager = CacheManager()\n\n\ndef cached(\n    ttl_seconds: Optional[int] = None,\n    cache_name: str = \"default\",\n    key_func: Optional[Callable] = None,\n    tags: Optional[list[str]] = None\n):\n    \"\"\"Decorator for caching function results.\"\"\"\n    def decorator(func: Callable[..., T]) -> Callable[..., T]:\n        def wrapper(*args, **kwargs) -> T:\n            # Generate cache key\n            if key_func:\n                cache_key = key_func(*args, **kwargs)\n            else:\n                cache_key = f\"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items())))}\"\n            \n            # Get cache\n            cache = cache_manager.get_cache(cache_name)\n            \n            # Try to get from cache\n            result = cache.get(cache_key)\n            if result is not None:\n                return result\n            \n            # Execute function\n            result = func(*args, **kwargs)\n            \n            # Cache result\n            cache.put(cache_key, result, ttl_seconds=ttl_seconds, tags=tags or [])\n            \n            return result\n        \n        return wrapper\n    return decorator\n\n\n@contextmanager\ndef cache_context(cache_name: str = \"default\"):\n    \"\"\"Context manager for cache operations.\"\"\"\n    cache = cache_manager.get_cache(cache_name)\n    try:\n        yield cache\n    except Exception as e:\n        logger.error(f\"Cache context error: {e}\")\n        raise"
            }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class IntelligentCache:
    """High-performance intelligent cache with adaptive eviction."""
    
    def __init__(
        self,
        max_size_mb: int = 500,
        ttl_seconds: int = 3600,
        cache_dir: Optional[str] = None,
        persistent: bool = True,
        compression: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.persistent = persistent
        self.compression = compression
        
        # Thread-safe storage
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = RLock()
        self._stats_lock = Lock()
        
        # Cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".modelcard" / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0,
            "entries": 0
        }
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache")
        
        # Load persistent cache
        if self.persistent:
            self._load_persistent_cache()
        
        logger.info(f"Intelligent cache initialized: {max_size_mb}MB, TTL: {ttl_seconds}s")
    
    def get(self, key: str, default: T = None) -> Optional[T]:
        """Get value from cache with intelligent prefetching."""
        start_time = time.time()
        
        try:
            with self._cache_lock:
                # Check memory cache
                if key in self._memory_cache:
                    entry = self._memory_cache[key]
                    
                    # Check expiration
                    if entry.is_expired:
                        del self._memory_cache[key]
                        self._update_stats("miss")
                        logger.debug(f"Cache expired for key: {key}")
                        return default
                    
                    # Update access info
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    self._update_stats("hit")
                    
                    duration_ms = (time.time() - start_time) * 1000
                    logger.debug(f"Cache hit for {key}: {duration_ms:.2f}ms")
                    
                    return entry.value
                
                # Check persistent cache
                if self.persistent:
                    persistent_value = self._load_from_disk(key)
                    if persistent_value is not None:
                        # Add back to memory cache
                        self.put(key, persistent_value, ttl_seconds=self.default_ttl)
                        self._update_stats("hit")
                        
                        duration_ms = (time.time() - start_time) * 1000
                        logger.debug(f"Cache hit from disk for {key}: {duration_ms:.2f}ms")
                        
                        return persistent_value
                
                self._update_stats("miss")
                
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Cache miss for {key}: {duration_ms:.2f}ms")
                
                return default
                
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            return default
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[list[str]] = None
    ) -> None:
        """Put value in cache with intelligent eviction."""
        try:
            ttl = ttl_seconds or self.default_ttl
            tags = tags or []
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            with self._cache_lock:
                # Check if we need to evict
                self._ensure_capacity(size_bytes)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    size_bytes=size_bytes,
                    tags=tags,
                    ttl_seconds=ttl
                )
                
                # Store in memory
                old_entry = self._memory_cache.get(key)
                self._memory_cache[key] = entry
                
                # Update stats
                if old_entry:
                    self._stats["size_bytes"] -= old_entry.size_bytes
                else:
                    self._stats["entries"] += 1
                
                self._stats["size_bytes"] += size_bytes
                
                # Store persistently
                if self.persistent:
                    self._executor.submit(self._save_to_disk, key, value, entry)
                
                logger.debug(f"Cached {key}: {size_bytes} bytes, TTL: {ttl}s")
                
        except Exception as e:
            logger.error(f"Cache put error for {key}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            with self._cache_lock:
                if key in self._memory_cache:
                    entry = self._memory_cache[key]
                    del self._memory_cache[key]
                    
                    self._stats["size_bytes"] -= entry.size_bytes
                    self._stats["entries"] -= 1
                    
                    # Delete from disk
                    if self.persistent:
                        self._executor.submit(self._delete_from_disk, key)
                    
                    logger.debug(f"Deleted cache entry: {key}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False
    
    def clear(self, tags: Optional[list[str]] = None) -> int:
        """Clear cache entries, optionally by tags."""
        try:
            cleared = 0
            
            with self._cache_lock:
                if tags:
                    # Clear by tags
                    to_delete = []
                    for key, entry in self._memory_cache.items():
                        if any(tag in entry.tags for tag in tags):
                            to_delete.append(key)
                    
                    for key in to_delete:
                        if self.delete(key):
                            cleared += 1
                else:
                    # Clear all
                    cleared = len(self._memory_cache)
                    self._memory_cache.clear()
                    self._stats["size_bytes"] = 0
                    self._stats["entries"] = 0
                    
                    # Clear disk cache
                    if self.persistent:
                        self._executor.submit(self._clear_disk_cache)
            
            logger.info(f"Cleared {cleared} cache entries")
            return cleared
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
        
        with self._cache_lock:
            stats["hit_rate"] = stats["hits"] / (stats["hits"] + stats["misses"]) if (stats["hits"] + stats["misses"]) > 0 else 0
            stats["size_mb"] = stats["size_bytes"] / (1024 * 1024)
            stats["utilization"] = stats["size_bytes"] / self.max_size_bytes
            stats["active_entries"] = len(self._memory_cache)
        
        return stats
    
    def _ensure_capacity(self, required_bytes: int) -> None:
        """Ensure cache has capacity for new entry."""
        if self._stats["size_bytes"] + required_bytes <= self.max_size_bytes:
            return
        
        # Calculate how much to evict
        bytes_to_evict = (self._stats["size_bytes"] + required_bytes) - self.max_size_bytes
        bytes_to_evict = max(bytes_to_evict, self.max_size_bytes * 0.1)  # Evict at least 10%
        
        evicted = self._evict_entries(bytes_to_evict)
        logger.debug(f"Evicted {evicted} entries to free {bytes_to_evict} bytes")
    
    def _evict_entries(self, target_bytes: int) -> int:
        """Evict entries using adaptive algorithm."""
        evicted_count = 0
        evicted_bytes = 0
        
        # Sort entries by eviction priority (LFU + LRU + size)
        entries = list(self._memory_cache.items())
        entries.sort(key=lambda x: self._eviction_score(x[1]))
        
        for key, entry in entries:
            if evicted_bytes >= target_bytes:
                break
            
            del self._memory_cache[key]
            evicted_bytes += entry.size_bytes
            evicted_count += 1
            
            # Delete from disk
            if self.persistent:
                self._executor.submit(self._delete_from_disk, key)
        
        self._stats["size_bytes"] -= evicted_bytes
        self._stats["entries"] -= evicted_count
        self._stats["evictions"] += evicted_count
        
        return evicted_count
    
    def _eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score (lower = evict first)."""
        # Factors: frequency, recency, size, age
        frequency_score = 1.0 / max(entry.access_count, 1)
        recency_score = (datetime.now() - entry.last_accessed).total_seconds() / 3600  # hours
        size_score = entry.size_bytes / (1024 * 1024)  # MB
        age_score = entry.age_seconds / 3600  # hours
        
        # Weighted combination
        return frequency_score * 0.3 + recency_score * 0.3 + size_score * 0.2 + age_score * 0.2
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            if hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            else:
                # Serialize to estimate size
                serialized = pickle.dumps(value)
                return len(serialized)
        except Exception:
            # Fallback estimate
            return 1024
    
    def _update_stats(self, event: str) -> None:
        """Update cache statistics."""
        with self._stats_lock:
            if event in self._stats:
                self._stats[event] += 1
    
    def _load_persistent_cache(self) -> None:
        """Load cache from disk."""
        try:
            index_file = self.cache_dir / "index.json"
            if not index_file.exists():
                return
            
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            loaded = 0
            for key, entry_data in index.items():
                try:
                    entry = CacheEntry.from_dict(entry_data)
                    if not entry.is_expired:
                        value = self._load_from_disk(key)
                        if value is not None:
                            self._memory_cache[key] = entry
                            self._stats["size_bytes"] += entry.size_bytes
                            self._stats["entries"] += 1
                            loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
            
            if loaded > 0:
                logger.info(f"Loaded {loaded} entries from persistent cache")
                
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
    
    def _save_to_disk(self, key: str, value: Any, entry: CacheEntry) -> None:
        """Save entry to disk."""
        try:
            # Save value
            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            with open(cache_file, 'wb') as f:
                if self.compression:
                    import gzip
                    with gzip.open(f, 'wb') as gz:
                        pickle.dump(value, gz)
                else:
                    pickle.dump(value, f)
            
            # Update index
            self._update_index(key, entry)
            
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key}: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load entry from disk."""
        try:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                if self.compression:
                    import gzip
                    with gzip.open(f, 'rb') as gz:
                        return pickle.load(gz)
                else:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            return None
    
    def _delete_from_disk(self, key: str) -> None:
        """Delete entry from disk."""
        try:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete cache file {key}: {e}")
    
    def _clear_disk_cache(self) -> None:
        """Clear all disk cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            index_file = self.cache_dir / "index.json"
            if index_file.exists():
                index_file.unlink()
                
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")
    
    def _update_index(self, key: str, entry: CacheEntry) -> None:
        """Update cache index."""
        try:
            index_file = self.cache_dir / "index.json"
            
            # Load existing index
            index = {}
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
            
            # Update entry
            index[key] = entry.to_dict()
            
            # Save index
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update cache index: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
        except Exception:
            pass


class CacheManager:
    """Global cache manager with multiple cache instances."""
    
    def __init__(self):
        self._caches: Dict[str, IntelligentCache] = {}
        self._cache_lock = Lock()
    
    def get_cache(self, name: str = "default") -> IntelligentCache:
        """Get or create cache instance."""
        with self._cache_lock:
            if name not in self._caches:
                config = get_config()
                self._caches[name] = IntelligentCache(
                    max_size_mb=config.cache.max_size_mb,
                    ttl_seconds=config.cache.ttl_seconds,
                    cache_dir=str(Path(config.cache.cache_dir) / name),
                    persistent=True,
                    compression=True
                )
            return self._caches[name]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._cache_lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        with self._cache_lock:
            for name, cache in self._caches.items():
                stats[name] = cache.get_stats()
        return stats


# Global cache manager
cache_manager = CacheManager()


def cached(
    ttl_seconds: Optional[int] = None,
    cache_name: str = "default",
    key_func: Optional[Callable] = None,
    tags: Optional[list[str]] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Get cache
            cache = cache_manager.get_cache(cache_name)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(cache_key, result, ttl_seconds=ttl_seconds, tags=tags or [])
            
            return result
        
        return wrapper
    return decorator


@contextmanager
def cache_context(cache_name: str = "default"):
    """Context manager for cache operations."""
    cache = cache_manager.get_cache(cache_name)
    try:
        yield cache
    except Exception as e:
        logger.error(f"Cache context error: {e}")
        raise