"""Distributed caching system for model card generation."""

import asyncio
import hashlib
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from threading import RLock
from concurrent.futures import ThreadPoolExecutor

from .logging_config import get_logger
from .config import get_config
from .exceptions import ResourceError

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    key: str
    value: T
    created_at: float
    last_accessed: float
    access_count: int
    ttl_seconds: Optional[float]
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at
    
    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_size_bytes = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                del self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                self._misses += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = len(str(value).encode('utf-8'))
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes
                del self._cache[key]
            
            # Create new entry
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=0,
                ttl_seconds=ttl,
                size_bytes=size_bytes
            )
            
            # Add to cache
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
            
            # Evict if necessary
            self._evict_if_needed()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full (must hold lock)."""
        while len(self._cache) > self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            entry = self._cache[oldest_key]
            self._current_size_bytes -= entry.size_bytes
            del self._cache[oldest_key]
            self._evictions += 1
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                del self._cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "size_bytes": self._current_size_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "memory_efficiency": self._current_size_bytes / max(1, len(self._cache))
            }
    
    def get_entries_info(self) -> List[Dict[str, Any]]:
        """Get information about cache entries."""
        with self._lock:
            return [
                {
                    "key": key,
                    "size_bytes": entry.size_bytes,
                    "age_seconds": entry.age_seconds,
                    "access_count": entry.access_count,
                    "ttl_seconds": entry.ttl_seconds,
                    "is_expired": entry.is_expired
                }
                for key, entry in self._cache.items()
            ]


class AsyncCache:
    """Async wrapper around cache with additional features."""
    
    def __init__(self, cache: LRUCache, executor: Optional[ThreadPoolExecutor] = None):
        self.cache = cache
        self.executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache")
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.cache.get, key, default)
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.cache.put, key, value, ttl_seconds)
    
    async def delete(self, key: str) -> bool:
        """Delete key asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.cache.delete, key)
    
    async def clear(self) -> None:
        """Clear cache asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.cache.clear)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.cache.get_stats)
    
    async def cleanup_if_needed(self) -> None:
        """Cleanup expired entries if needed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            loop = asyncio.get_event_loop()
            removed = await loop.run_in_executor(self.executor, self.cache.cleanup_expired)
            self._last_cleanup = now
            if removed > 0:
                logger.debug(f"Cache cleanup removed {removed} expired entries")


class PersistentCache:
    """Cache with disk persistence."""
    
    def __init__(self, cache_dir: str, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = LRUCache(max_size, default_ttl)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="persistent_cache")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Create safe filename from key
        safe_key = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"cache_{safe_key}.pkl"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from memory cache first, then disk."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(self.executor, self._load_from_disk, file_path)
                
                if data is not None:
                    entry_data, value = data
                    # Check if expired
                    if entry_data.get('ttl_seconds') is not None:
                        age = time.time() - entry_data.get('created_at', 0)
                        if age > entry_data['ttl_seconds']:
                            # Expired, remove file
                            await loop.run_in_executor(self.executor, file_path.unlink, True)
                            return default
                    
                    # Add back to memory cache
                    self.memory_cache.put(key, value, entry_data.get('ttl_seconds'))
                    return value
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
        
        return default
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value in both memory and disk cache."""
        # Put in memory cache
        self.memory_cache.put(key, value, ttl_seconds)
        
        # Put in disk cache
        try:
            file_path = self._get_file_path(key)
            entry_data = {
                'key': key,
                'created_at': time.time(),
                'ttl_seconds': ttl_seconds
            }
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._save_to_disk, file_path, entry_data, value)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, file_path: Path) -> Optional[tuple]:
        """Load data from disk (runs in executor)."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data.get('metadata'), data.get('value')
        except Exception:
            return None
    
    def _save_to_disk(self, file_path: Path, entry_data: Dict, value: Any) -> None:
        """Save data to disk (runs in executor)."""
        try:
            data = {
                'metadata': entry_data,
                'value': value
            }
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete from both memory and disk cache."""
        # Delete from memory
        memory_deleted = self.memory_cache.delete(key)
        
        # Delete from disk
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, file_path.unlink)
                return True
        except Exception as e:
            logger.warning(f"Failed to delete from disk cache: {e}")
        
        return memory_deleted
    
    async def clear(self) -> None:
        """Clear both memory and disk cache."""
        # Clear memory
        self.memory_cache.clear()
        
        # Clear disk
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._clear_disk_cache)
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")
    
    def _clear_disk_cache(self) -> None:
        """Clear disk cache (runs in executor)."""
        try:
            for file_path in self.cache_dir.glob("cache_*.pkl"):
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including disk usage."""
        memory_stats = self.memory_cache.get_stats()
        
        # Get disk usage
        try:
            loop = asyncio.get_event_loop()
            disk_stats = await loop.run_in_executor(self.executor, self._get_disk_stats)
        except Exception:
            disk_stats = {"disk_files": 0, "disk_size_bytes": 0}
        
        return {
            "memory": memory_stats,
            "disk": disk_stats,
            "total_entries": memory_stats["size"] + disk_stats["disk_files"]
        }
    
    def _get_disk_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        try:
            files = list(self.cache_dir.glob("cache_*.pkl"))
            total_size = sum(f.stat().st_size for f in files)
            return {
                "disk_files": len(files),
                "disk_size_bytes": total_size
            }
        except Exception:
            return {"disk_files": 0, "disk_size_bytes": 0}


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (persistent) tiers."""
    
    def __init__(self, l1_size: int = 500, l2_cache_dir: Optional[str] = None, default_ttl: Optional[float] = 3600):
        self.l1_cache = AsyncCache(LRUCache(l1_size, default_ttl))
        
        if l2_cache_dir:
            self.l2_cache: Optional[PersistentCache] = PersistentCache(l2_cache_dir, l1_size * 2, default_ttl)
        else:
            self.l2_cache = None
        
        self.default_ttl = default_ttl
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0
        }
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from L1, then L2, then return default."""
        # Try L1 first
        value = await self.l1_cache.get(key)
        if value is not None:
            self._stats["l1_hits"] += 1
            return value
        
        # Try L2 if available
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                await self.l1_cache.put(key, value, self.default_ttl)
                self._stats["l2_hits"] += 1
                return value
        
        self._stats["misses"] += 1
        return default
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Put value in both L1 and L2."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        
        # Put in L1
        await self.l1_cache.put(key, value, ttl)
        
        # Put in L2 if available
        if self.l2_cache:
            await self.l2_cache.put(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete from both L1 and L2."""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = await self.l2_cache.delete(key) if self.l2_cache else False
        return l1_deleted or l2_deleted
    
    async def clear(self) -> None:
        """Clear both L1 and L2."""
        await self.l1_cache.clear()
        if self.l2_cache:
            await self.l2_cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats() if self.l2_cache else {}
        
        total_requests = sum(self._stats.values())
        
        return {
            "l1": l1_stats,
            "l2": l2_stats,
            "hit_distribution": self._stats,
            "total_requests": total_requests,
            "overall_hit_rate": (self._stats["l1_hits"] + self._stats["l2_hits"]) / max(1, total_requests)
        }


# Cache decorators
def cached(ttl_seconds: Optional[float] = None, cache_instance: Optional[MultiLevelCache] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache = cache_instance or default_cache
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs) -> Any:
                # Create cache key
                key_data = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': args,
                    'kwargs': kwargs
                }
                cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
                
                # Try cache first
                result = await cache.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await cache.put(cache_key, result, ttl_seconds)
                return result
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs) -> Any:
                # For sync functions, we can't use async cache directly
                # This would need to be implemented with a sync cache or run in event loop
                return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator


# Global cache instances
config = get_config()
cache_config = config.cache

# Initialize default multi-level cache
default_cache = MultiLevelCache(
    l1_size=cache_config.max_size_mb * 10,  # Estimate entries from MB
    l2_cache_dir=cache_config.cache_dir if cache_config.enabled else None,
    default_ttl=float(cache_config.ttl_seconds)
)