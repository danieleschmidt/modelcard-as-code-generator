"""Simplified intelligent caching system for model card generator."""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable, TypeVar
from datetime import datetime, timedelta
from threading import Lock
from contextlib import contextmanager

from .exceptions import ResourceError
from .logging_config import get_logger
from .config import get_config

logger = get_logger(__name__)
T = TypeVar('T')


class SimpleCache:
    """Simple high-performance cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if datetime.now() > entry["expires"]:
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return default
                
                # Update access time
                entry["accessed"] = datetime.now()
                self._stats["hits"] += 1
                return entry["value"]
            
            self._stats["misses"] += 1
            return default
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Put value in cache."""
        ttl = ttl_seconds or self.ttl_seconds
        
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            # Store entry
            self._cache[key] = {
                "value": value,
                "created": datetime.now(),
                "accessed": datetime.now(),
                "expires": datetime.now() + timedelta(seconds=ttl)
            }
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "entries": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate
            }
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k]["accessed"])
        del self._cache[lru_key]
        self._stats["evictions"] += 1


class CacheManager:
    """Global cache manager."""
    
    def __init__(self):
        self._caches: Dict[str, SimpleCache] = {}
        self._lock = Lock()
    
    def get_cache(self, name: str = "default") -> SimpleCache:
        """Get or create cache instance."""
        with self._lock:
            if name not in self._caches:
                self._caches[name] = SimpleCache()
            return self._caches[name]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        with self._lock:
            for name, cache in self._caches.items():
                stats[name] = cache.get_stats()
        return stats


# Global cache manager
cache_manager = CacheManager()


def cached(
    ttl_seconds: Optional[int] = None,
    cache_name: str = "default",
    key_func: Optional[Callable] = None
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
            cache.put(cache_key, result, ttl_seconds=ttl_seconds)
            
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