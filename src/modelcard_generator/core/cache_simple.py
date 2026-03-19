"""Minimal in-memory cache (no external dependencies)."""

import time
from typing import Any, Optional


class SimpleCache:
    def __init__(self):
        self._data: dict = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry and time.time() > expiry:
            del self._data[key]
            return None
        return value

    def put(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        expiry = time.time() + ttl_seconds if ttl_seconds else None
        self._data[key] = (value, expiry)

    def clear(self) -> None:
        self._data.clear()


class _CacheManager:
    _cache = SimpleCache()

    def get_cache(self) -> SimpleCache:
        return self._cache


cache_manager = _CacheManager()
