import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from fluiq.optimization.caching.base import BaseCache


class InMemoryCache(BaseCache):
    """Thread-safe LRU cache with optional per-entry TTL.

    Backed by an ``OrderedDict``. Eviction is strictly LRU once ``max_size``
    is exceeded. ``default_ttl`` (seconds) applies to ``set`` calls that
    don't pass an explicit ttl; ``None`` means entries never expire.
    """

    def __init__(self, max_size: int = 1024, default_ttl: Optional[float] = None):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._data: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl is not None else None
        with self._lock:
            self._data[key] = (value, expires_at)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
