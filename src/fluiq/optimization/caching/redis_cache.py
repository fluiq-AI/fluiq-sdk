from __future__ import annotations

import json
from typing import Any, Optional

from fluiq.optimization.caching.base import BaseCache


class RedisCache(BaseCache):
    """Redis-backed cache for Fluiq paid plans.

    Requires the ``redis`` package (``pip install redis``).
    The connection URL is provisioned by Fluiq — do not configure manually.
    """

    def __init__(
        self,
        url: str,
        *,
        default_ttl: Optional[float] = None,
        prefix: str = "fluiq:",
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisCache requires the 'redis' package. "
                "Install with: pip install redis"
            ) from exc
        self._client = redis.from_url(url, decode_responses=True)
        self._default_ttl = int(default_ttl) if default_ttl else None
        self._prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self._client.get(self._k(key))
            return json.loads(raw) if raw is not None else None
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        try:
            effective_ttl = int(ttl) if ttl is not None else self._default_ttl
            serialized = json.dumps(value)
            if effective_ttl:
                self._client.setex(self._k(key), effective_ttl, serialized)
            else:
                self._client.set(self._k(key), serialized)
        except Exception:
            pass

    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(self._k(key)))
        except Exception:
            return False

    def clear(self) -> None:
        try:
            keys = self._client.keys(f"{self._prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception:
            pass