from __future__ import annotations

import json
from typing import Any, Optional

from fluiq.optimization.caching.base import BaseCache

# _LOG = "[fluiq.cache]"


# ---------------------------------------------------------------------------
# Numpy-aware JSON helpers (numpy is optional — plain dicts work without it)
# ---------------------------------------------------------------------------

class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return {
                    "__ndarray__": True,
                    "dtype": str(obj.dtype),
                    "shape": list(obj.shape),
                    "data": obj.tolist(),
                }
        except ImportError:
            pass
        # Pydantic models (Anthropic, OpenAI SDK response objects)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # SimpleNamespace, dataclasses, other plain objects with __dict__
        d = getattr(obj, "__dict__", None)
        if d is not None:
            return d
        return super().default(obj)


def _decode_hook(d: dict) -> Any:
    if d.get("__ndarray__"):
        import numpy as np
        return np.array(d["data"], dtype=d["dtype"]).reshape(d["shape"])
    return d


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
        # print(f"{_LOG} connecting to Redis  url={url!r}  prefix={prefix!r}  ttl={default_ttl}", flush=True)
        try:
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
            # print(f"{_LOG} Redis connection OK", flush=True)
        except Exception as exc:
            # print(f"{_LOG} Redis connection FAILED: {exc!r}", flush=True)
            raise
        self._default_ttl = int(default_ttl) if default_ttl else None
        self._prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        full_key = self._k(key)
        # print(f"{_LOG} GET  key={full_key!r}", flush=True)
        try:
            raw = self._client.get(full_key)
            if raw is None:
                # print(f"{_LOG} GET  result=MISS", flush=True)
                return None
            parsed = json.loads(raw, object_hook=_decode_hook)
            # print(f"{_LOG} GET  result=HIT  value={str(parsed)[:120]!r}", flush=True)
            return parsed
        except Exception as exc:
            # print(f"{_LOG} GET  error={exc!r}", flush=True)
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        full_key = self._k(key)
        effective_ttl = int(ttl) if ttl is not None else self._default_ttl
        # print(f"{_LOG} SET  key={full_key!r}  ttl={effective_ttl}  value={str(value)[:120]!r}", flush=True)
        try:
            serialized = json.dumps(value, cls=_Encoder)
            if effective_ttl:
                self._client.setex(full_key, effective_ttl, serialized)
            else:
                self._client.set(full_key, serialized)
            # print(f"{_LOG} SET  OK", flush=True)
        except Exception as exc:
            # print(f"{_LOG} SET  error={exc!r}", flush=True)
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