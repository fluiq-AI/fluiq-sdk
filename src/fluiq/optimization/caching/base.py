import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCache(ABC):
    """Abstract key-value cache with TTL support.

    Implementations are expected to be thread-safe. ``ttl`` semantics:

    * ``None`` on ``set`` falls back to the backend's ``default_ttl``.
    * ``None`` for both means the entry never expires.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


def make_key(*parts: Any) -> str:
    """Stable SHA-256 over an ordered tuple of JSON-serializable parts.

    Used by the specialized caches to derive a deterministic key from
    ``(domain, model, payload, params)`` so the same inputs always hash to
    the same slot — across processes, machines, and Python versions.
    """
    payload = json.dumps([_normalize(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_normalize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _normalize(v[k]) for k in sorted(v.keys(), key=str)}
    # Pydantic models (Anthropic, OpenAI SDK objects) — stable dict representation
    if hasattr(v, "model_dump"):
        return _normalize(v.model_dump())
    # SimpleNamespace, dataclasses, other plain objects
    d = getattr(v, "__dict__", None)
    if d is not None:
        return _normalize(d)
    return str(v)
