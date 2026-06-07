"""API-proxied cache — routes GET/SET through the Fluiq backend.

Used when the SDK cannot reach Redis directly (e.g. Render internal-only URL).
GET is synchronous with a 1 s timeout so cache misses are fast-fail.
SET is fire-and-forget in a daemon thread so it never blocks the LLM call.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from fluiq.optimization.caching.base import BaseCache


class ApiCache(BaseCache):
    """Cache that proxies reads/writes through the Fluiq API.

    Does not require direct Redis access — all operations go through
    ``GET /optimize/cache/{key}`` and ``POST /optimize/cache``.

    A write-through in-process dict is maintained so that keys SET within
    the same process are available immediately on GET, even before the
    fire-and-forget background POST has completed on the backend.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        default_ttl: Optional[int] = None,
    ) -> None:
        self._base = endpoint.rstrip("/")
        self._api_key = api_key
        self._default_ttl = default_ttl
        self._local: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def get(self, key: str) -> Optional[Any]:
        if key in self._local:
            return self._local[key]
        try:
            import requests
            resp = requests.get(
                f"{self._base}/optimize/cache/{key}",
                headers=self._headers(),
                timeout=1.0,
            )
            if resp.status_code == 200:
                return resp.json().get("value")
            return None
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self._local[key] = value
        effective_ttl = int(ttl) if ttl is not None else self._default_ttl
        payload: dict[str, Any] = {"key": key, "value": value}
        if effective_ttl is not None:
            payload["ttl"] = effective_ttl

        def _fire() -> None:
            try:
                import requests
                requests.post(
                    f"{self._base}/optimize/cache",
                    json=payload,
                    headers=self._headers(),
                    timeout=3.0,
                )
            except Exception:
                pass

        threading.Thread(target=_fire, daemon=True).start()

    def delete(self, key: str) -> bool:
        self._local.pop(key, None)
        return False

    def clear(self) -> None:
        self._local.clear()
