"""Optimization client — trace-analysis-driven Redis caching.

On first LLM call the client lazily fetches the cache profile from
``/optimize/profile``.  The profile is provisioned by Fluiq's backend after
analysing the account's historical traces and contains:

    {
        "redis_url":    "redis://...",   # Fluiq-hosted Redis instance
        "models":       ["gpt-4o", ...], # models whose responses to cache
        "ttl_seconds":  3600             # default TTL
    }

An empty ``models`` list means "cache all models".
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from fluiq.optimization.caching.base import make_key

_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _state():
    from fluiq.optimization._state import _state as s
    return s


def _config():
    from fluiq.config import _config as c
    return c


# ---------------------------------------------------------------------------
# Lazy initialisation
# ---------------------------------------------------------------------------

def _ensure_initialized() -> None:
    s = _state()
    if s.initialized:
        return
    with _init_lock:
        if s.initialized:
            return
        s.initialized = True
        _fetch_profile()


def _fetch_profile() -> None:
    """Fetch the optimization profile from the Fluiq backend (non-blocking)."""
    cfg = _config()
    try:
        import requests
        resp = requests.get(
            f"{cfg['endpoint']}/{cfg['version']}/optimize/profile",
            headers={"x-api-key": cfg["api_key"]},
            timeout=5.0,
        )
        if resp.status_code == 200:
            profile = resp.json()
            s = _state()
            s.profile = profile
            redis_url = profile.get("redis_url")
            if redis_url:
                from fluiq.optimization.caching.redis_cache import RedisCache
                s.cache = RedisCache(
                    redis_url,
                    default_ttl=profile.get("ttl_seconds"),
                    prefix=profile.get("key_prefix", "fluiq:"),
                )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Key helpers — must produce the same key for pre-call kwargs and post-call
# trace dicts so lookup and populate are in sync.
# ---------------------------------------------------------------------------

def _messages_from(d: dict[str, Any]) -> Any:
    """Normalise across OpenAI (messages), Anthropic (messages), Gemini (contents / input)."""
    return d.get("messages") or d.get("contents") or d.get("input") or ""


def _cache_key(d: dict[str, Any]) -> str:
    return make_key(
        "llm",
        d.get("model", ""),
        _messages_from(d),
        d.get("system", ""),
    )


def _model_allowed(model: str) -> bool:
    s = _state()
    allowed = (s.profile or {}).get("models", [])
    return not allowed or model in allowed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_cache(kwargs: dict[str, Any]) -> Optional[str]:
    """Return the cached response text for *kwargs*, or ``None`` on miss.

    Only runs when ``fluiq.optimize()`` has been called and the backend has
    provisioned a Redis instance.
    """
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    if not _model_allowed(kwargs.get("model", "")):
        return None
    return s.cache.get(_cache_key(kwargs))


def populate_cache(data: dict[str, Any]) -> None:
    """Store the response text from a real LLM trace into Redis.

    *data* is the LogTrace dict produced by the integration patches.
    No-ops if the cache is not initialised or the model is not in scope.
    """
    s = _state()
    if s.cache is None:
        return
    response_text = data.get("response", "")
    if not response_text:
        return
    if not _model_allowed(data.get("model", "")):
        return
    s.cache.set(_cache_key(data), response_text)