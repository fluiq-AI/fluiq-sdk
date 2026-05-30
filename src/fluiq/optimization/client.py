"""Optimization client — trace-analysis-driven caching via API proxy.

On first LLM call the client lazily fetches the cache profile from
``/optimize/profile``.  The profile is provisioned by Fluiq's backend after
analysing the account's historical traces and contains:

    {
        "models":       ["gpt-4o", ...], # models whose responses to cache
        "ttl_seconds":  3600             # default TTL
    }

Cache reads/writes are then proxied through ``GET /optimize/cache/{key}``
and ``POST /optimize/cache`` so no direct Redis access is required.

An empty ``models`` list means "cache all models".

Cached payload format
---------------------
LLM traces store a dict::

    {
        "type":        "llm",
        "response":    str | None,
        "tool_calls":  list | None,   # OpenAI format
        "tool_uses":   list | None,   # Anthropic format
        "mcp_calls":   list | None,
        "mcp_results": list | None,
        "mcp_servers": list | None,
    }

Embedding traces store::

    {"type": "embedding", "response": <serialised CreateEmbeddingResponse dict>}

Old entries written before this format are plain strings; they are wrapped
on read for backwards compatibility.
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
    url = f"{cfg['endpoint']}/{cfg['version']}/optimize/profile"

    import requests
    resp = requests.get(url, headers={"x-api-key": cfg["api_key"]}, timeout=5.0)
    if resp.status_code == 200:
        profile = resp.json()
        s = _state()
        s.profile = profile
        from fluiq.optimization.caching.api_cache import ApiCache
        cfg = _config()
        s.cache = ApiCache(
            f"{cfg['endpoint']}/{cfg['version']}",
            cfg["api_key"],
            default_ttl=profile.get("ttl_seconds"),
        )
               
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
        d.get("model") or "",
        _messages_from(d),
        d.get("system") or d.get("system_instruction") or "",
        d.get("tools") or d.get("tool_config"),
        d.get("mcp_servers"),
    )


def _embedding_cache_key(d: dict[str, Any]) -> str:
    return make_key(
        "embedding",
        d.get("model", ""),
        d.get("input") or d.get("prompt") or d.get("contents") or d.get("texts") or "",
    )


def _model_allowed(model: str) -> bool:
    s = _state()
    allowed = (s.profile or {}).get("models", [])
    return not allowed or model in allowed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_cache(kwargs: dict[str, Any]) -> Optional[dict]:
    """Return the cached payload dict for *kwargs*, or ``None`` on miss.

    The payload dict has the shape described in the module docstring.
    Old string-valued entries are wrapped for backwards compatibility.

    Only runs when ``fluiq.optimize()`` has been called and the backend has
    provisioned a Redis instance.
    """
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    if not _model_allowed(kwargs.get("model", "")):
        return None
    raw = s.cache.get(_cache_key(kwargs))
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"type": "llm", "response": raw}
    if isinstance(raw, dict):
        return raw
    return None


def lookup_embedding_cache(kwargs: dict[str, Any]) -> Optional[dict]:
    """Return the cached embedding payload for *kwargs*, or ``None`` on miss."""
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    if not _model_allowed(kwargs.get("model", "")):
        return None
    raw = s.cache.get(_embedding_cache_key(kwargs))
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    return None


# ---------------------------------------------------------------------------
# Vectorstore query-result cache  (with generation-based invalidation)
# ---------------------------------------------------------------------------

def _collection_gen_key(integration: str, target: str) -> str:
    return f"_vs_gen:{integration.lower()}:{target}"


def _get_collection_generation(integration: str, target: str) -> str:
    """Return the current mutation generation for this collection (default '0')."""
    s = _state()
    if s.cache is None:
        return "0"
    val = s.cache.get(_collection_gen_key(integration, target))
    return str(val) if val is not None else "0"


def vectorstore_cache_key(
    integration: str,
    target: str,
    query: Any,
    top_k: Any,
    filter_val: Any,
) -> str:
    """Stable key for a vector-store query.

    Includes the collection's mutation generation so that any mutation
    (add / upsert / update / delete) automatically invalidates all cached
    query results for that collection without needing explicit key deletion.
    """
    _ensure_initialized()
    gen = _get_collection_generation(integration, target or "")
    return make_key("vectorstore", integration, target or "", gen, query or "", top_k, filter_val)


def invalidate_vectorstore_cache(integration: str, target: str) -> None:
    """Bump the mutation generation for *target*, making all cached query
    results for that collection unreachable.

    Called automatically by mutation wrappers (add, upsert, update, delete).
    Uses ttl=0 so the generation key never expires — it outlives individual
    cached entries and stays consistent across restarts.
    """
    import time as _time
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return
    s.cache.set(_collection_gen_key(integration, target or ""), str(_time.time()), ttl=0)
    print(f"[fluiq.cache] invalidated  integration={integration!r}  target={target!r}", flush=True)


def lookup_vectorstore_cache(cache_key: str) -> Optional[dict]:
    """Return cached vectorstore query payload, or ``None`` on miss."""
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    raw = s.cache.get(cache_key)
    if isinstance(raw, dict):
        return raw
    return None


def populate_vectorstore_cache(cache_key: str, result: dict) -> None:
    """Store a vectorstore query result in Redis."""
    s = _state()
    if s.cache is None:
        return
    s.cache.set(cache_key, {"type": "vectorstore", "result": result})


def _tool_cache_key(tool_name: str, args_json: str) -> str:
    return make_key("tool", tool_name, args_json)


def lookup_tool_cache(tool_name: str, args: "dict | str") -> Optional[Any]:
    """Return the cached result for a tool call, or ``None`` on miss.

    *args* can be a dict (recommended) or a JSON string.  Keys are sorted
    before hashing so ``{"b": 2, "a": 1}`` and ``{"a": 1, "b": 2}`` produce
    the same cache key.
    """
    import json as _json
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    if isinstance(args, dict):
        args_str = _json.dumps(args, sort_keys=True)
    else:
        try:
            args_str = _json.dumps(_json.loads(str(args)), sort_keys=True)
        except Exception:
            args_str = str(args)
    raw = s.cache.get(_tool_cache_key(tool_name, args_str))
    if isinstance(raw, dict) and raw.get("type") == "tool":
        return raw.get("result")
    return None


def populate_tool_cache(tool_name: str, args: "dict | str", result: Any) -> None:
    """Store a tool call result in the cache keyed by (tool_name, args)."""
    import json as _json
    s = _state()
    if s.cache is None:
        return
    if isinstance(args, dict):
        args_str = _json.dumps(args, sort_keys=True)
    else:
        try:
            args_str = _json.dumps(_json.loads(str(args)), sort_keys=True)
        except Exception:
            args_str = str(args)
    s.cache.set(_tool_cache_key(tool_name, args_str), {"type": "tool", "result": result})


def _function_cache_key(func_name: str, args_str: str) -> str:
    return make_key("function", func_name, args_str)


def lookup_function_cache(func_name: str, args_str: str) -> Optional[dict]:
    """Return ``{"type": "function", "result": <value>}`` on hit, ``None`` on miss.

    Returns a dict (not the raw value) so callers can distinguish a cached
    ``None`` result from a cache miss.
    """
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    raw = s.cache.get(_function_cache_key(func_name, args_str))
    if isinstance(raw, dict) and raw.get("type") == "function":
        return raw
    return None


def populate_function_cache(func_name: str, args_str: str, result: Any) -> None:
    """Store a traced function's return value in Redis."""
    s = _state()
    if s.cache is None:
        return
    s.cache.set(_function_cache_key(func_name, args_str), {"type": "function", "result": result})


# ---------------------------------------------------------------------------
# MCP list_tools() cache  (keyed by server URL)
# ---------------------------------------------------------------------------

def _mcp_list_tools_key(server_url: str) -> str:
    return make_key("mcp_list_tools", server_url)


def lookup_mcp_tools_cache(server_url: str) -> Optional[list]:
    """Return the cached list_tools() result for this MCP server, or None on miss."""
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    raw = s.cache.get(_mcp_list_tools_key(server_url))
    if isinstance(raw, dict) and raw.get("type") == "mcp_list_tools":
        return raw.get("tools")
    return None


def populate_mcp_tools_cache(server_url: str, tools: list) -> None:
    """Cache the list_tools() response for this MCP server."""
    s = _state()
    if s.cache is None:
        return
    s.cache.set(_mcp_list_tools_key(server_url), {"type": "mcp_list_tools", "tools": tools})


def invalidate_mcp_tools_cache(server_url: str) -> None:
    """Evict the cached list_tools() for this server — called on re-initialize (server restart)."""
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return
    s.cache.delete(_mcp_list_tools_key(server_url))


# ---------------------------------------------------------------------------
# MCP call_tool() cache  (keyed by server URL + tool name + args hash)
# ---------------------------------------------------------------------------

def _mcp_call_key(server_url: str, tool_name: str, args_json: str) -> str:
    return make_key("mcp_call", server_url, tool_name, args_json)


def lookup_mcp_call_cache(server_url: str, tool_name: str, args: "dict | str") -> Optional[dict]:
    """Return the cached call_tool() payload for (server_url, tool_name, args), or None on miss.

    Returns the full payload dict so the caller can reconstruct the result.
    """
    import json as _json
    _ensure_initialized()
    s = _state()
    if s.cache is None:
        return None
    if isinstance(args, dict):
        args_str = _json.dumps(args, sort_keys=True)
    else:
        try:
            args_str = _json.dumps(_json.loads(str(args)), sort_keys=True)
        except Exception:
            args_str = str(args)
    raw = s.cache.get(_mcp_call_key(server_url, tool_name, args_str))
    if isinstance(raw, dict) and raw.get("type") == "mcp_call":
        return raw
    return None


def populate_mcp_call_cache(
    server_url: str,
    tool_name: str,
    args: "dict | str",
    content: list,
    is_error: bool = False,
) -> None:
    """Store an MCP call_tool() result in Redis."""
    import json as _json
    s = _state()
    if s.cache is None:
        return
    if isinstance(args, dict):
        args_str = _json.dumps(args, sort_keys=True)
    else:
        try:
            args_str = _json.dumps(_json.loads(str(args)), sort_keys=True)
        except Exception:
            args_str = str(args)
    s.cache.set(
        _mcp_call_key(server_url, tool_name, args_str),
        {"type": "mcp_call", "content": content, "isError": is_error},
    )


def populate_cache(data: dict[str, Any]) -> None:
    """Store a real LLM or embedding trace into Redis.

    *data* is the LogTrace dict produced by the integration patches.
    No-ops when the cache is not initialised, the model is out of scope,
    or there is nothing meaningful to cache.
    """
    s = _state()
    if s.cache is None:
        return

    model = data.get("model", "")

    if data.get("api") == "embeddings":
        embedding_response = data.get("response")
        if not embedding_response:
            return
        if not _model_allowed(model):
            return
        s.cache.set(_embedding_cache_key(data), {"type": "embedding", "response": embedding_response})
        return

    # LLM response
    response_text = data.get("response") or None
    tool_calls = data.get("tool_calls") or None       # OpenAI
    tool_uses = data.get("tool_uses") or None         # Anthropic
    function_calls = data.get("function_calls") or None  # Gemini
    mcp_calls = data.get("mcp_calls") or None

    if not any([response_text, tool_calls, tool_uses, function_calls, mcp_calls]):
        return
    if not _model_allowed(model):
        return

    payload = {
        "type": "llm",
        "response": response_text,
        "tool_calls": tool_calls,
        "tool_uses": tool_uses,
        "function_calls": function_calls,
        "mcp_calls": mcp_calls,
        "mcp_results": data.get("mcp_results") or None,
        "mcp_servers": data.get("mcp_servers") or None,
    }
    s.cache.set(_cache_key(data), payload)