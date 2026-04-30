import time
from typing import Any, Callable, Dict, Optional

from fluiq.optimization.caching.base import BaseCache, make_key
from fluiq.optimization.caching.memory import InMemoryCache
from fluiq.optimization.caching._tracing import emit_cache_trace


class ToolCache:
    """Caches deterministic agent-tool results keyed by ``(name, args)``.

    Wraps a registry of tool callables so an agent loop can call
    ``tools(name, **kwargs)`` and reuse cached output for repeat calls.
    Best fit for tools whose output depends only on their arguments \u2014
    web fetches, DB reads, code interpreters, retrieval calls. Don't
    register tools with hidden state (clocks, random seeds, write side
    effects) unless you also pass a busting argument.

    The cache key includes the tool name plus a normalized JSON of the
    keyword arguments, so two calls with the same name and args share a
    slot regardless of dict ordering.

    Set ``trace=True`` to emit a ``cache`` span per call carrying the
    hit/miss flag for the dashboard hit-rate card.
    """

    def __init__(
        self,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        backend: Optional[BaseCache] = None,
        ttl: Optional[float] = None,
        trace: bool = False,
    ):
        # See EmbeddingCache: ``backend or ...`` clobbers an empty shared
        # InMemoryCache because of its ``__len__``.
        self.backend = backend if backend is not None else InMemoryCache(max_size=1024)
        self.ttl = ttl
        self.trace = trace
        self._tools: Dict[str, Callable[..., Any]] = dict(tools or {})

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Add or replace a tool implementation."""
        self._tools[name] = fn

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    @property
    def tools(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._tools)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(
                f"Unknown tool {name!r}. Registered: {sorted(self._tools)}"
            )
        start = time.time()
        key = make_key("tool", name, kwargs)
        cached = self.backend.get(key)
        if cached is not None:
            if self.trace:
                emit_cache_trace(
                    kind="tool", model=name,
                    hits=1, misses=0, latency=time.time() - start,
                    function=f"tool:{name}",
                )
            return cached
        result = self._tools[name](**kwargs)
        self.backend.set(key, result, ttl=self.ttl)
        if self.trace:
            emit_cache_trace(
                kind="tool", model=name,
                hits=0, misses=1, latency=time.time() - start,
                function=f"tool:{name}",
            )
        return result

    def __call__(self, name: str, **kwargs: Any) -> Any:
        return self.call(name, **kwargs)

    def invalidate(self, name: str, **kwargs: Any) -> bool:
        return self.backend.delete(make_key("tool", name, kwargs))
