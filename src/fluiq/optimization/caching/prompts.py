import time
from typing import Any, Callable, Optional

from fluiq.optimization.caching.base import BaseCache, make_key
from fluiq.optimization.caching.memory import InMemoryCache
from fluiq.optimization.caching._tracing import emit_cache_trace


class PromptCache:
    """Caches LLM responses keyed by ``(model, prompt, params)``.

    Wraps any ``llm_fn(prompt, **params) -> Any``. Identical calls return
    the cached response without hitting the model. Useful for deterministic
    prompts (temperature 0), eval pipelines, and replayable RAG demos.

    Non-deterministic params (temperature, top_p, seed) are part of the
    cache key by default, so two calls with different sampling settings get
    different cache slots.

    Set ``trace=True`` to emit a ``cache`` span per call carrying the
    hit/miss flag for the dashboard hit-rate card.
    """

    def __init__(
        self,
        llm_fn: Callable[..., Any],
        model: str,
        backend: Optional[BaseCache] = None,
        ttl: Optional[float] = None,
        trace: bool = False,
    ):
        self.llm_fn = llm_fn
        self.model = model
        # See EmbeddingCache: ``backend or ...`` would clobber an empty
        # shared InMemoryCache because of its ``__len__``.
        self.backend = backend if backend is not None else InMemoryCache(max_size=1024)
        self.ttl = ttl
        self.trace = trace

    def call(self, prompt: str, **params: Any) -> Any:
        start = time.time()
        key = make_key("prompt", self.model, prompt, params)
        cached = self.backend.get(key)
        if cached is not None:
            if self.trace:
                emit_cache_trace(
                    kind="prompt", model=self.model,
                    hits=1, misses=0, latency=time.time() - start,
                )
            return cached
        result = self.llm_fn(prompt, **params)
        self.backend.set(key, result, ttl=self.ttl)
        if self.trace:
            emit_cache_trace(
                kind="prompt", model=self.model,
                hits=0, misses=1, latency=time.time() - start,
            )
        return result

    def __call__(self, prompt: str, **params: Any) -> Any:
        return self.call(prompt, **params)

    def invalidate(self, prompt: str, **params: Any) -> bool:
        return self.backend.delete(make_key("prompt", self.model, prompt, params))
