import time
from typing import Callable, List, Optional, Sequence

from fluiq.optimization.caching.base import BaseCache, make_key
from fluiq.optimization.caching.memory import InMemoryCache
from fluiq.optimization.caching._tracing import emit_cache_trace


EmbedFn = Callable[[List[str]], List[List[float]]]


class EmbeddingCache:
    """Caches embedding vectors keyed by ``(model, text)``.

    Wraps any ``embed_fn(texts) -> list[list[float]]`` (OpenAI, Cohere,
    sentence-transformers, etc.). On each call only the cache misses are
    forwarded to the underlying model; hits are served from the backend in
    the original input order, so callers don't need to rewire indices.

    Parameters
    ----------
    embed_fn:
        The underlying embedding function. Must accept a list of strings and
        return a list of equal length, where each element is the embedding
        vector for the corresponding input.
    model:
        Model identifier baked into every cache key. Switching models
        invalidates all entries from previous models automatically.
    backend:
        Any :class:`BaseCache`. Defaults to an in-memory LRU of 10k entries.
    ttl:
        Optional default TTL in seconds, forwarded to the backend on writes.
    trace:
        When ``True``, emit a ``cache`` span per ``embed()`` call carrying
        per-batch hit/miss counts. Disabled by default to keep the cache
        free of network side-effects when ``fluiq.instrument()`` hasn't
        been called.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        model: str,
        backend: Optional[BaseCache] = None,
        ttl: Optional[float] = None,
        trace: bool = False,
    ):
        self.embed_fn = embed_fn
        self.model = model
        # ``backend or InMemoryCache(...)`` would replace a freshly-shared
        # empty in-memory backend (``__len__ == 0`` is falsy), silently
        # breaking the shared-backend guarantee assumed by auto_optimize().
        self.backend = backend if backend is not None else InMemoryCache(max_size=10_000)
        self.ttl = ttl
        self.trace = trace

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        start = time.time()
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_idx: List[int] = []
        miss_text: List[str] = []
        for i, text in enumerate(texts):
            key = make_key("embedding", self.model, text)
            cached = self.backend.get(key)
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(text)

        if miss_text:
            fresh = self.embed_fn(miss_text)
            if len(fresh) != len(miss_text):
                raise RuntimeError(
                    f"embed_fn returned {len(fresh)} vectors for "
                    f"{len(miss_text)} inputs"
                )
            for j, vec in zip(miss_idx, fresh):
                results[j] = vec
                self.backend.set(
                    make_key("embedding", self.model, texts[j]),
                    vec,
                    ttl=self.ttl,
                )

        if self.trace:
            emit_cache_trace(
                kind="embedding",
                model=self.model,
                hits=len(texts) - len(miss_idx),
                misses=len(miss_idx),
                latency=time.time() - start,
            )
        return [r if r is not None else [] for r in results]

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed(texts)
