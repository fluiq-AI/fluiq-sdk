import time
from typing import Any, Callable, List, Optional

from fluiq.optimization.caching.base import BaseCache, make_key
from fluiq.optimization.caching.memory import InMemoryCache
from fluiq.optimization.caching._tracing import emit_cache_trace


class DocumentCache:
    """Caches preprocessed / chunked documents by source identifier.

    Use to skip re-chunking and re-cleaning when the same source document
    is ingested multiple times across runs (notebook reloads, repeated CI
    pipelines, batch reindex jobs).

    Keys can be any stable identifier — a URL, a file path, a content hash.
    Pass the same key on every run to get cache hits.

    Set ``trace=True`` to emit a ``cache`` span per :meth:`get_or_chunk`
    call so chunking hit/miss rates show up alongside embedding and prompt
    caches in the dashboard trace tree.
    """

    def __init__(
        self,
        backend: Optional[BaseCache] = None,
        ttl: Optional[float] = None,
        trace: bool = False,
    ):
        # See EmbeddingCache: ``backend or ...`` would clobber an empty
        # shared InMemoryCache because of its ``__len__``.
        self.backend = backend if backend is not None else InMemoryCache(max_size=4096)
        self.ttl = ttl
        self.trace = trace

    def get(self, source: str) -> Optional[List[str]]:
        return self.backend.get(make_key("document", source))

    def set(self, source: str, chunks: List[str]) -> None:
        self.backend.set(
            make_key("document", source),
            list(chunks),
            ttl=self.ttl,
        )

    def delete(self, source: str) -> bool:
        return self.backend.delete(make_key("document", source))

    def get_or_chunk(
        self,
        source: str,
        chunk_fn: Callable[..., List[str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[str]:
        """Return cached chunks or compute and store them.

        ``chunk_fn`` is invoked as ``chunk_fn(*args, **kwargs)`` only on
        cache miss. Pass the raw document text (or whatever the chunker
        needs) via ``*args`` / ``**kwargs`` so it's not loaded on hits.
        """
        start = time.time()
        cached = self.get(source)
        if cached is not None:
            if self.trace:
                emit_cache_trace(
                    kind="document", model=source,
                    hits=1, misses=0, latency=time.time() - start,
                    function="document_cache",
                )
            return cached
        chunks = list(chunk_fn(*args, **kwargs))
        self.set(source, chunks)
        if self.trace:
            emit_cache_trace(
                kind="document", model=source,
                hits=0, misses=1, latency=time.time() - start,
                function="document_cache",
            )
        return chunks
