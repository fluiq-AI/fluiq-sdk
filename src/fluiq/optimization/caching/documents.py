from typing import Any, Callable, List, Optional

from fluiq.optimization.caching.base import BaseCache, make_key
from fluiq.optimization.caching.memory import InMemoryCache


class DocumentCache:
    """Caches preprocessed / chunked documents by source identifier.

    Use to skip re-chunking and re-cleaning when the same source document
    is ingested multiple times across runs (notebook reloads, repeated CI
    pipelines, batch reindex jobs).

    Keys can be any stable identifier — a URL, a file path, a content hash.
    Pass the same key on every run to get cache hits.
    """

    def __init__(
        self,
        backend: Optional[BaseCache] = None,
        ttl: Optional[float] = None,
    ):
        # See EmbeddingCache: ``backend or ...`` would clobber an empty
        # shared InMemoryCache because of its ``__len__``.
        self.backend = backend if backend is not None else InMemoryCache(max_size=4096)
        self.ttl = ttl

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
        cached = self.get(source)
        if cached is not None:
            return cached
        chunks = list(chunk_fn(*args, **kwargs))
        self.set(source, chunks)
        return chunks
