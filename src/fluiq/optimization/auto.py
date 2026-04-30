"""One-shot wiring for everything in ``fluiq.optimization``.

``auto_optimize()`` returns an :class:`OptimizedRAG` bundle holding a
shared cache backend, the three specialized caches, and a configured
reranker. The user supplies their ``embed_fn`` / ``llm_fn`` and gets
ready-to-use callables back; everything else falls back to sensible
defaults so a partial call (``auto_optimize(rerank="bm25")``) still
produces a usable bundle.
"""
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from fluiq.optimization.caching import (
    BaseCache,
    DiskCache,
    DocumentCache,
    EmbeddingCache,
    InMemoryCache,
    PromptCache,
    ToolCache,
)
from fluiq.optimization.context import (
    compress_context,
    count_tokens_default,
    pack_context,
)
from fluiq.optimization.query_transforms import HyDE, MultiQuery
from fluiq.optimization.rerankers import (
    BaseReranker,
    BM25Reranker,
    CrossEncoderReranker,
    HybridReranker,
    MMRReranker,
    RerankResult,
)


def _instrumentation_active() -> bool:
    """Return ``True`` iff ``fluiq.instrument()`` has set an API key."""
    try:
        from fluiq.config import _config
        return bool(_config.get("api_key")) and bool(_config.get("enabled", True))
    except Exception:
        return False


def _sentence_transformers_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


@dataclass
class OptimizedRAG:
    """Bundle returned by :func:`auto_optimize`.

    Holds the shared cache backend plus each specialized cache and the
    configured reranker. Convenience methods (:meth:`embed`, :meth:`ask`,
    :meth:`rerank`, :meth:`chunk`) delegate to the underlying components
    so calling code stays short. Components that weren't requested
    (``embed_fn=None``, ``rerank=None``) are left as ``None`` and the
    matching convenience method raises a clear error.
    """

    backend: BaseCache
    reranker: Optional[BaseReranker] = None
    embeddings: Optional[EmbeddingCache] = None
    prompts: Optional[PromptCache] = None
    documents: DocumentCache = field(default_factory=DocumentCache)
    tools: Optional[ToolCache] = None

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self.embeddings is None:
            raise RuntimeError(
                "auto_optimize() was called without embed_fn; cannot embed. "
                "Pass embed_fn=... to enable embedding caching."
            )
        return self.embeddings(texts)

    def ask(self, prompt: str, **params: Any) -> Any:
        if self.prompts is None:
            raise RuntimeError(
                "auto_optimize() was called without llm_fn; cannot ask. "
                "Pass llm_fn=... to enable prompt caching."
            )
        return self.prompts(prompt, **params)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        if self.reranker is None:
            raise RuntimeError(
                "auto_optimize() was called with rerank=None; no reranker."
            )
        return self.reranker.rerank(query, documents, top_k=top_k)

    def chunk(
        self,
        source: str,
        chunk_fn: Callable[..., List[str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[str]:
        return self.documents.get_or_chunk(source, chunk_fn, *args, **kwargs)

    def pack(
        self,
        chunks: Sequence[str],
        max_tokens: int,
        *,
        separator: str = "\n\n",
        reorder: Optional[str] = "lost-in-middle",
        count_tokens: Callable[[str], int] = count_tokens_default,
    ) -> str:
        return pack_context(
            chunks,
            max_tokens,
            separator=separator,
            reorder=reorder,
            count_tokens=count_tokens,
        )

    def compress(
        self,
        chunks: Sequence[str],
        query: str,
        *,
        threshold: float = 0.0,
        scorer: Optional[Callable[[str, str], float]] = None,
    ) -> List[str]:
        return compress_context(
            chunks, query, threshold=threshold, scorer=scorer
        )

    def hyde(self, query: str, *, n: int = 1, **llm_params: Any) -> List[str]:
        """Generate ``n`` HyDE pseudo-answers for ``query``.

        Routes through :attr:`prompts` when a PromptCache is configured so
        identical questions reuse cached rewrites.
        """
        llm = self._llm_callable("hyde")
        return HyDE(llm, n=n, **llm_params).transform(query)

    def multi_query(
        self,
        query: str,
        *,
        n: int = 4,
        include_original: bool = True,
        **llm_params: Any,
    ) -> List[str]:
        """Generate ``n`` paraphrases of ``query`` (plus the original)."""
        llm = self._llm_callable("multi_query")
        return MultiQuery(
            llm, n=n, include_original=include_original, **llm_params,
        ).transform(query)

    def _llm_callable(self, who: str) -> Callable[..., Any]:
        if self.prompts is not None:
            return self.prompts
        raise RuntimeError(
            f"auto_optimize() was called without llm_fn; cannot run {who}. "
            "Pass llm_fn=... to enable query transforms."
        )

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a deterministic tool for cached invocation."""
        if self.tools is None:
            raise RuntimeError("ToolCache missing on this bundle.")
        self.tools.register(name, fn)

    def tool(self, name: str, **kwargs: Any) -> Any:
        """Invoke a registered tool, returning the cached result on a hit."""
        if self.tools is None:
            raise RuntimeError("ToolCache missing on this bundle.")
        return self.tools.call(name, **kwargs)


def auto_optimize(
    *,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    llm_fn: Optional[Callable[..., Any]] = None,
    embed_model: str = "embedding-model",
    llm_model: str = "llm-model",
    backend: Optional[BaseCache] = None,
    cache_dir: Optional[str] = None,
    rerank: Optional[str] = "hybrid",
    fusion: str = "rrf",
    alpha: float = 0.5,
    mmr_lambda: float = 0.5,
    tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ttl: Optional[float] = None,
    trace: Union[bool, str] = "auto",
) -> OptimizedRAG:
    """Configure every optimization the SDK ships in one call.

    Returns an :class:`OptimizedRAG` bundle wiring:

    * a shared cache backend \u2014 :class:`DiskCache` when ``cache_dir`` is
      set, otherwise an LRU :class:`InMemoryCache` of 10k entries;
    * an :class:`EmbeddingCache` if ``embed_fn`` is provided;
    * a :class:`PromptCache` if ``llm_fn`` is provided;
    * a :class:`DocumentCache` for chunked-document reuse;
    * a :class:`ToolCache` whenever the bundle is built (empty by default;
      register tools via ``opt.register_tool()`` or pass ``tools={...}``);
    * a reranker \u2014 ``"hybrid"`` (default), ``"bm25"``,
      ``"cross-encoder"``, ``"mmr"``, or ``None`` to skip. ``"hybrid"``
      falls back to BM25 (with a warning) when ``sentence-transformers``
      isn't installed, so ``auto_optimize()`` always returns a working
      bundle. ``"mmr"`` wraps BM25 with diversity-aware re-selection;
      pass ``embed_fn`` to upgrade the doc-doc similarity to cosine.

    ``trace`` defaults to ``"auto"`` \u2014 cache spans are emitted iff
    ``fluiq.instrument()`` has been called. Pass ``True`` / ``False`` to
    force on/off.
    """
    if trace == "auto":
        trace = _instrumentation_active()
    elif not isinstance(trace, bool):
        raise ValueError(f"trace must be bool or 'auto', got {trace!r}")

    if backend is None:
        backend = (
            DiskCache(cache_dir) if cache_dir is not None
            else InMemoryCache(max_size=10_000)
        )

    embeddings_cache: Optional[EmbeddingCache] = None
    if embed_fn is not None:
        embeddings_cache = EmbeddingCache(
            embed_fn=embed_fn,
            model=embed_model,
            backend=backend,
            ttl=ttl,
            trace=trace,
        )

    prompts_cache: Optional[PromptCache] = None
    if llm_fn is not None:
        prompts_cache = PromptCache(
            llm_fn=llm_fn,
            model=llm_model,
            backend=backend,
            ttl=ttl,
            trace=trace,
        )

    documents_cache = DocumentCache(backend=backend, ttl=ttl)

    tools_cache = ToolCache(
        tools=tools,
        backend=backend,
        ttl=ttl,
        trace=trace,
    )

    reranker_obj: Optional[BaseReranker]
    if rerank is None:
        reranker_obj = None
    elif rerank == "hybrid":
        # Hybrid needs a cross-encoder, which is an optional dep. Falling
        # back to BM25 keeps auto_optimize() always-usable in environments
        # that haven't installed ``fluiq[rerank]``.
        if _sentence_transformers_available():
            reranker_obj = HybridReranker(fusion=fusion, alpha=alpha)
        else:
            warnings.warn(
                "auto_optimize(rerank='hybrid') requires sentence-transformers; "
                "falling back to BM25. Install with `pip install fluiq[rerank]` "
                "to enable the semantic half of the hybrid reranker.",
                RuntimeWarning,
                stacklevel=2,
            )
            reranker_obj = BM25Reranker()
    elif rerank == "bm25":
        reranker_obj = BM25Reranker()
    elif rerank in ("cross-encoder", "cross_encoder"):
        reranker_obj = CrossEncoderReranker()
    elif rerank == "mmr":
        reranker_obj = MMRReranker(
            embed_fn=embed_fn,
            lambda_mult=mmr_lambda,
        )
    else:
        raise ValueError(
            f"Unknown rerank={rerank!r}. Use 'hybrid', 'bm25', "
            f"'cross-encoder', 'mmr', or None."
        )

    return OptimizedRAG(
        backend=backend,
        reranker=reranker_obj,
        embeddings=embeddings_cache,
        prompts=prompts_cache,
        documents=documents_cache,
        tools=tools_cache,
    )
