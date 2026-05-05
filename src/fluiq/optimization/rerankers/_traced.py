"""Tracing wrapper for rerankers.

:class:`TracedReranker` adapts any :class:`BaseReranker` so each call to
``rerank()`` emits a span linked to the surrounding trace context:

* On entry it pushes a fresh ``trace_id`` onto the parent context. Any
  nested reranker (e.g. :class:`HybridReranker`'s keyword/semantic legs,
  :class:`MMRReranker`'s relevance leg) that is *also* wrapped will read
  the pushed id via :func:`current_parent_id` and emit its own span as a
  child, producing a real tree in the dashboard rather than a flat list.
* On exit it emits a ``rerank`` span with reranker name, input/output
  counts, top-k, latency, score min/max/mean, and the indices selected so
  the dashboard can render the operation in its trace tree.

The wrapper preserves :class:`BaseReranker`'s public surface (``rerank``,
``name``) so it is a drop-in for the original instance — including the
recursive wrap performed by :func:`auto_optimize` for hybrid / MMR.
"""
import time
import uuid
from typing import List, Optional, Sequence

from fluiq.integrations.shared.context import (
    current_parent_id,
    push_trace_id,
    pop_trace_id,
)
from fluiq.optimization.rerankers.base import (
    BaseReranker,
    RerankResult,
    _coerce_documents,
)
from fluiq.optimization.rerankers._tracing import (
    emit_rerank_trace,
    summarize_scores,
)


class TracedReranker(BaseReranker):
    """Wrap a reranker so every ``rerank()`` call emits a trace span."""

    def __init__(self, inner: BaseReranker):
        self._inner = inner
        # Mirror the wrapped reranker's identity so callers introspecting
        # ``.name`` (and downstream code branching on ``isinstance``) keep
        # working. We deliberately don't subclass dynamically — the wrapper
        # only needs to delegate the abstract ``rerank``.
        self.name = getattr(inner, "name", "reranker")

    @property
    def inner(self) -> BaseReranker:
        return self._inner

    def __getattr__(self, item):
        # Delegate any attribute the wrapper itself doesn't expose to the
        # underlying reranker so user code that reaches into reranker-specific
        # config (``r.alpha``, ``r.k1``, ``r.lambda_mult``, ...) keeps working.
        return getattr(self._inner, item)

    def rerank(
        self,
        query,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        docs = _coerce_documents(documents)
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        token = push_trace_id(trace_id)
        start = time.time()
        result: Optional[RerankResult] = None
        err: Optional[BaseException] = None
        try:
            result = self._inner.rerank(query, documents, top_k=top_k)
            return result
        except BaseException as exc:  # noqa: BLE001 — propagate after emit
            err = exc
            raise
        finally:
            try:
                pop_trace_id(token)
            except Exception:
                pass
            try:
                latency = time.time() - start
                if result is not None:
                    out_docs = result.documents
                    scores: List[float] = [d.score for d in out_docs]
                    indices: List[int] = [d.index for d in out_docs]
                    extras = summarize_scores(scores)
                else:
                    scores = []
                    indices = []
                    extras = {}
                emit_rerank_trace(
                    trace_id=trace_id,
                    parent_id=parent_id,
                    reranker=self.name,
                    query=query,
                    input_count=len(docs),
                    output_count=len(scores),
                    top_k=top_k,
                    latency=latency,
                    started_at=start,
                    success=err is None,
                    error=str(err) if err is not None else None,
                    scores=scores,
                    indices=indices,
                    extras=extras,
                )
            except Exception:
                pass


def apply_tracing(reranker: BaseReranker) -> BaseReranker:
    """Recursively wrap ``reranker`` and any composed sub-rerankers.

    Hybrid and MMR delegate to inner rerankers; wrapping them too is what
    makes the dashboard show a proper tree (Hybrid -> [BM25, CrossEncoder]
    or MMR -> [BM25]) instead of a single opaque box.
    """
    if reranker is None or isinstance(reranker, TracedReranker):
        return reranker

    # Deferred imports — these modules build TracedReranker via this helper
    # at construction time, so importing them here is cycle-free.
    from fluiq.optimization.rerankers.hybrid import HybridReranker
    from fluiq.optimization.rerankers.mmr import MMRReranker

    if isinstance(reranker, HybridReranker):
        if reranker.keyword is not None and not isinstance(
            reranker.keyword, TracedReranker
        ):
            reranker.keyword = TracedReranker(reranker.keyword)
        if reranker.semantic is not None and not isinstance(
            reranker.semantic, TracedReranker
        ):
            reranker.semantic = TracedReranker(reranker.semantic)
    elif isinstance(reranker, MMRReranker):
        if reranker.relevance is not None and not isinstance(
            reranker.relevance, TracedReranker
        ):
            reranker.relevance = TracedReranker(reranker.relevance)

    return TracedReranker(reranker)
