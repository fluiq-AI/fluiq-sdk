import re
from typing import Callable, List, Optional, Sequence, Set, Tuple

from fluiq.optimization.rerankers.base import (
    BaseReranker,
    RerankResult,
    _coerce_documents,
)
from fluiq.optimization.rerankers.bm25 import BM25Reranker


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    if len(u) != len(v) or not u:
        return 0.0
    dot = nu = nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu ** 0.5 * nv ** 0.5)


class MMRReranker(BaseReranker):
    """Diversity-aware reranker using Maximal Marginal Relevance.

    Iteratively selects documents that maximize relevance to the query
    while penalising redundancy with the documents already picked::

        MMR_i = lambda * rel(D_i, Q)
              - (1 - lambda) * max_{D_j in S} sim(D_i, D_j)

    Use this on top of a relevance reranker (BM25, cross-encoder, hybrid)
    when the top-K is dominated by near-duplicates and you want broader
    coverage in the LLM context.

    Parameters:

    * ``relevance`` \u2014 a :class:`BaseReranker` that scores documents for
      the query. Defaults to :class:`BM25Reranker` (no extra deps); pass
      a cross-encoder or hybrid reranker for higher-quality relevance.
    * ``embed_fn`` \u2014 optional ``(List[str]) -> List[List[float]]`` used
      to measure document-to-document similarity via cosine. When omitted,
      similarity falls back to Jaccard token overlap so MMR works with no
      extra dependencies.
    * ``lambda_mult`` \u2014 relevance/diversity tradeoff in ``[0, 1]``.
      ``1.0`` is pure relevance (equivalent to the underlying reranker),
      ``0.0`` is pure diversity, ``0.5`` (default) balances both.
    """

    name = "mmr"

    def __init__(
        self,
        relevance: Optional[BaseReranker] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        lambda_mult: float = 0.5,
    ):
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                f"lambda_mult must be in [0, 1], got {lambda_mult}"
            )
        self.relevance = relevance or BM25Reranker()
        self.embed_fn = embed_fn
        self.lambda_mult = lambda_mult

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        docs = _coerce_documents(documents)
        if not docs:
            return RerankResult(documents=[])

        # Normalize underlying relevance scores so the lambda tradeoff is
        # meaningful regardless of the reranker's score scale (BM25 is
        # unbounded; cross-encoders are typically logits).
        rel = self.relevance.rerank(query, docs, top_k=None)
        rel_raw = {d.index: d.score for d in rel.documents}
        rel_scores = self._normalize(
            [rel_raw.get(i, 0.0) for i in range(len(docs))]
        )

        if self.embed_fn is not None:
            vecs = self.embed_fn(list(docs))

            def sim(i: int, j: int) -> float:
                return _cosine(vecs[i], vecs[j])
        else:
            toks = [_tokens(d) for d in docs]

            def sim(i: int, j: int) -> float:
                return _jaccard(toks[i], toks[j])

        n = len(docs)
        target = min(top_k if top_k is not None else n, n)
        selected: List[int] = []
        remaining = set(range(n))
        while remaining and len(selected) < target:
            best_idx = -1
            best_score = float("-inf")
            for i in remaining:
                redundancy = (
                    max(sim(i, j) for j in selected) if selected else 0.0
                )
                mmr = (
                    self.lambda_mult * rel_scores[i]
                    - (1.0 - self.lambda_mult) * redundancy
                )
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(best_idx)
            remaining.discard(best_idx)

        # Preserve MMR pick order in the output: the first pick is the most
        # relevant, later picks are progressively chosen for diversity.
        # _finalize sorts descending by score, so assign decreasing
        # synthetic scores keyed to pick rank.
        ordered: List[Tuple[int, float]] = [
            (idx, float(len(selected) - rank))
            for rank, idx in enumerate(selected)
        ]
        return self._finalize(ordered, docs, top_k)

    @staticmethod
    def _normalize(vals: List[float]) -> List[float]:
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return [0.0 for _ in vals]
        return [(v - lo) / (hi - lo) for v in vals]
