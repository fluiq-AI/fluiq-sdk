from typing import Dict, List, Optional, Sequence, Tuple

from fluiq.optimization.rerankers.base import (
    BaseReranker,
    RerankResult,
    _coerce_documents,
)
from fluiq.optimization.rerankers.bm25 import BM25Reranker
from fluiq.optimization.rerankers.cross_encoder import CrossEncoderReranker


class HybridReranker(BaseReranker):
    """Hybrid reranker fusing keyword (BM25) and semantic (cross-encoder) scores.

    Two fusion modes:

    * ``rrf`` (default) — Reciprocal Rank Fusion. Combines rankers by rank
      position, ignoring raw score scales. Robust default and the academic
      standard for heterogeneous rankers.
    * ``weighted`` — min-max normalize each ranker's scores into ``[0, 1]``,
      then take ``alpha * semantic + (1 - alpha) * keyword``. Use when you
      want a tunable knob between lexical and semantic matching.

    ``alpha`` controls the semantic weight in both modes (``0`` = keyword
    only, ``1`` = semantic only, ``0.5`` = balanced).
    """

    name = "hybrid"

    def __init__(
        self,
        keyword: Optional[BaseReranker] = None,
        semantic: Optional[BaseReranker] = None,
        fusion: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        if fusion not in ("rrf", "weighted"):
            raise ValueError(
                f"fusion must be 'rrf' or 'weighted', got {fusion!r}"
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.keyword = keyword or BM25Reranker()
        self.semantic = semantic or CrossEncoderReranker()
        self.fusion = fusion
        self.alpha = alpha
        self.rrf_k = rrf_k

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        docs = _coerce_documents(documents)
        if not docs:
            return RerankResult(documents=[])
        kw = self.keyword.rerank(query, docs, top_k=None)
        sm = self.semantic.rerank(query, docs, top_k=None)
        if self.fusion == "rrf":
            scored = self._rrf(kw, sm, len(docs))
        else:
            scored = self._weighted(kw, sm, len(docs))
        return self._finalize(scored, docs, top_k)

    def _rrf(
        self, kw: RerankResult, sm: RerankResult, n: int
    ) -> List[Tuple[int, float]]:
        kw_rank = {d.index: r for r, d in enumerate(kw.documents)}
        sm_rank = {d.index: r for r, d in enumerate(sm.documents)}
        scored: List[Tuple[int, float]] = []
        for i in range(n):
            score = 0.0
            if i in kw_rank:
                score += (1.0 - self.alpha) / (self.rrf_k + kw_rank[i] + 1)
            if i in sm_rank:
                score += self.alpha / (self.rrf_k + sm_rank[i] + 1)
            scored.append((i, score))
        return scored

    def _weighted(
        self, kw: RerankResult, sm: RerankResult, n: int
    ) -> List[Tuple[int, float]]:
        kw_norm = self._normalize({d.index: d.score for d in kw.documents}, n)
        sm_norm = self._normalize({d.index: d.score for d in sm.documents}, n)
        return [
            (i, (1.0 - self.alpha) * kw_norm[i] + self.alpha * sm_norm[i])
            for i in range(n)
        ]

    @staticmethod
    def _normalize(scores: Dict[int, float], n: int) -> List[float]:
        vals = [scores.get(i, 0.0) for i in range(n)]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return [0.0 for _ in vals]
        return [(v - lo) / (hi - lo) for v in vals]
