import math
import re
from collections import Counter
from typing import List, Optional, Sequence

from fluiq.optimization.rerankers.base import (
    BaseReranker,
    RerankResult,
    _coerce_documents,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English stopword list. Pass `stopwords=[...]` to override or `stopwords=()`
# to disable filtering entirely (useful for code search and non-English corpora).
_DEFAULT_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "of",
    "for", "by", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
    "they", "we", "what", "which", "who", "from", "not", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should", "may",
})


def _tokenize(text: str, stopwords: frozenset) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in stopwords]


class BM25Reranker(BaseReranker):
    """Keyword reranker using Okapi BM25.

    Pure-Python implementation with no external dependencies. Suitable for the
    short candidate lists typical of RAG retrieval (10-100 documents). For
    corpus-wide indexing, use a dedicated search engine instead.

    Parameters mirror the canonical BM25 formula:

    * ``k1`` controls term-frequency saturation (default 1.5)
    * ``b``  controls length normalization (default 0.75)
    """

    name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[Sequence[str]] = None,
    ):
        self.k1 = k1
        self.b = b
        self.stopwords = (
            frozenset(stopwords) if stopwords is not None else _DEFAULT_STOPWORDS
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        docs = _coerce_documents(documents)
        if not docs:
            return RerankResult(documents=[])

        tokenized = [_tokenize(d, self.stopwords) for d in docs]
        n_docs = len(tokenized)
        avg_dl = sum(len(t) for t in tokenized) / n_docs if n_docs else 0.0

        df: Counter = Counter()
        for toks in tokenized:
            for term in set(toks):
                df[term] += 1

        # BM25+ idf guard keeps the value non-negative for terms appearing in
        # most documents.
        idf = {
            term: max(math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0), 0.0)
            for term, n in df.items()
        }

        q_terms = _tokenize(query, self.stopwords)
        scored: List = []
        for i, toks in enumerate(tokenized):
            tf = Counter(toks)
            dl = len(toks) or 1
            score = 0.0
            for term in q_terms:
                ft = tf.get(term)
                if not ft:
                    continue
                num = ft * (self.k1 + 1)
                denom = ft + self.k1 * (1 - self.b + self.b * dl / (avg_dl or 1))
                score += idf.get(term, 0.0) * num / denom
            scored.append((i, score))

        return self._finalize(scored, docs, top_k)
