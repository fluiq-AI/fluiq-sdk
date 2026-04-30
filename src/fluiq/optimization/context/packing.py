"""Context-window helpers: packing and compression.

These tools sit between the reranker and the LLM. After reranking returns
a list of relevance-sorted chunks, ``pack_context`` greedily fits them
into a token budget and optionally reorders to mitigate the
"lost-in-the-middle" attention dropoff. ``compress_context`` drops
sentences with no query-term overlap so each chunk costs fewer tokens.
"""
import re
from typing import Any, Callable, List, Optional, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens_default(text: str) -> int:
    """Rough char-based token estimator (~4 chars/token).

    Pure-Python fallback for users who don't have ``tiktoken`` installed.
    Pass a real tokenizer via the ``count_tokens`` argument when accuracy
    matters (e.g. ``count_tokens=lambda s: len(enc.encode(s))``).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def lost_in_middle_reorder(items: Sequence[Any]) -> List[Any]:
    """Reorder a relevance-sorted sequence to place strongest items at the
    boundaries.

    Empirically, LLMs attend less to the middle of long contexts (Liu et
    al. 2023). Given items sorted by descending relevance, this returns
    an interleaving that places rank 0 first, rank 1 last, rank 2 second,
    rank 3 second-to-last, etc., so the strongest evidence sits where the
    model attends most.
    """
    items = list(items)
    n = len(items)
    if n <= 2:
        return items
    head: List[Any] = []
    tail: List[Any] = []
    for i, item in enumerate(items):
        if i % 2 == 0:
            head.append(item)
        else:
            tail.append(item)
    return head + list(reversed(tail))


def pack_context(
    chunks: Sequence[str],
    max_tokens: int,
    *,
    separator: str = "\n\n",
    reorder: Optional[str] = None,
    count_tokens: Callable[[str], int] = count_tokens_default,
) -> str:
    """Greedy-pack relevance-ranked chunks into a token budget.

    Walks ``chunks`` in order, including each chunk whose token cost
    still fits within ``max_tokens`` (separator overhead is accounted for
    once per included chunk after the first). Chunks that overflow the
    remaining budget are skipped, but later shorter chunks may still fit
    \u2014 this is the "best-effort knapsack" heuristic, not optimal.

    Parameters:

    * ``reorder`` \u2014 ``None`` (default) keeps the input order;
      ``"lost-in-middle"`` reorders included chunks so the highest-rank
      ones sit at the start and end of the packed string.
    * ``count_tokens`` \u2014 token-counting callable; defaults to a rough
      char-based heuristic. Pass ``tiktoken.encoding_for_model(...).encode``
      (wrapped in ``len``) for production accuracy.
    """
    if max_tokens <= 0 or not chunks:
        return ""
    if reorder not in (None, "lost-in-middle"):
        raise ValueError(
            f"reorder must be None or 'lost-in-middle', got {reorder!r}"
        )

    sep_cost = count_tokens(separator) if separator else 0
    used = 0
    included: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        cost = count_tokens(chunk) + (sep_cost if included else 0)
        if used + cost > max_tokens:
            continue
        included.append(chunk)
        used += cost

    if reorder == "lost-in-middle":
        included = lost_in_middle_reorder(included)
    return separator.join(included)


def compress_context(
    chunks: Sequence[str],
    query: str,
    *,
    threshold: float = 0.0,
    scorer: Optional[Callable[[str, str], float]] = None,
) -> List[str]:
    """Drop sentences from each chunk that don't carry the query.

    Splits each chunk on sentence boundaries and keeps only sentences
    whose ``scorer(sentence, query)`` strictly exceeds ``threshold``. The
    default scorer is Jaccard token overlap, so a sentence with at least
    one query token survives the default ``threshold=0.0``. Chunks that
    keep no sentences are dropped entirely.

    Pass a custom ``scorer`` (e.g. embedding cosine, an LLM-based
    relevance score, or LLMLingua) for stronger compression. Returning a
    list of chunks (not a single string) keeps the output compatible with
    :func:`pack_context` downstream.
    """
    if scorer is None:
        scorer = _jaccard_scorer
    out: List[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        kept = [
            s for s in _split_sentences(chunk)
            if scorer(s, query) > threshold
        ]
        if kept:
            out.append(" ".join(kept))
    return out


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


def _jaccard_scorer(sentence: str, query: str) -> float:
    a = set(_TOKEN_RE.findall(sentence.lower()))
    b = set(_TOKEN_RE.findall(query.lower()))
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0
