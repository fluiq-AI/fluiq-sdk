import re
from typing import Any, Callable, List


_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


class MultiQuery:
    """Generate multiple paraphrases of a query for fan-out retrieval.

    Each paraphrase preserves the original meaning but varies vocabulary
    and structure, which broadens the lexical surface area the vector
    store sees. Pair with Reciprocal Rank Fusion (already implemented in
    :class:`HybridReranker`) to merge the per-query candidate lists.

    Pass any ``llm_fn(prompt, **params) -> str``. As with
    :class:`HyDE`, you can pass a :class:`PromptCache` to cache
    rewrites so identical questions reuse the same paraphrases.

    Example::

        mq  = MultiQuery(opt.prompts, n=4)
        qs  = mq("How did the dinosaurs go extinct?")
        # qs == [original, variant1, variant2, ...]
    """

    DEFAULT_PROMPT = (
        "Generate {n} alternative phrasings of the question below. Each "
        "phrasing must preserve the original meaning but vary vocabulary "
        "and sentence structure. Output one phrasing per line, with no "
        "numbering, bullets, or extra commentary.\n\n"
        "Question: {query}\n"
        "Variations:"
    )

    def __init__(
        self,
        llm_fn: Callable[..., Any],
        *,
        prompt: str = DEFAULT_PROMPT,
        n: int = 4,
        include_original: bool = True,
        **llm_params: Any,
    ):
        if "{query}" not in prompt or "{n}" not in prompt:
            raise ValueError(
                "prompt must contain both {query} and {n} placeholders"
            )
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.llm_fn = llm_fn
        self.prompt = prompt
        self.n = n
        self.include_original = include_original
        self.llm_params = llm_params

    def transform(self, query: str) -> List[str]:
        """Return ``n`` paraphrases (plus the original if configured)."""
        rendered = self.prompt.format(query=query, n=self.n)
        raw = self.llm_fn(rendered, **self.llm_params)
        text = raw if isinstance(raw, str) else str(raw)

        variants: List[str] = []
        seen: set = set()
        for line in text.splitlines():
            cleaned = _BULLET_RE.sub("", line).strip().strip('"\'')
            if not cleaned or cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            variants.append(cleaned)
            if len(variants) >= self.n:
                break

        out: List[str] = list(variants)
        if self.include_original:
            if query.lower() not in {v.lower() for v in out}:
                out.insert(0, query)
        return out

    def __call__(self, query: str) -> List[str]:
        return self.transform(query)
