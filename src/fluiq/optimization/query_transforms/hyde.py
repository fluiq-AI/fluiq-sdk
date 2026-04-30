from typing import Any, Callable, List


class HyDE:
    """Hypothetical Document Embeddings (Gao et al. 2022).

    Generates a plausible-but-fake answer with an LLM and returns it for
    use as the *retrieval query* (typically embedded). The fake answer
    sits closer in embedding space to real answer documents than the
    raw question does, which on factoid-heavy corpora reliably improves
    recall by 10\u201320%.

    Pass any ``llm_fn(prompt, **params) -> str``. To cache the rewriting
    calls themselves, pass a :class:`PromptCache` instance as ``llm_fn``
    \u2014 :class:`PromptCache` is callable with the same signature.

    Example::

        hyde = HyDE(opt.prompts)        # routes through PromptCache
        fake = hyde("What killed the dinosaurs?")
        vecs = opt.embed(fake)
        candidates = vector_store.search(vecs[0], k=20)
    """

    DEFAULT_PROMPT = (
        "Write a single concise paragraph that plausibly answers the "
        "question below. Speculate confidently and stay on topic; the "
        "answer does not need to be factually correct, only realistic. "
        "Do not refuse or hedge.\n\n"
        "Question: {query}\n"
        "Hypothetical answer:"
    )

    def __init__(
        self,
        llm_fn: Callable[..., Any],
        *,
        prompt: str = DEFAULT_PROMPT,
        n: int = 1,
        **llm_params: Any,
    ):
        if "{query}" not in prompt:
            raise ValueError("prompt must contain a {query} placeholder")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.llm_fn = llm_fn
        self.prompt = prompt
        self.n = n
        self.llm_params = llm_params

    def transform(self, query: str) -> List[str]:
        """Return ``n`` hypothetical answer documents for the query."""
        rendered = self.prompt.format(query=query)
        out: List[str] = []
        for _ in range(self.n):
            answer = self.llm_fn(rendered, **self.llm_params)
            out.append(answer if isinstance(answer, str) else str(answer))
        return out

    def __call__(self, query: str) -> List[str]:
        return self.transform(query)
