from typing import Optional, Sequence

from fluiq.optimization.rerankers.base import (
    BaseReranker,
    RerankResult,
    _coerce_documents,
)


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Semantic reranker using a cross-encoder model.

    Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2`` from
    sentence-transformers — a 22M-parameter model trained on MS MARCO that
    runs comfortably on CPU and outperforms bi-encoder cosine similarity for
    relevance ranking on most RAG workloads.

    Requires the optional ``sentence-transformers`` package::

        pip install sentence-transformers

    The model is downloaded from the Hugging Face Hub on first use and cached
    locally; subsequent ``rerank`` calls are offline.
    """

    name = "cross-encoder"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        self.model_name = model
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "CrossEncoderReranker requires the `sentence-transformers` "
                "package. Install it with: pip install sentence-transformers"
            ) from exc
        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        docs = _coerce_documents(documents)
        if not docs:
            return RerankResult(documents=[])
        model = self._load()
        pairs = [(query, d) for d in docs]
        raw_scores = model.predict(pairs, batch_size=self.batch_size)
        scored = [(i, float(s)) for i, s in enumerate(raw_scores)]
        return self._finalize(scored, docs, top_k)
