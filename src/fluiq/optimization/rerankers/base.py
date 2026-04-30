from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict


class RerankedDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    document: str
    score: float


class RerankResult(BaseModel):
    """Reranked documents, sorted descending by score."""

    documents: List[RerankedDocument]

    @property
    def texts(self) -> List[str]:
        return [d.document for d in self.documents]

    @property
    def indices(self) -> List[int]:
        return [d.index for d in self.documents]


class BaseReranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
    ) -> RerankResult: ...

    def _finalize(
        self,
        scored: List[Tuple[int, float]],
        documents: Sequence[str],
        top_k: Optional[int],
    ) -> RerankResult:
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return RerankResult(
            documents=[
                RerankedDocument(index=i, document=documents[i], score=float(s))
                for i, s in scored
            ]
        )


def _coerce_documents(documents: Any) -> List[str]:
    if documents is None:
        return []
    if isinstance(documents, str):
        return [documents]
    if isinstance(documents, (list, tuple)):
        return [str(d) for d in documents if d is not None]
    return [str(documents)]
