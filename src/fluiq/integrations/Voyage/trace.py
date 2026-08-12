import time
from fluiq.tracer import log_trace
from fluiq.integrations.shared.context import current_parent_id


def _serialize_embeddings(result) -> dict:
    """Convert a Voyage EmbeddingsObject to a JSON-serializable dict."""
    embeddings = getattr(result, "embeddings", []) or []
    return {
        "data": [
            {"values": list(emb) if not isinstance(emb, list) else emb, "index": i}
            for i, emb in enumerate(embeddings)
        ]
    }


def patch_voyage_embeddings():
    try:
        import voyageai
    except ImportError:
        return
    if not hasattr(voyageai, "Client"):
        return
    original = voyageai.Client.embed

    def wrapped(self, texts, model, **kwargs):
        cache_kwargs = {"model": model, "texts": texts}
        start = time.time()
        result = original(self, texts, model, **kwargs)
        end = time.time()
        serialized = _serialize_embeddings(result)
        log_trace({
            "type":        "llm",
            "integration": "voyage",
            "api":         "embeddings",
            "model":       model,
            "input":       texts,
            "response":    serialized,
            "latency":     end - start,
            "parent_id":   current_parent_id(),
            "tokens":      None,
        })
        return result

    voyageai.Client.embed = wrapped


def patch_voyage_embeddings_async():
    try:
        import voyageai
    except ImportError:
        return
    if not hasattr(voyageai, "AsyncClient"):
        return
    original = voyageai.AsyncClient.embed

    async def wrapped(self, texts, model, **kwargs):
        cache_kwargs = {"model": model, "texts": texts}
        start = time.time()
        result = await original(self, texts, model, **kwargs)
        end = time.time()
        serialized = _serialize_embeddings(result)
        log_trace({
            "type":        "llm",
            "integration": "voyage",
            "api":         "embeddings",
            "model":       model,
            "input":       texts,
            "response":    serialized,
            "latency":     end - start,
            "parent_id":   current_parent_id(),
            "tokens":      None,
        })
        return result

    voyageai.AsyncClient.embed = wrapped
