"""Reranker tracing glue.

Mirrors :mod:`fluiq.optimization.caching._tracing`: every wrapped reranker
emits one ``LogTrace`` span per ``rerank()`` call carrying the reranker
name, candidate counts, top-k, latency, and (when wired) score statistics.
The span flows through the same ingestion path as every other trace so it
shows up as a child of the surrounding LLM / agent / ``@trace`` span in the
dashboard trace tree.

The fields below are additive — ``LogTrace.model_config`` allows extras —
so we can surface new metrics (per-document scores, fusion mode, MMR
selection trail, etc.) without touching the schema.
"""
from typing import Any, List, Optional, Sequence


def emit_rerank_trace(
    *,
    trace_id: str,
    parent_id: Optional[str],
    reranker: str,
    query: Any,
    input_count: int,
    output_count: int,
    top_k: Optional[int],
    latency: float,
    started_at: float,
    success: bool,
    error: Optional[str] = None,
    scores: Optional[Sequence[float]] = None,
    indices: Optional[Sequence[int]] = None,
    extras: Optional[dict] = None,
) -> None:
    """Emit one rerank span. Best-effort; never raises."""
    try:
        from fluiq.tracer import log_trace

        # Truncate the query to keep payloads bounded — callers may pass long
        # synthesized queries (HyDE, multi-query). 2KB is plenty for the
        # dashboard hover preview.
        q = query
        if isinstance(q, str) and len(q) > 2048:
            q = q[:2048] + "\u2026"

        payload = {
            "trace_id": trace_id,
            "parent_id": parent_id,
            "type": "rerank",
            "function": f"rerank:{reranker}",
            "reranker": reranker,
            "query": q,
            "input_count": int(input_count),
            "output_count": int(output_count),
            "top_k": int(top_k) if top_k is not None else None,
            "latency": float(latency),
            "started_at": float(started_at),
            "success": bool(success),
            "status": "success" if success else "error",
        }
        if error is not None:
            payload["error"] = str(error)[:2048]
        if scores is not None:
            try:
                payload["scores"] = [float(s) for s in scores]
            except Exception:
                pass
        if indices is not None:
            try:
                payload["selected_indices"] = [int(i) for i in indices]
            except Exception:
                pass
        if extras:
            for k, v in extras.items():
                if k not in payload:
                    payload[k] = v
        log_trace(payload)
    except Exception:
        # Rerankers must stay transparent even if instrumentation breaks.
        pass


def summarize_scores(scores: Sequence[float]) -> dict:
    """Return ``{min, max, mean}`` for a score list. Empty -> empty dict."""
    if not scores:
        return {}
    try:
        vals: List[float] = [float(s) for s in scores]
    except (TypeError, ValueError):
        return {}
    if not vals:
        return {}
    n = len(vals)
    s = sum(vals)
    return {
        "score_min": min(vals),
        "score_max": max(vals),
        "score_mean": s / n,
    }
