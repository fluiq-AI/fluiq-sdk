"""Cache tracing glue.

When a cache wrapper is constructed with ``trace=True`` it emits a single
``LogTrace`` event per call describing how many lookups hit and missed.
The event flows through the same ingestion path as every other span, so
hit-rate analytics fall out of standard trace aggregation queries.

The fields below are intentionally additive — ``LogTrace`` allows extra
keys via ``ConfigDict(extra="allow")`` so the schema does not need to
change to surface a new metric.
"""
import time
import uuid
from typing import Optional


def emit_cache_trace(
    *,
    kind: str,
    model: str,
    hits: int,
    misses: int,
    latency: float,
    function: Optional[str] = None,
) -> None:
    """Emit one cache span. Best-effort; never raises."""
    try:
        from fluiq.tracer import log_trace
        from fluiq.integrations.shared.context import current_parent_id

        size = hits + misses
        rate = (hits / size) if size else 0.0
        payload = {
            "trace_id": str(uuid.uuid4()),
            "parent_id": current_parent_id(),
            "type": "cache",
            "cache_kind": kind,
            "cache_hits": int(hits),
            "cache_misses": int(misses),
            "cache_size": int(size),
            "cache_hit_rate": round(rate, 4),
            "cache_hit": misses == 0 and size > 0,
            "model": model,
            "function": function or f"{kind}_cache",
            "latency": float(latency),
            "timestamp": time.time(),
            "success": True,
        }
        log_trace(payload)
    except Exception:
        # Caches must stay transparent even if the SDK isn't instrumented.
        pass
