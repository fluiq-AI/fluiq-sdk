import time
import uuid

from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace
from fluiq.integrations.shared.context import (
    current_parent_id,
    push_llm_trace_id,
    pop_llm_trace_id,
)


def emit_llm_start(integration, *, type_="llm", **fields):
    # Lightweight live-progress signal mirroring the LangChain / GoogleADK
    # pattern: same trace_id as the eventual completion emission so the
    # frontend can replace the running row in place. No latency / output /
    # tokens — those land on completion. Caller MUST pop the returned token
    # via `pop_llm_trace_id` once the trace completes (or errors).
    trace_id = str(uuid.uuid4())
    payload_kwargs = {
        "type": type_,
        "integration": integration,
        "trace_id": trace_id,
        "parent_id": current_parent_id(),
        "status": "running",
        "started_at": time.time(),
        **{k: v for k, v in fields.items() if v is not None},
    }
    try:
        log_trace(LogTrace(**payload_kwargs).model_dump(mode="json"))
    except Exception:
        pass
    token = push_llm_trace_id(trace_id)
    return trace_id, token


def end_llm_start(token):
    pop_llm_trace_id(token)
