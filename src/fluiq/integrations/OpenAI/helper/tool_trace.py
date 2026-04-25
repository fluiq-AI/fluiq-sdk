import time
import threading
from fluiq.integrations.OpenAI.helper.utils import _to_jsonable

_pending_tool_calls: dict[str, tuple[float, str | None]] = {}
_pending_tool_calls_lock = threading.Lock()

def _record_dispatched_tool_calls(choices):
    if not choices:
        return
    now = time.time()
    with _pending_tool_calls_lock:
        for choice in choices:
            message = getattr(choice, "message", None)
            tool_calls = getattr(message, "tool_calls", None) if message else None
            for tc in tool_calls or []:
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                    name = (tc.get("function") or {}).get("name")
                else:
                    tc_id = getattr(tc, "id", None)
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", None) if fn else None
                if tc_id:
                    _pending_tool_calls[tc_id] = (now, name)


def _gc_pending_tool_calls(ttl=3600):
    cutoff = time.time() - ttl
    with _pending_tool_calls_lock:
        stale = [k for k, (ts, _) in _pending_tool_calls.items() if ts < cutoff]
        for k in stale:
            _pending_tool_calls.pop(k, None)


def _compute_tool_call_latencies(request_messages):
    if not request_messages:
        return None
    now = time.time()
    latencies = []
    with _pending_tool_calls_lock:
        for msg in request_messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                tc_id = msg.get("tool_call_id")
            else:
                role = getattr(msg, "role", None)
                tc_id = getattr(msg, "tool_call_id", None)
            if role != "tool" or not tc_id:
                continue
            entry = _pending_tool_calls.pop(tc_id, None)
            if not entry:
                continue
            dispatched_at, name = entry
            latencies.append({
                "tool_call_id": tc_id,
                "name": name,
                "latency": now - dispatched_at,
            })
    return latencies or None

def _extract_tool_calls(choices):
    calls = []
    for choice in choices or []:
        message = getattr(choice, "message", None)
        tool_calls = getattr(message, "tool_calls", None) if message else None
        if not tool_calls:
            continue
        for tc in tool_calls:
            calls.append(_to_jsonable(tc))
    
    return calls or None

def _finish_reasons(choices):
    reasons = [getattr(c, "finish_reason", None) for c in choices or []]
    return reasons or None