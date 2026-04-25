import time
import threading
from fluiq.integrations.Anthropic.helper.utils import _to_jsonable

_pending_tool_calls: dict[str, tuple[float, str | None]] = {}
_pending_tool_calls_lock = threading.Lock()


def _extract_tool_use(content):
    if not isinstance(content, list):
        return None
    uses = []
    for block in content:
        if isinstance(block, dict):
            ptype = block.get("type")
        else:
            ptype = getattr(block,"type",None)

        if ptype == "tool_use":
            uses.append(_to_jsonable(block))
    return uses or None


def _record_dispatched_tool_calls(content):
    if not isinstance(content, list):
        return
    now = time.time()
    with _pending_tool_calls_lock:
        for block in content:
            if isinstance(block, dict):
                ptype = block.get("type")
                tc_id = block.get("id")
                name = block.get("name")
            else:
                ptype = getattr(block, "type", None)
                tc_id = getattr(block, "id", None)
                name = getattr(block, "name", None)
            if ptype == "tool_use" and tc_id:
                _pending_tool_calls[tc_id] = (now, name)


def _gc_pending_tool_calls(ttl=3600):
    cutoff = time.time() - ttl
    with _pending_tool_calls_lock:
        stale = [k for k, (ts, _) in _pending_tool_calls.items() if ts < cutoff]
        for k in stale:
            _pending_tool_calls.pop(k, None)


def _compute_tool_call_latencies(messages):
    if not messages:
        return None
    now = time.time()
    latencies = []
    with _pending_tool_calls_lock:
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", None)
            if role != "user" or not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    tc_id = block.get("tool_use_id")
                else:
                    btype = getattr(block, "type", None)
                    tc_id = getattr(block, "tool_use_id", None)
                if btype != "tool_result" or not tc_id:
                    continue
                entry = _pending_tool_calls.pop(tc_id, None)
                if not entry:
                    continue
                dispatched_at, name = entry
                latencies.append({
                    "tool_use_id": tc_id,
                    "name": name,
                    "latency": now - dispatched_at,
                })
    return latencies or None