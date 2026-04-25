import time
import threading
from collections import deque
from fluiq.integrations.Gemini.helper.utils import _to_jsonable

_pending_tool_calls: dict[str, tuple[float, str | None]] = {}
_pending_tool_calls_by_name: dict[str, deque] = {}
_pending_tool_calls_lock = threading.Lock()


def _extract_function_calls(candidates):
    calls = []
    for cand in candidates or []:
        if isinstance(cand, dict):
            content = cand.get("content")
            parts = (content or {}).get("parts") if content else None
        else:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
        for part in parts or []:
            fc = (
                part.get("function_call") if isinstance(part, dict)
                else getattr(part, "function_call", None)
            )
            if fc:
                calls.append(_to_jsonable(fc))
    return calls or None


def _finish_reasons(candidates):
    reasons = []
    for cand in candidates or []:
        r = cand.get("finish_reason") if isinstance(cand, dict) \
            else getattr(cand, "finish_reason", None)
        reasons.append(str(r) if r is not None else None)
    return reasons or None


def _extract_request_tools(kwargs, instance=None):
    """Tools/tool_config can live at top-level kwargs (vertexai, older genai)
    or inside kwargs['config'] (google-genai v1 pattern), or on the model
    instance itself (vertexai GenerativeModel(tools=...))."""
    tools = kwargs.get("tools")
    tool_config = kwargs.get("tool_config")

    config = kwargs.get("config")
    if config is not None:
        if isinstance(config, dict):
            tools = tools or config.get("tools")
            tool_config = tool_config or config.get("tool_config")
        else:
            tools = tools or getattr(config, "tools", None)
            tool_config = tool_config or getattr(config, "tool_config", None)

    if instance is not None and not tools:
        tools = getattr(instance, "_tools", None) or getattr(instance, "tools", None)

    return _to_jsonable(tools), _to_jsonable(tool_config)


def _fc_id_and_name(fc):
    if isinstance(fc, dict):
        return fc.get("id"), fc.get("name")
    return getattr(fc, "id", None), getattr(fc, "name", None)


def _record_dispatched_tool_calls(candidates):
    if not candidates:
        return
    now = time.time()
    with _pending_tool_calls_lock:
        for cand in candidates:
            if isinstance(cand, dict):
                content = cand.get("content")
                parts = (content or {}).get("parts") if content else None
            else:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
            for part in parts or []:
                fc = (
                    part.get("function_call") if isinstance(part, dict)
                    else getattr(part, "function_call", None)
                )
                if not fc:
                    continue
                fc_id, name = _fc_id_and_name(fc)
                if fc_id:
                    _pending_tool_calls[fc_id] = (now, name)
                elif name:
                    _pending_tool_calls_by_name.setdefault(name, deque()).append(now)


def _gc_pending_tool_calls(ttl=3600):
    cutoff = time.time() - ttl
    with _pending_tool_calls_lock:
        stale = [k for k, (ts, _) in _pending_tool_calls.items() if ts < cutoff]
        for k in stale:
            _pending_tool_calls.pop(k, None)
        empty_names = []
        for name, dq in _pending_tool_calls_by_name.items():
            while dq and dq[0] < cutoff:
                dq.popleft()
            if not dq:
                empty_names.append(name)
        for name in empty_names:
            _pending_tool_calls_by_name.pop(name, None)


def _iter_contents(contents):
    if contents is None or isinstance(contents, str):
        return
    if isinstance(contents, list):
        for item in contents:
            yield item
    else:
        yield contents


def _compute_tool_call_latencies(contents):
    if contents is None:
        return None
    now = time.time()
    latencies = []
    with _pending_tool_calls_lock:
        for item in _iter_contents(contents):
            if isinstance(item, dict):
                parts = item.get("parts")
            else:
                parts = getattr(item, "parts", None)
            for part in parts or []:
                fr = (
                    part.get("function_response") if isinstance(part, dict)
                    else getattr(part, "function_response", None)
                )
                if not fr:
                    continue
                fr_id, name = _fc_id_and_name(fr)
                dispatched_at = None
                if fr_id and fr_id in _pending_tool_calls:
                    dispatched_at, rec_name = _pending_tool_calls.pop(fr_id)
                    name = name or rec_name
                elif name:
                    dq = _pending_tool_calls_by_name.get(name)
                    if dq:
                        dispatched_at = dq.popleft()
                        if not dq:
                            _pending_tool_calls_by_name.pop(name, None)
                if dispatched_at is None:
                    continue
                latencies.append({
                    "id": fr_id,
                    "name": name,
                    "latency": now - dispatched_at,
                })
    return latencies or None