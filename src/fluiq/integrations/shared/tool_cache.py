"""Automatic tool result learning from LLM conversation history.

Called pre-call in each provider patch. Scans the incoming messages/contents
for (tool_call → tool_result) pairs and populates the tool cache so that
future identical tool calls can be served from cache via
``fluiq.lookup_tool_result(name, args)``.
"""
from __future__ import annotations


def _try_populate(tool_name: str, args, result) -> None:
    try:
        from fluiq.optimization.client import populate_tool_cache
        populate_tool_cache(tool_name, args, result)
    except Exception:
        pass


def learn_from_openai_messages(messages) -> None:
    """Scan OpenAI-format messages for tool_call/tool_result pairs and cache."""
    if not messages:
        return
    # Build map: tool_call_id → (name, args_json_str)
    pending: dict[str, tuple[str, str]] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                    fn    = tc.get("function") or {}
                    name  = fn.get("name")
                    args  = fn.get("arguments", "{}")
                elif hasattr(tc, "id"):
                    tc_id = tc.id
                    name  = getattr(tc.function, "name", None)
                    args  = getattr(tc.function, "arguments", "{}")
                else:
                    continue
                if tc_id and name:
                    pending[tc_id] = (name, args)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            tc_id   = msg.get("tool_call_id")
            content = msg.get("content", "")
            if tc_id and tc_id in pending:
                name, args = pending[tc_id]
                _try_populate(name, args, content)


def learn_from_anthropic_messages(messages) -> None:
    """Scan Anthropic-format messages for tool_use/tool_result pairs and cache."""
    if not messages:
        return
    pending: dict[str, tuple[str, dict]] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            content = msg.get("content") or []
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tu_id  = block.get("id")
                    name   = block.get("name")
                    input_ = block.get("input") or {}
                    if tu_id and name:
                        pending[tu_id] = (name, input_)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            content = msg.get("content") or []
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tu_id  = block.get("tool_use_id")
                    result = block.get("content", "")
                    if tu_id and tu_id in pending:
                        name, args = pending[tu_id]
                        _try_populate(name, args, result)


def learn_from_gemini_contents(contents) -> None:
    """Scan Gemini-format contents for function_call/function_response pairs and cache."""
    if not contents:
        return
    # Gemini has no per-call id in function_response; match by name (last seen).
    pending: dict[str, dict] = {}
    for content in contents:
        if isinstance(content, dict):
            role  = content.get("role")
            parts = content.get("parts") or []
        else:
            role  = getattr(content, "role", None)
            parts = getattr(content, "parts", None) or []

        if role == "model":
            for part in parts:
                fc = part.get("function_call") if isinstance(part, dict) else getattr(part, "function_call", None)
                if fc is None:
                    continue
                name = fc.get("name") if isinstance(fc, dict) else getattr(fc, "name", None)
                args = fc.get("args", {}) if isinstance(fc, dict) else getattr(fc, "args", {})
                if name:
                    pending[name] = args or {}

        elif role in ("tool", "function", "user"):
            for part in parts:
                fr = part.get("function_response") if isinstance(part, dict) else getattr(part, "function_response", None)
                if fr is None:
                    continue
                name     = fr.get("name") if isinstance(fr, dict) else getattr(fr, "name", None)
                response = fr.get("response") if isinstance(fr, dict) else getattr(fr, "response", None)
                if name and name in pending:
                    _try_populate(name, pending[name], response)
