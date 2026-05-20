"""Pre-call optimization gate.

Called from each LLM integration's patched method *before* the actual API
call is dispatched.  Does nothing when ``fluiq.optimize()`` has not been
called or when optimize_mode is ``"observe"``.

When a cache hit is found it returns a lightweight provider-shaped mock
object so the patched wrapper can return early without hitting the LLM API.
The integration is also responsible for emitting a cache-hit trace via
``log_trace`` with ``_cache_hit=True`` so the dashboard shows the save.

Supported providers: ``"openai"``, ``"anthropic"``, ``"gemini"``.

The mock object carries ``_fluiq_payload`` — the raw cached dict — so the
integration's cache-hit trace block can emit all fields (tool_calls, mcp_calls,
etc.) without re-reading Redis.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from fluiq.config import _config


def _is_cache_active() -> bool:
    return (
        bool(_config.get("optimize"))
        and _config.get("optimize_mode", "cache") == "cache"
        and bool(_config.get("api_key"))
    )


# ---------------------------------------------------------------------------
# LLM gate
# ---------------------------------------------------------------------------

def pre_call_optimize(kwargs: dict[str, Any], provider: str) -> Optional[Any]:
    """Return a mock cached response, or ``None`` to proceed normally.

    No-ops immediately when:
    - ``fluiq.optimize()`` has not been called
    - ``optimize_mode`` is ``"observe"`` (trace-only, no cache serving)
    - no api_key is configured
    - cache miss
    """
    if not _is_cache_active():
        return None

    from fluiq.optimization.client import lookup_cache
    payload = lookup_cache(kwargs)
    if payload is None:
        return None

    from fluiq.integrations.shared.context import mark_inner_cache_hit
    mark_inner_cache_hit()
    return _build_mock(payload, provider, kwargs.get("model", ""))


# ---------------------------------------------------------------------------
# Embedding gate
# ---------------------------------------------------------------------------

def pre_call_optimize_embedding(kwargs: dict[str, Any], provider: str) -> Optional[Any]:
    """Return a mock cached embedding response, or ``None`` to proceed normally."""
    if not _is_cache_active():
        return None

    from fluiq.optimization.client import lookup_embedding_cache
    payload = lookup_embedding_cache(kwargs)
    if payload is None:
        return None

    from fluiq.integrations.shared.context import mark_inner_cache_hit
    mark_inner_cache_hit()
    return _build_embedding_mock(payload, provider, kwargs.get("model", ""))


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

def _make_tool_call_ns(tc: dict) -> Any:
    fn = tc.get("function") or {}
    return SimpleNamespace(
        id=tc.get("id"),
        type=tc.get("type", "function"),
        function=SimpleNamespace(
            name=fn.get("name"),
            arguments=fn.get("arguments", ""),
        ),
    )


def _build_mock(payload: dict, provider: str, model: str) -> Any:
    """Construct a minimal provider-shaped response object from a cached payload."""
    text = payload.get("response") or ""
    tool_calls = payload.get("tool_calls") or []
    tool_uses = payload.get("tool_uses") or []

    if provider == "openai":
        tc_ns = [_make_tool_call_ns(tc) for tc in tool_calls] or None
        msg = SimpleNamespace(
            content=text or None,
            role="assistant",
            tool_calls=tc_ns,
            refusal=None,
        )
        choice = SimpleNamespace(
            message=msg,
            finish_reason="tool_calls" if tc_ns else "stop",
            index=0,
            logprobs=None,
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            id="fluiq-cached",
            object="chat.completion",
            usage=None,
            _fluiq_cached=True,
            _fluiq_payload=payload,
        )

    if provider == "anthropic":
        content_blocks = []
        if text:
            content_blocks.append(SimpleNamespace(type="text", text=text))
        for tu in tool_uses:
            content_blocks.append(SimpleNamespace(
                type="tool_use",
                id=tu.get("id"),
                name=tu.get("name"),
                input=tu.get("input"),
            ))
        if not content_blocks:
            content_blocks = [SimpleNamespace(type="text", text="")]
        return SimpleNamespace(
            id="fluiq-cached",
            type="message",
            role="assistant",
            content=content_blocks,
            model=model,
            stop_reason="tool_use" if tool_uses else "end_turn",
            stop_sequence=None,
            usage=None,
            _fluiq_cached=True,
            _fluiq_payload=payload,
        )

    if provider == "openai_responses":
        text_parts = []
        if text:
            text_parts.append(SimpleNamespace(type="output_text", text=text, annotations=[]))
        message = SimpleNamespace(
            type="message",
            id="fluiq-cached",
            status="completed",
            role="assistant",
            content=text_parts or [SimpleNamespace(type="output_text", text="", annotations=[])],
        )
        return SimpleNamespace(
            id="fluiq-cached",
            object="response",
            model=model,
            output=[message],
            usage=None,
            _fluiq_cached=True,
            _fluiq_payload=payload,
        )

    if provider == "gemini":
        function_calls = payload.get("function_calls") or []
        parts = []
        if text:
            parts.append(SimpleNamespace(text=text, function_call=None))
        for fc in function_calls:
            parts.append(SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(
                    name=fc.get("name"),
                    args=fc.get("args"),
                    id=fc.get("id"),
                ),
            ))
        if not parts:
            parts = [SimpleNamespace(text="", function_call=None)]
        content = SimpleNamespace(parts=parts, role="model")
        candidate = SimpleNamespace(content=content, finish_reason=1, index=0, safety_ratings=[])
        return SimpleNamespace(
            candidates=[candidate],
            model=model,
            usage_metadata=None,
            _fluiq_cached=True,
            _fluiq_payload=payload,
        )

    return None


def _build_embedding_mock(payload: dict, provider: str, model: str) -> Any:
    """Construct a minimal provider-shaped embedding response from a cached payload."""
    embedding_response = payload.get("response")

    if provider == "openai":
        if isinstance(embedding_response, dict):
            data_ns = [
                SimpleNamespace(
                    embedding=item.get("embedding", []),
                    index=item.get("index", i),
                    object="embedding",
                )
                for i, item in enumerate(embedding_response.get("data", []))
            ]
        else:
            return None

        return SimpleNamespace(
            data=data_ns,
            model=model,
            object="list",
            usage=None,
            _fluiq_cached=True,
            _fluiq_payload=payload,
        )

    return None
