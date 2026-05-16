"""Pre-call optimization gate.

Called from each LLM integration's patched method *before* the actual API
call is dispatched.  Does nothing when ``fluiq.optimize()`` has not been
called or when optimize_mode is ``"observe"``.

When a cache hit is found it returns a lightweight provider-shaped mock
object so the patched wrapper can return early without hitting the LLM API.
The integration is also responsible for emitting a cache-hit trace via
``log_trace`` with ``_cache_hit=True`` so the dashboard shows the save.

Supported providers: ``"openai"``, ``"anthropic"``, ``"gemini"``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from fluiq.config import _config


def pre_call_optimize(kwargs: dict[str, Any], provider: str) -> Optional[Any]:
    """Return a mock cached response, or ``None`` to proceed normally.

    No-ops immediately when:
    - ``fluiq.optimize()`` has not been called
    - ``optimize_mode`` is ``"observe"`` (trace-only, no cache serving)
    - no api_key is configured
    - cache miss
    """
    if not _config.get("optimize"):
        return None
    if _config.get("optimize_mode", "cache") != "cache":
        return None
    if not _config.get("api_key"):
        return None

    from fluiq.optimization.client import lookup_cache
    cached_text = lookup_cache(kwargs)
    if cached_text is None:
        return None

    return _build_mock(cached_text, provider, kwargs.get("model", ""))


def _build_mock(text: str, provider: str, model: str) -> Any:
    """Construct a minimal provider-shaped response object from cached text."""
    if provider == "openai":
        msg = SimpleNamespace(content=text, role="assistant", tool_calls=None, refusal=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop", index=0, logprobs=None)
        return SimpleNamespace(
            choices=[choice],
            model=model,
            id="fluiq-cached",
            object="chat.completion",
            usage=None,
            _fluiq_cached=True,
        )

    if provider == "anthropic":
        block = SimpleNamespace(type="text", text=text)
        return SimpleNamespace(
            id="fluiq-cached",
            type="message",
            role="assistant",
            content=[block],
            model=model,
            stop_reason="end_turn",
            stop_sequence=None,
            usage=None,
            _fluiq_cached=True,
        )

    if provider == "gemini":
        part = SimpleNamespace(text=text)
        content = SimpleNamespace(parts=[part], role="model")
        candidate = SimpleNamespace(content=content, finish_reason=1, index=0, safety_ratings=[])
        return SimpleNamespace(
            candidates=[candidate],
            model=model,
            usage_metadata=None,
            _fluiq_cached=True,
        )

    return None