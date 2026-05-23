"""Pre-call security gate.

Called from each LLM integration's patched method *before* the actual API
call is dispatched.  Does nothing when fluiq.secure() has not been called or
when secure_mode is 'warn'.  When secure_mode is 'block' it calls
/secure/check and raises FluiqSecurityError if the server returns allow=False.

The function accepts the raw kwargs dict from the LLM call so it can extract
the prompt text regardless of provider-specific argument shapes.
"""
from __future__ import annotations

from typing import Any

from fluiq.config import _config


def _extract_prompt(kwargs: dict[str, Any]) -> str:
    """Best-effort prompt extraction across OpenAI / Anthropic / Gemini shapes."""
    # OpenAI chat completions
    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return "\n".join(parts)

    # OpenAI responses API / Gemini
    inp = kwargs.get("input") or kwargs.get("contents")
    if inp:
        if isinstance(inp, str):
            return inp
        if isinstance(inp, list):
            return "\n".join(str(i) for i in inp)

    # Anthropic
    prompt = kwargs.get("prompt")
    if isinstance(prompt, str):
        return prompt

    return ""


def pre_call_guard(kwargs: dict[str, Any]) -> None:
    """Run the pre-call security check.  Raises FluiqSecurityError if blocked.

    No-ops immediately when:
    - fluiq.secure() has not been called
    - secure_mode is 'warn' (post-call only)
    - no api_key is configured
    """
    if not _config.get("secure"):
        return
    if _config.get("secure_mode", "warn") != "block":
        return
    if not _config.get("api_key"):
        return

    prompt = _extract_prompt(kwargs)
    if not prompt.strip():
        return

    from fluiq.security.client import pre_call_check
    context = {k: kwargs.get(k) for k in ("model", "messages", "system", "tools") if kwargs.get(k) is not None}
    pre_call_check(prompt, context=context)
