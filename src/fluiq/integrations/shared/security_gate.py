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
    """Extract user-controlled input for security scanning.

    System messages are developer-written and excluded to prevent false
    positives on legitimate assistant instructions.  Only user and tool
    messages (user-controlled content) are scanned pre-call.
    """
    # OpenAI chat completions — scan user / tool roles only
    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") not in ("user", "tool"):
                continue
            content = m.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return "\n".join(parts)

    # OpenAI responses API / Gemini contents list
    inp = kwargs.get("input") or kwargs.get("contents")
    if inp:
        if isinstance(inp, str):
            return inp
        if isinstance(inp, list):
            # Gemini: filter to user-role parts; fallback to raw join for strings
            user_parts: list[str] = []
            for item in inp:
                if isinstance(item, str):
                    user_parts.append(item)
                elif isinstance(item, dict):
                    if item.get("role") in ("user", None):
                        for p in (item.get("parts") or []):
                            if isinstance(p, str):
                                user_parts.append(p)
                            elif isinstance(p, dict) and p.get("text"):
                                user_parts.append(p["text"])
            return "\n".join(user_parts) if user_parts else "\n".join(str(i) for i in inp)

    # Anthropic — system is a separate kwarg; messages already filtered above
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
    from fluiq.integrations.shared.context import current_parent_id
    context = {k: kwargs.get(k) for k in ("model", "messages", "system", "tools") if kwargs.get(k) is not None}
    parent = current_parent_id()
    if parent:
        context["parent_id"] = parent
    guardrail = _config.get("secure_guardrail", "default")
    pre_call_check(prompt, context=context, guardrail=guardrail)
