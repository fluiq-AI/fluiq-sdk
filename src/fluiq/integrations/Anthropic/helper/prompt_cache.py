"""Automatic cache_control injection for Anthropic prompt caching.

When fluiq.optimize() is active, this module injects
``cache_control: {"type": "ephemeral"}`` on the system prompt and the last
tool definition before each API call.  Anthropic silently ignores the
directive on blocks below the minimum cacheable size (~1024 tokens), so
injecting unconditionally is safe.

Copies are made for the modified parts to avoid mutating the caller's objects.
"""
from __future__ import annotations

from typing import Any


def maybe_inject_anthropic_cache_control(kwargs: dict[str, Any]) -> None:
    """Inject cache_control on system prompt and last tool block in-place on kwargs.

    No-op when fluiq.optimize() has not been called.
    """
    from fluiq.config import _config
    if not _config.get("optimize"):
        return

    _inject_system(kwargs)
    _inject_last_tool(kwargs)


def _inject_system(kwargs: dict[str, Any]) -> None:
    system = kwargs.get("system")
    if not system:
        return

    if isinstance(system, str):
        kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return

    if isinstance(system, list):
        # Copy list; patch the last text block that lacks cache_control.
        new_system = list(system)
        for i in range(len(new_system) - 1, -1, -1):
            block = new_system[i]
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and not block.get("cache_control"):
                new_system[i] = {**block, "cache_control": {"type": "ephemeral"}}
                break
        kwargs["system"] = new_system


def _inject_last_tool(kwargs: dict[str, Any]) -> None:
    tools = kwargs.get("tools")
    if not isinstance(tools, list) or not tools:
        return

    last = tools[-1]
    if not isinstance(last, dict) or last.get("cache_control"):
        return

    kwargs["tools"] = [*tools[:-1], {**last, "cache_control": {"type": "ephemeral"}}]
