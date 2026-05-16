"""Thin SDK client for the fluiq-api /secure and /secure/check endpoints.

call_secure()      — post-call: sends prompt + response, enriches trace dict.
pre_call_check()   — pre-call:  sends prompt only, raises FluiqSecurityError
                                 when mode='block' and the server blocks.

Both functions never raise on network errors — they log a warning and return
so the user's application is never interrupted by observability infrastructure.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from fluiq.config import _config

logger = logging.getLogger(__name__)


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content") or item.get("text") or ""
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return str(value)


def _base_url() -> str:
    return f"{_config['endpoint']}/{_config['version']}"


def call_secure(trace: dict[str, Any]) -> None:
    """POST trace text to /secure, enrich trace with results.  Never raises."""
    try:
        prompt   = _extract_text(trace.get("input")    or trace.get("messages"))
        response = _extract_text(trace.get("response") or trace.get("output"))

        # Collect tool outputs and context docs if present
        tool_outputs: list[str] = []
        for tc in (trace.get("tool_calls") or []):
            if isinstance(tc, dict):
                result = tc.get("result") or tc.get("output") or ""
                if result:
                    tool_outputs.append(str(result))

        r = requests.post(
            f"{_base_url()}/secure",
            json={
                "api_key":      _config["api_key"],
                "prompt":       prompt,
                "response":     response,
                "tool_outputs": tool_outputs,
            },
            timeout=3,
        )

        if r.status_code == 402:
            logger.warning("[fluiq.secure] %s", r.json().get("detail", "Plan upgrade required"))
            return

        r.raise_for_status()
        fields = r.json()

        # Redact originals when server flags HIGH risk
        if fields.get("should_block"):
            if "input" in trace and isinstance(trace["input"], str):
                trace["input"] = fields["prompt_redacted"]
            if "messages" in trace:
                trace["messages"] = None
            if "response" in trace and isinstance(trace["response"], str):
                trace["response"] = fields["response_redacted"]
            if "output" in trace and isinstance(trace["output"], str):
                trace["output"] = fields["response_redacted"]

        trace.update({
            "prompt_redacted":             fields.get("prompt_redacted",             ""),
            "response_redacted":           fields.get("response_redacted",           ""),
            "security_risk_level":         fields.get("security_risk_level",         "clean"),
            "security_risk_score":         fields.get("security_risk_score",         0.0),
            "pii_entities_prompt":         fields.get("pii_entities_prompt",         []),
            "pii_entities_response":       fields.get("pii_entities_response",       []),
            "injection_detected":          fields.get("injection_detected",          False),
            "injection_patterns":          fields.get("injection_patterns",          []),
            "jailbreak_detected":          fields.get("jailbreak_detected",          False),
            "jailbreak_patterns":          fields.get("jailbreak_patterns",          []),
            "skeleton_key_detected":       fields.get("skeleton_key_detected",       False),
            "skeleton_key_patterns":       fields.get("skeleton_key_patterns",       []),
            "secrets_detected":            fields.get("secrets_detected",            False),
            "secret_types":                fields.get("secret_types",                []),
            "indirect_injection_detected": fields.get("indirect_injection_detected", False),
            "indirect_injection_sources":  fields.get("indirect_injection_sources",  []),
            "semantic_attack_score":       fields.get("semantic_attack_score",       0.0),
        })

    except Exception as exc:
        logger.warning("[fluiq.secure] post-call scan failed: %s", repr(exc))


def pre_call_check(prompt_text: str) -> None:
    """Call /secure/check before the LLM call.

    When mode='block' and the server returns allow=False, raises
    FluiqSecurityError.  In all other cases (network error, 402, warn mode)
    returns silently so the LLM call proceeds normally.
    """
    mode = _config.get("secure_mode", "warn")
    try:
        r = requests.post(
            f"{_base_url()}/secure/check",
            json={
                "api_key": _config["api_key"],
                "prompt":  prompt_text,
            },
            timeout=2,
        )

        if r.status_code == 402:
            # Plan doesn't include secure — silently fall back to warn
            logger.warning("[fluiq.secure] %s", r.json().get("detail", "Plan upgrade required"))
            return

        r.raise_for_status()
        result = r.json()

        if not result.get("allow", True) and mode == "block":
            from fluiq.exceptions import FluiqSecurityError
            raise FluiqSecurityError(
                block_reason = result.get("block_reason", "Blocked by fluiq.secure"),
                risk_level   = result.get("risk_level", "high"),
                attack_types = result.get("attack_types", []),
            )

    except Exception as exc:
        # Re-raise FluiqSecurityError — everything else is a network/infra failure
        from fluiq.exceptions import FluiqSecurityError
        if isinstance(exc, FluiqSecurityError):
            raise
        logger.warning("[fluiq.secure] pre-call check failed: %s", repr(exc))
