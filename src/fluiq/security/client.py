"""Thin SDK client for the fluiq-api /secure/check endpoint.

pre_call_check() — pre-call synchronous guard: sends prompt only, raises
                   FluiqSecurityError when mode='block' and the server blocks.

Post-call security scanning is handled asynchronously by the evaluator worker.
The SDK embeds _security_config in the trace event; /ingest strips it and fans
out an sdk_security job to the evaluator Kafka topic.
"""
from __future__ import annotations

import logging

import requests

from fluiq.config import _config

logger = logging.getLogger(__name__)


def _base_url() -> str:
    return f"{_config['endpoint']}/{_config['version']}"


def pre_call_check(prompt_text: str, context: dict | None = None, guardrail: str = "default") -> None:
    """Call /secure/check before the LLM call.

    When mode='block' and the server returns allow=False, raises
    FluiqSecurityError.  In all other cases (network error, 402, warn mode)
    returns silently so the LLM call proceeds normally.

    Passes the current trace_id so the API can publish the blocked trace
    event directly, transitioning the dashboard row from "running" to "blocked".
    """
    mode = _config.get("secure_mode", "warn")
    try:
        from fluiq.integrations.shared.context import current_llm_trace_id
        trace_id = current_llm_trace_id()
        body: dict = {
            "api_key":   _config["api_key"],
            "prompt":    prompt_text,
            "trace_id":  trace_id,
            "guardrail": guardrail,
        }
        if context:
            body["context"] = context
        r = requests.post(
            f"{_base_url()}/secure/check",
            json=body,
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
