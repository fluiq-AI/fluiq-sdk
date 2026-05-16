"""Thin SDK client for the fluiq-api /evaluate endpoint.

call_evaluate() — post-call: sends prompt + response to the Fluiq backend,
which runs LLM-as-judge and stores results in ClickHouse.  Returns a
{metric: score} dict, or None on any network / server error.

Never raises — all failures are logged and swallowed so the user's
application is never interrupted by observability infrastructure.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from fluiq.config import _config

logger = logging.getLogger(__name__)


def _extract_question(trace: dict[str, Any]) -> str:
    """Pull the user-facing question out of a trace dict."""
    messages = trace.get("messages") or trace.get("contents") or trace.get("input") or []
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str):
                    return content
        parts = [
            str(m.get("content") or "")
            for m in messages
            if isinstance(m, dict) and m.get("content")
        ]
        return "\n".join(parts)
    return str(messages)


def call_evaluate(trace: dict[str, Any]) -> dict[str, float] | None:
    """POST trace data to /evaluate, return ``{metric: score}`` or ``None``.

    Called from ``tracer.log_trace()`` when ``fluiq.eval()`` is active.
    """
    try:
        metrics    = _config.get("eval_metrics") or ["hallucination", "relevance"]
        judge_model = _config.get("eval_judge_model", "gpt-4o-mini")
        thresholds  = _config.get("eval_thresholds", {})

        base = f"{_config['endpoint']}/{_config['version']}"
        r = requests.post(
            f"{base}/evaluate",
            json={
                "api_key":     _config["api_key"],
                "trace_id":    trace.get("trace_id"),
                "model":       trace.get("model", ""),
                "prompt":      _extract_question(trace),
                "response":    trace.get("response") or trace.get("output") or "",
                "context":     "",
                "metrics":     list(metrics),
                "judge_model": judge_model,
                "thresholds":  dict(thresholds),
            },
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("scores") or {}
        logger.warning("[fluiq.eval] /evaluate returned HTTP %s", r.status_code)
        return None
    except Exception as exc:
        logger.warning("[fluiq.eval] evaluation failed: %s", repr(exc))
        return None