"""Thin SDK client for block-mode evaluation.

call_evaluate_block() — synchronously calls /evaluate and returns
{metric: score} or None. Only used in block mode; warn-mode evaluation
is handled entirely server-side via the Kafka → worker pipeline.

Never raises — failures are logged and swallowed. The caller in tracer.py
inspects the returned scores and raises FluiqEvalError when appropriate.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from fluiq.config import _config, auth_headers

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
                if isinstance(content, str) and content.strip():
                    return content
        parts = [
            str(m.get("content") or "")
            for m in messages
            if isinstance(m, dict) and m.get("content")
        ]
        return "\n".join(parts)
    return str(messages) if messages else ""


def call_evaluate_block(trace: dict[str, Any]) -> dict[str, float] | None:
    """POST trace to /evaluate and return {metric: score} or None (block mode only)."""
    try:
        metrics       = _config.get("eval_metrics") or ["hallucination", "relevance"]
        judge_model   = _config.get("eval_judge_model", "claude-haiku-4-5-20251001")
        thresholds    = _config.get("eval_thresholds", {})
        custom_judges = _config.get("eval_custom_judges", {})

        base = f"{_config['endpoint']}/{_config['version']}"
        r = requests.post(
            f"{base}/evaluate",
            headers=auth_headers(),
            json={
                "trace_id":      trace.get("trace_id"),
                "model":         trace.get("model") or "",
                "prompt":        _extract_question(trace),
                "response":      trace.get("response") or trace.get("output") or "",
                "context":       "",
                "metrics":       list(metrics),
                "judge_model":   judge_model,
                "thresholds":    dict(thresholds),
                "custom_judges": dict(custom_judges),
            },
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("scores") or {}
        logger.warning("[fluiq.eval] /evaluate returned HTTP %s", r.status_code)
        return None
    except Exception as exc:
        logger.warning("[fluiq.eval] block-mode evaluation failed: %s", repr(exc))
        return None
