import uuid

from fluiq.client import send_event, send_event_gated
from fluiq.integrations.shared.chain import compute_chain_id
from fluiq.integrations.shared.context import current_llm_trace_id, current_parent_id

from fluiq.config import _config


def log_trace(data):
    try:
        data['timestamp'] = __import__("time").time()

        if not data.get('trace_id'):
            ctx_trace_id = current_llm_trace_id()
            data['trace_id'] = ctx_trace_id or str(uuid.uuid4())

        if not data.get('parent_id'):
            ctx_parent = current_parent_id()
            if ctx_parent:
                data['parent_id'] = ctx_parent
            else:
                chain_id = compute_chain_id(data)
                if chain_id is not None:
                    data['parent_id'] = chain_id

        # Remove legacy local-scan flag if present (no-op, kept for compat)
        data.pop("_security_scan", None)

        _is_pre_blocked = data.pop("_security_pre_blocked", False)
        if _config.get("secure", False) and not _is_pre_blocked:
            # Embed security config so /ingest fans out to the evaluator worker.
            # The worker runs the full post-call scan asynchronously.
            data["_security_config"] = {
                "mode":      _config.get("secure_mode", "warn"),
                "guardrail": _config.get("secure_guardrail", "default"),
            }

        is_cache_hit = data.pop("_cache_hit", False)
        if is_cache_hit:
            data["cache_hit"] = True
        elif _config.get("optimize") and data.get("type") in ("llm", "function") and data.get("latency") is not None:
            # LLM completion trace that went through the optimize path but was a
            # cache miss (real API call). Mark explicitly so the backend can
            # compute an accurate hit rate for the Optimize dashboard.
            data["cache_hit"] = False

        if _config.get("optimize", False) and not is_cache_hit:
            try:
                from fluiq.optimization.client import populate_cache
                populate_cache(data)
            except Exception:
                pass

        # Warn mode: embed eval config so /ingest strips it and fans out to the
        # eval worker. Block mode: /ingest gets no config; we call /evaluate
        # synchronously below after the trace is stored.
        if _config.get("eval", False) and not is_cache_hit:
            _resp = data.get("response")
            _resp_str = _resp if isinstance(_resp, str) else (
                " ".join(str(x) for x in _resp) if isinstance(_resp, list) else ""
            )
            if data.get("type") == "llm" and _resp_str.strip():
                if _config.get("eval_mode", "warn") == "warn":
                    data["_eval_config"] = {
                        "metrics": _config.get("eval_metrics") or ["hallucination", "relevance"],
                        "judge_model": _config.get("eval_judge_model", "claude-haiku-4-5-20251001"),
                        "thresholds": _config.get("eval_thresholds", {}),
                    }

        # Response gate: when secure mode='block', read /ingest's return value so
        # we can raise FluiqSecurityError before the LLM output reaches the caller.
        # Warm path (scan_responses=False on the server) returns {} immediately.
        _use_gate = (
            _config.get("secure")
            and _config.get("secure_mode") == "block"
            and not _is_pre_blocked
            and data.get("type") == "llm"
        )
        if _use_gate:
            gate = send_event_gated(data)
            if gate.get("response_blocked"):
                from fluiq.exceptions import FluiqSecurityError
                raise FluiqSecurityError(
                    block_reason=gate.get("block_reason", "Response blocked by fluiq.secure()"),
                    risk_level=gate.get("risk_level", "high"),
                    attack_types=gate.get("attack_types", []),
                )
        else:
            send_event(data)

        # Block mode: thin synchronous call to /evaluate after trace is stored.
        # Raises FluiqEvalError if any metric falls below its threshold.
        if _config.get("eval", False) and not is_cache_hit:
            _resp = data.get("response")
            _resp_str = _resp if isinstance(_resp, str) else (
                " ".join(str(x) for x in _resp) if isinstance(_resp, list) else ""
            )
            if _config.get("eval_mode") == "block":
                if data.get("type") == "llm" and _resp_str.strip():
                    from fluiq.evals.client import call_evaluate_block
                    from fluiq.exceptions import FluiqEvalError
                    scores = call_evaluate_block(data)
                    if scores:
                        thresholds = _config.get("eval_thresholds", {})
                        failures = {m: s for m, s in scores.items() if s < thresholds.get(m, 0.0)}
                        if failures:
                            raise FluiqEvalError(failures, scores)

    except Exception as exc:
        from fluiq.exceptions import FluiqEvalError, FluiqSecurityError
        if isinstance(exc, (FluiqEvalError, FluiqSecurityError)):
            raise
        pass
