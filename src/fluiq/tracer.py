import uuid

from fluiq.client import send_event
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

        if _config.get("secure", False):
            try:
                from fluiq.security.client import call_secure
                call_secure(data)
            except Exception:
                pass

        is_cache_hit = data.pop("_cache_hit", False)

        if _config.get("optimize", False) and not is_cache_hit:
            try:
                from fluiq.optimization.client import populate_cache
                populate_cache(data)
            except Exception:
                pass

        send_event(data)

        if _config.get("eval", False) and not is_cache_hit:
            _run_eval(data)

    except Exception as exc:
        from fluiq.exceptions import FluiqEvalError
        if isinstance(exc, FluiqEvalError):
            raise
        pass


def _run_eval(data: dict) -> None:
    """Dispatch post-call evaluation in warn or block mode."""
    mode = _config.get("eval_mode", "warn")
    if mode == "block":
        try:
            from fluiq.evals.client import call_evaluate
            from fluiq.exceptions import FluiqEvalError
            scores = call_evaluate(data)
            if scores:
                thresholds = _config.get("eval_thresholds", {})
                failures = {m: s for m, s in scores.items() if s < thresholds.get(m, 0.0)}
                if failures:
                    raise FluiqEvalError(failures, scores)
        except Exception as exc:
            from fluiq.exceptions import FluiqEvalError
            if isinstance(exc, FluiqEvalError):
                raise
            import logging
            logging.getLogger("fluiq").warning(
                "[fluiq.eval] block-mode evaluation error: %s", repr(exc)
            )
    else:
        import threading

        data_snap = dict(data)

        def _warn() -> None:
            try:
                from fluiq.evals.client import call_evaluate
                scores = call_evaluate(data_snap)
                if scores:
                    thresholds = _config.get("eval_thresholds", {})
                    import logging
                    log = logging.getLogger("fluiq")
                    for metric, score in scores.items():
                        threshold = thresholds.get(metric, 0.0)
                        if threshold > 0 and score < threshold:
                            log.warning(
                                "[fluiq.eval] %s score %.3f below threshold %.3f"
                                " (trace_id=%s)",
                                metric, score, threshold,
                                data_snap.get("trace_id", "?"),
                            )
            except Exception:
                pass

        threading.Thread(target=_warn, daemon=True).start()
