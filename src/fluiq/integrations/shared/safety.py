import functools


def _fail_open(fn):
    """Run `fn` and swallow any exception. The SDK's observability pipeline
    must never crash the user's application — quota errors, unreachable
    ingest, payload-construction bugs and serialization faults all return
    None instead of propagating.

    FluiqEvalError and FluiqSecurityError are intentional user-facing signals
    (block mode) and must propagate through."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            from fluiq.exceptions import FluiqEvalError, FluiqSecurityError
            if isinstance(exc, (FluiqEvalError, FluiqSecurityError)):
                raise
            return None
    return wrapper
