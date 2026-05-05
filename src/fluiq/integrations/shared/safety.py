import functools


def _fail_open(fn):
    """Run `fn` and swallow any exception. The SDK's observability pipeline
    must never crash the user's application — quota errors, unreachable
    ingest, payload-construction bugs and serialization faults all return
    None instead of propagating."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None
    return wrapper
