import traceback
from contextlib import contextmanager
from contextvars import ContextVar

_in_langchain_llm: ContextVar = ContextVar("fluiq_in_langchain_llm", default=False)
_current_trace_id: ContextVar = ContextVar("fluiq_current_trace_id", default=None)
_current_llm_trace_id: ContextVar = ContextVar("fluiq_current_llm_trace_id", default=None)
_inner_cache_hit: ContextVar[bool] = ContextVar("fluiq_inner_cache_hit", default=False)
# Multiple parents declared for the *next* traced call — the generic seam for
# agent-to-agent / custom multi-agent join nodes (see fluiq.join_parents).
_declared_parent_ids: ContextVar = ContextVar("fluiq_declared_parent_ids", default=None)


def is_in_langchain_llm() -> bool:
    return bool(_in_langchain_llm.get())


def enter_langchain_llm():
    return _in_langchain_llm.set(True)


def exit_langchain_llm(token):
    if token is None:
        return
    try:
        _in_langchain_llm.reset(token)
    except (ValueError, LookupError):
        pass


def current_parent_id():
    return _current_trace_id.get()


def push_trace_id(trace_id):
    return _current_trace_id.set(trace_id)


def pop_trace_id(token):
    if token is None:
        return
    try:
        _current_trace_id.reset(token)
    except (ValueError, LookupError):
        pass


def current_llm_trace_id():
    return _current_llm_trace_id.get()


def push_llm_trace_id(trace_id):
    return _current_llm_trace_id.set(trace_id)


def pop_llm_trace_id(token):
    if token is None:
        return
    try:
        _current_llm_trace_id.reset(token)
    except (ValueError, LookupError):
        pass


def take_declared_parents():
    """Return and CLEAR the parents declared for the next traced call.

    Cleared so the declaration applies only to the immediate traced call — a
    nested traced call inside it does not inherit the join parents."""
    ids = _declared_parent_ids.get()
    if ids:
        _declared_parent_ids.set(None)
    return ids


@contextmanager
def declare_parents(*parent_ids):
    """Scope-declare the join parents for the traced call made inside the block.

    Accepts either varargs (``join_parents(a, b)``) or a single iterable
    (``join_parents([a, b])``)."""
    if len(parent_ids) == 1 and isinstance(parent_ids[0], (list, tuple, set)):
        parent_ids = tuple(parent_ids[0])
    cleaned = [str(p) for p in parent_ids if p]
    token = _declared_parent_ids.set(cleaned or None)
    try:
        yield
    finally:
        try:
            _declared_parent_ids.reset(token)
        except (ValueError, LookupError):
            pass


def mark_inner_cache_hit() -> None:
    """Signal that an inner LLM/vectorstore call was served from cache.

    Called by integration patches when they detect a cache hit.  The
    enclosing @fluiq.trace wrapper reads and clears this flag after func()
    returns so the function-level trace also shows _cache_hit=True.
    """
    _inner_cache_hit.set(True)


def format_error_traceback(error):
    if error is None:
        return None
    tb = getattr(error, "__traceback__", None)
    if tb is not None:
        return "".join(traceback.format_exception(type(error), error, tb))
    formatted = traceback.format_exc()
    return formatted if formatted and formatted.strip() != "NoneType: None" else None
