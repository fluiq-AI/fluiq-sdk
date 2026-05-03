import traceback
from contextvars import ContextVar

_in_langchain_llm: ContextVar = ContextVar("fluiq_in_langchain_llm", default=False)
_current_trace_id: ContextVar = ContextVar("fluiq_current_trace_id", default=None)
_current_llm_trace_id: ContextVar = ContextVar("fluiq_current_llm_trace_id", default=None)


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


def format_error_traceback(error):
    if error is None:
        return None
    tb = getattr(error, "__traceback__", None)
    if tb is not None:
        return "".join(traceback.format_exception(type(error), error, tb))
    formatted = traceback.format_exc()
    return formatted if formatted and formatted.strip() != "NoneType: None" else None
