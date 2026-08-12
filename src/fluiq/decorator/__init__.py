import asyncio
import time
import uuid
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import (
    current_parent_id,
    push_trace_id,
    pop_trace_id,
    take_declared_parents,
)
from fluiq.integrations.shared.safety import _fail_open


@_fail_open
def _emit_start(trace_id, parent_id, func_name, args, kwargs, start, parent_ids=None):
    payload = LogTrace(
        trace_id=trace_id,
        parent_id=parent_id,
        parent_ids=parent_ids,
        integration=TraceType.General_Function,
        function=func_name,
        type="function",
        input=str(args) + str(kwargs),
        status="running",
        started_at=start,
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit(trace_id, parent_id, func_name, args, kwargs, result, exc, start, end, *, parent_ids=None):
    success = exc is None
    payload = LogTrace(
        trace_id=trace_id,
        parent_id=parent_id,
        parent_ids=parent_ids,
        integration=TraceType.General_Function,
        function=func_name,
        type="function",
        input=str(args) + str(kwargs),
        output=str(result) if success else str(exc),
        latency=end - start,
        success=success,
        status="success" if success else "error",
        started_at=start,
    )
    log_trace(payload.model_dump(mode="json"))


def _build_wrapper(func, func_name: str):
    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            declared = take_declared_parents()
            token = push_trace_id(trace_id)
            start = time.time()
            _emit_start(trace_id, parent_id, func_name, args, kwargs, start, parent_ids=declared)
            exc = None
            result = None
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                exc = e
            end = time.time()
            pop_trace_id(token)
            _emit(trace_id, parent_id, func_name, args, kwargs, result, exc, start, end, parent_ids=declared)
            if exc is not None:
                raise exc
            return result
        async_wrapper.__name__ = func.__name__
        async_wrapper.__doc__  = func.__doc__
        return async_wrapper

    def wrapper(*args, **kwargs):
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        declared = take_declared_parents()
        token = push_trace_id(trace_id)
        start = time.time()
        _emit_start(trace_id, parent_id, func_name, args, kwargs, start, parent_ids=declared)
        exc = None
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exc = e
        end = time.time()
        pop_trace_id(token)
        _emit(trace_id, parent_id, func_name, args, kwargs, result, exc, start, end, parent_ids=declared)
        if exc is not None:
            raise exc
        return result

    wrapper.__name__ = func.__name__
    wrapper.__doc__  = func.__doc__
    return wrapper


def trace(func=None, *, name=None):
    if func is not None and callable(func):
        return _build_wrapper(func, name or func.__name__)

    def decorator(f):
        return _build_wrapper(f, name or f.__name__)

    return decorator
