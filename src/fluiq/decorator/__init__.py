import asyncio
import time
import uuid
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import (
    current_parent_id,
    push_trace_id,
    pop_trace_id,
)


def _emit(trace_id, parent_id, func_name, args, kwargs, result, exc, start, end):
    success = exc is None
    payload = LogTrace(
        trace_id=trace_id,
        parent_id=parent_id,
        integration=TraceType.General_Function,
        function=func_name,
        input=str(args) + str(kwargs),
        output=str(result) if success else str(exc),
        latency=end - start,
        success=success,
    )
    log_trace(payload.model_dump(mode="json"))


def trace(func):
    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            token = push_trace_id(trace_id)
            start = time.time()
            exc = None
            result = None
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                exc = e
            end = time.time()
            pop_trace_id(token)
            _emit(trace_id, parent_id, func.__name__, args, kwargs, result, exc, start, end)
            if exc is not None:
                raise exc
            return result
        return async_wrapper

    def wrapper(*args, **kwargs):
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        token = push_trace_id(trace_id)
        start = time.time()
        exc = None
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exc = e
        end = time.time()
        pop_trace_id(token)
        _emit(trace_id, parent_id, func.__name__, args, kwargs, result, exc, start, end)
        if exc is not None:
            raise exc
        return result

    return wrapper