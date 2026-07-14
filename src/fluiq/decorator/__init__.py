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
    _inner_cache_hit,
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
def _emit(trace_id, parent_id, func_name, args, kwargs, result, exc, start, end, *, cache_hit=False, parent_ids=None):
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
    data = payload.model_dump(mode="json")
    if cache_hit:
        data["_cache_hit"] = True
    log_trace(data)


def _build_wrapper(func, func_name: str):
    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            from fluiq.config import _config
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            declared = take_declared_parents()
            token = push_trace_id(trace_id)
            start = time.time()
            _optimize = _config.get("optimize")
            _args_key = str(args) + str(kwargs)
            _cached_payload = None
            if _optimize:
                try:
                    from fluiq.optimization.client import lookup_function_cache
                    _cached_payload = lookup_function_cache(func_name, _args_key)
                except Exception:
                    pass
            if _cached_payload is not None:
                cached_result = _cached_payload.get("result")
                end = time.time()
                _emit_start(trace_id, parent_id, func_name, args, kwargs, start, parent_ids=declared)
                pop_trace_id(token)
                _emit(trace_id, parent_id, func_name, args, kwargs, cached_result, None, start, end, cache_hit=True, parent_ids=declared)
                return cached_result
            _emit_start(trace_id, parent_id, func_name, args, kwargs, start, parent_ids=declared)
            exc = None
            result = None
            _hit_token = _inner_cache_hit.set(False)
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                exc = e
            end = time.time()
            _hit = _inner_cache_hit.get()
            _inner_cache_hit.reset(_hit_token)
            if _optimize and exc is None:
                try:
                    from fluiq.optimization.client import populate_function_cache
                    populate_function_cache(func_name, _args_key, result)
                except Exception:
                    pass
            pop_trace_id(token)
            _emit(trace_id, parent_id, func_name, args, kwargs, result, exc, start, end, cache_hit=_hit, parent_ids=declared)
            if exc is not None:
                raise exc
            return result
        async_wrapper.__name__ = func.__name__
        async_wrapper.__doc__  = func.__doc__
        return async_wrapper

    def wrapper(*args, **kwargs):
        from fluiq.config import _config
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        declared = take_declared_parents()
        token = push_trace_id(trace_id)
        start = time.time()
        _optimize = _config.get("optimize")
        _args_key = str(args) + str(kwargs)
        _cached_payload = None
        if _optimize:
            try:
                from fluiq.optimization.client import lookup_function_cache
                _cached_payload = lookup_function_cache(func_name, _args_key)
            except Exception:
                pass
        if _cached_payload is not None:
            cached_result = _cached_payload.get("result")
            end = time.time()
            _emit_start(trace_id, parent_id, func_name, args, kwargs, start, parent_ids=declared)
            pop_trace_id(token)
            _emit(trace_id, parent_id, func_name, args, kwargs, cached_result, None, start, end, cache_hit=True, parent_ids=declared)
            return cached_result
        _emit_start(trace_id, parent_id, func_name, args, kwargs, start, parent_ids=declared)
        exc = None
        result = None
        _hit_token = _inner_cache_hit.set(False)
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exc = e
        end = time.time()
        _hit = _inner_cache_hit.get()
        _inner_cache_hit.reset(_hit_token)
        if _optimize and exc is None:
            try:
                from fluiq.optimization.client import populate_function_cache
                populate_function_cache(func_name, _args_key, result)
            except Exception:
                pass
        pop_trace_id(token)
        _emit(trace_id, parent_id, func_name, args, kwargs, result, exc, start, end, cache_hit=_hit, parent_ids=declared)
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
