import time
import uuid
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType

def trace(func):
    def wrapper(*args, **kwargs):
        trace_id = str(uuid.uuid4())
        start = time.time()
        exc = None

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            exc = e

        end = time.time()

        payload = LogTrace(
            trace_id=trace_id,
            integration=TraceType.General_Function,
            function=func.__name__,
            input=str(args) + str(kwargs),
            output=str(result) if success else str(exc),
            latency=end - start,
            success=success,
        )
        log_trace(payload.model_dump(mode="json"))

        if exc is not None:
            raise exc

        return result

    return wrapper