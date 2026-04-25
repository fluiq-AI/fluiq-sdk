import time
import uuid
from ..tracer import log_trace

def trace(func):
    def wrapper(*args, **kwargs):
        trace_id = str(uuid.uuid4())
        start = time.time()

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = str(e)
            success = False
            return
        
        end = time.time()

        log_trace({
            "trace_id": trace_id,
            "function": func.__name__,
            "input": str(args) + str(kwargs),
            "output": str(result),
            "latency": end - start,
            "success": success
        })

        return result
    
    return wrapper