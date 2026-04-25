import time
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.OpenAI.helper.utils import _to_jsonable, _strip_media
from fluiq.integrations.OpenAI.helper.tool_trace import (
    _extract_tool_calls,
    _finish_reasons,
    _compute_tool_call_latencies,
    _gc_pending_tool_calls,
    _record_dispatched_tool_calls,
)
from fluiq.integrations.OpenAI.helper.thinking_trace import _extract_thinking

def patch_openai():
    from openai.resources.chat.completions import Completions

    original = Completions.create

    def wrapped(self, *args, **kwargs):

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))

        start = time.time()
        response = original(self, *args, **kwargs)
        end = time.time()
        usage = getattr(response, "usage", None)

        choices = getattr(response, "choices", None) or []
        _record_dispatched_tool_calls(choices)
        thinking = _extract_thinking(choices, usage=usage)

        text = None
        for choice in choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message else None
            extracted = _strip_media(content)
            if extracted is not None:
                text = extracted
                break

        payload = LogTrace(
            type="llm",
            integration=TraceType.OpenAI,
            model=kwargs.get("model"),
            messages=kwargs.get("messages"),
            tools=_to_jsonable(kwargs.get("tools")),
            tool_choice=_to_jsonable(kwargs.get("tool_choice")),
            response=text,
            thinking=thinking,
            tool_calls=_extract_tool_calls(choices=choices),
            tool_call_latencies=tool_call_latencies,
            finish_reasons=_finish_reasons(choices=choices),
            latency=end - start,
            tokens={
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            } if usage else None,
        )
        log_trace(payload.model_dump(mode="json"))

        return response

    Completions.create = wrapped

def patch_openai_responses():
    from openai.resources.responses import Responses
    from fluiq.integrations.OpenAI.helper.mcp_trace import (
        _extract_mcp_servers_from_tools,
        _extract_mcp_calls_from_output,
    )
    original = Responses.create

    def wrapped(self, *args, **kwargs):
        start = time.time()
        response = original(self, *args, **kwargs)
        end = time.time()

        output = getattr(response, "output", None)
        usage = getattr(response, "usage", None)

        payload = LogTrace(
            type="llm",
            integration=TraceType.OpenAI,
            api="responses",
            model=kwargs.get("model"),
            input=_to_jsonable(kwargs.get("input")),
            tools=_to_jsonable(kwargs.get("tools")),
            mcp_servers=_extract_mcp_servers_from_tools(kwargs.get("tools")),
            mcp_calls=_extract_mcp_calls_from_output(output),
            response=_to_jsonable(output),
            latency=end - start,
            tokens=_to_jsonable(usage),
        )
        log_trace(payload.model_dump(mode="json"))
        return response

    Responses.create = wrapped