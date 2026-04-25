import time
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace
from fluiq.integrations.Gemini.helper.utils import _to_jsonable, _strip_media
from fluiq.integrations.Gemini.helper.tool_trace import (
    _extract_function_calls,
    _extract_request_tools,
    _finish_reasons,
    _compute_tool_call_latencies,
    _gc_pending_tool_calls,
    _record_dispatched_tool_calls,
)
from fluiq.integrations.Gemini.helper.thinking_trace import _extract_thinking
from fluiq.integrations.Gemini.helper.mcp_trace import _extract_mcp_servers

def _usage_dict(usage):
    if not usage:
        return None
    return {
        "prompt": getattr(usage, "prompt_token_count", None),
        "completion": getattr(usage, "candidates_token_count", None),
        "total": getattr(usage, "total_token_count", None),
    }

def patch_genai():
    from google.genai.models import Models

    original = Models.generate_content

    def wrapped(self, *args, **kwargs):

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("contents"))

        start = time.time()
        response = original(self, *args, **kwargs)
        end = time.time()
        usage = getattr(response, "usage_metadata", None)

        candidates = getattr(response, "candidates", None)
        _record_dispatched_tool_calls(candidates)
        thinking = _extract_thinking(candidates)
        text = _strip_media(candidates)
        function_calls = _extract_function_calls(candidates)
        tools, tool_config = _extract_request_tools(kwargs)
        mcp_servers = _extract_mcp_servers(kwargs)

        payload = LogTrace(
            type="llm",
            model=kwargs.get("model"),
            contents=_to_jsonable(kwargs.get("contents")),
            system_instruction=_to_jsonable(
                (kwargs.get("config") or {}).get("system_instruction")
                if isinstance(kwargs.get("config"), dict)
                else getattr(kwargs.get("config"), "system_instruction", None)
            ),
            tools=tools,
            tool_config=tool_config,
            response=text,
            thinking=thinking,
            mcp_servers=mcp_servers,
            function_calls=function_calls,
            tool_call_latencies=tool_call_latencies,
            finish_reasons=_finish_reasons(candidates),
            latency=end - start,
            tokens=_usage_dict(usage),
        )
        log_trace(payload.model_dump(mode="json"))

        return response

    Models.generate_content = wrapped

def patch_vertexai():
    from vertexai.generative_models import GenerativeModel
    original = GenerativeModel.generate_content

    def wrapped(self, *args, **kwargs):

        _gc_pending_tool_calls()
        request_contents = args[0] if args else kwargs.get("contents")
        tool_call_latencies = _compute_tool_call_latencies(request_contents)

        start = time.time()
        response = original(self, *args, **kwargs)
        end = time.time()
        usage = getattr(response, "usage_metadata", None)

        candidates = getattr(response, "candidates", None)
        _record_dispatched_tool_calls(candidates)
        thinking = _extract_thinking(candidates)
        text = _strip_media(candidates)
        function_calls = _extract_function_calls(candidates)
        tools, tool_config = _extract_request_tools(kwargs, instance=self)
        mcp_servers = _extract_mcp_servers(kwargs)

        model = getattr(self, "_model_name",None) or getattr(self, "model_name",None)

        payload = LogTrace(
            type="llm",
            model=model,
            contents=_to_jsonable(request_contents),
            tools=tools,
            tool_config=tool_config,
            response=text,
            thinking=thinking,
            mcp_servers=mcp_servers,
            function_calls=function_calls,
            tool_call_latencies=tool_call_latencies,
            finish_reasons=_finish_reasons(candidates),
            latency=end - start,
            tokens=_usage_dict(usage),
        )
        log_trace(payload.model_dump(mode="json"))

        return response

    GenerativeModel.generate_content = wrapped
    