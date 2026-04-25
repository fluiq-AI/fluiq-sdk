import time
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace
from fluiq.integrations.Anthropic.helper.utils import _strip_media, _to_jsonable
from fluiq.integrations.Anthropic.helper.tool_trace import (
    _extract_tool_use,
    _compute_tool_call_latencies,
    _gc_pending_tool_calls,
    _record_dispatched_tool_calls,
)
from fluiq.integrations.Anthropic.helper.thinking_trace import _extract_thinking
from fluiq.integrations.Anthropic.helper.mcp_trace import (
    _extract_mcp_servers,
    _extract_mcp_blocks,
    _extract_mcp_results_from_messages
)

def patch_anthropic():
    from anthropic.resources.messages import Messages
    original = Messages.create

    def wrapped(self, *args, **kwargs):

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))

        start = time.time()
        response = original(self, *args, **kwargs)
        end = time.time()
        usage = getattr(response, "usage", None)

        content = getattr(response, "content", None)
        _record_dispatched_tool_calls(content)
        thinking = _extract_thinking(content)
        text = _strip_media(content)
        tool_uses = _extract_tool_use(text)
        mcp_servers = _extract_mcp_servers(kwargs)
        mcp_calls = _extract_mcp_blocks(content)
        mcp_results = _extract_mcp_results_from_messages(kwargs.get("messages"))

        total_tokens = (usage.input_tokens + usage.output_tokens) if usage else None

        payload = LogTrace(
            type="llm",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
            system=_to_jsonable(kwargs.get("system")),
            tools=_to_jsonable(kwargs.get("tools")),
            tool_choice=_to_jsonable(kwargs.get("tool_choice")),
            response=text,
            thinking=thinking,
            tool_uses=tool_uses,
            tool_call_latencies=tool_call_latencies,
            mcp_servers=mcp_servers,
            mcp_calls=mcp_calls,
            mcp_results=mcp_results,
            stop_reason=getattr(response, "stop_reason", None),
            latency=end - start,
            tokens={
                "prompt": usage.input_tokens,
                "completion": usage.output_tokens,
                "total": total_tokens,
            } if usage else None,
        )
        log_trace(payload.model_dump(mode="json"))

        return response

    Messages.create = wrapped
