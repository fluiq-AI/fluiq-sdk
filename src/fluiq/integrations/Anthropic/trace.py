import time
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import (
    is_in_langchain_llm,
    current_parent_id,
    format_error_traceback,
    push_llm_trace_id,
    pop_llm_trace_id,
)
from fluiq.integrations.shared.llm_start import emit_llm_start
from fluiq.integrations.shared.safety import _fail_open
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
    _extract_mcp_results_from_messages,
)
from fluiq.integrations.Anthropic.helper.streaming import _MessageStreamAccumulator
from fluiq.integrations.OpenAI.helper.streaming import _StreamProxy, _AsyncStreamProxy


@_fail_open
def _emit_messages_trace(kwargs, response, start, end, tool_call_latencies):
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
        integration=TraceType.Anthropic,
        model=kwargs.get("model") or getattr(response, "model", None),
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
        parent_id=current_parent_id(),
        tokens={
            "prompt": usage.input_tokens,
            "completion": usage.output_tokens,
            "total": total_tokens,
        } if usage else None,
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit_messages_stream_trace(kwargs, acc, start, end, tool_call_latencies):
    data = acc.assemble()
    mcp_servers = _extract_mcp_servers(kwargs)
    mcp_results = _extract_mcp_results_from_messages(kwargs.get("messages"))
    payload = LogTrace(
        type="llm",
        integration=TraceType.Anthropic,
        api="messages.stream",
        model=kwargs.get("model") or data["model"],
        messages=_to_jsonable(kwargs.get("messages")),
        system=_to_jsonable(kwargs.get("system")),
        tools=_to_jsonable(kwargs.get("tools")),
        tool_choice=_to_jsonable(kwargs.get("tool_choice")),
        response=data["text"],
        thinking=data["thinking"],
        tool_uses=data["tool_uses"],
        tool_call_latencies=tool_call_latencies,
        mcp_servers=mcp_servers,
        mcp_results=mcp_results,
        stop_reason=data["stop_reason"],
        latency=end - start,
        parent_id=current_parent_id(),
        tokens=data["tokens"],
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit_messages_error(kwargs, error, start, end, api=None):
    payload = LogTrace(
        type="llm",
        integration=TraceType.Anthropic,
        api=api,
        model=kwargs.get("model"),
        messages=_to_jsonable(kwargs.get("messages")),
        system=_to_jsonable(kwargs.get("system")),
        tools=_to_jsonable(kwargs.get("tools")),
        output=str(error),
        error_traceback=format_error_traceback(error),
        latency=end - start,
        parent_id=current_parent_id(),
        success=False,
    )
    log_trace(payload.model_dump(mode="json"))


def _wrap_messages_stream(stream, kwargs, start, tool_call_latencies, async_=False, trace_id=None):
    acc = _MessageStreamAccumulator()

    def on_chunk(chunk):
        acc.feed(chunk)

    def on_end():
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_messages_stream_trace(kwargs, acc, start, time.time(), tool_call_latencies)
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)

    def on_error(exc):
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_messages_error(kwargs, exc, start, time.time(), api="messages.stream")
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)

    Proxy = _AsyncStreamProxy if async_ else _StreamProxy
    return Proxy(stream, on_chunk, on_end, on_error=on_error)


def _build_messages_wrapper(original):
    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))

        trace_id, ctx_tok = emit_llm_start(
            TraceType.Anthropic,
            api="messages",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
            system=_to_jsonable(kwargs.get("system")),
        )
        start = time.time()
        try:
            try:
                response = original(self, *args, **kwargs)
            except Exception as e:
                _emit_messages_error(kwargs, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _wrap_messages_stream(response, kwargs, start, tool_call_latencies, async_=False, trace_id=trace_id)

            end = time.time()
            _emit_messages_trace(kwargs, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    return wrapped


def _build_async_messages_wrapper(original):
    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))

        trace_id, ctx_tok = emit_llm_start(
            TraceType.Anthropic,
            api="messages",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
            system=_to_jsonable(kwargs.get("system")),
        )
        start = time.time()
        try:
            try:
                response = await original(self, *args, **kwargs)
            except Exception as e:
                _emit_messages_error(kwargs, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _wrap_messages_stream(response, kwargs, start, tool_call_latencies, async_=True, trace_id=trace_id)

            end = time.time()
            _emit_messages_trace(kwargs, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    return wrapped


class _AnthropicStreamManagerProxy:
    def __init__(self, manager, kwargs, tool_call_latencies, trace_id=None):
        self._mgr = manager
        self._kwargs = kwargs
        self._tcl = tool_call_latencies
        self._stream = None
        self._start = None
        self._trace_id = trace_id

    def __enter__(self):
        self._start = time.time()
        self._stream = self._mgr.__enter__()
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._mgr.__exit__(exc_type, exc, tb)
        finally:
            self._emit()

    def _emit(self):
        try:
            final = self._stream.get_final_message() if self._stream else None
        except Exception:
            final = getattr(self._stream, "current_message_snapshot", None)
        if final is not None:
            tok = push_llm_trace_id(self._trace_id) if self._trace_id else None
            try:
                _emit_messages_trace(self._kwargs, final, self._start, time.time(), self._tcl)
            finally:
                if tok is not None:
                    pop_llm_trace_id(tok)


class _AsyncAnthropicStreamManagerProxy:
    def __init__(self, manager, kwargs, tool_call_latencies, trace_id=None):
        self._mgr = manager
        self._kwargs = kwargs
        self._tcl = tool_call_latencies
        self._stream = None
        self._start = None
        self._trace_id = trace_id

    async def __aenter__(self):
        self._start = time.time()
        self._stream = await self._mgr.__aenter__()
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._mgr.__aexit__(exc_type, exc, tb)
        finally:
            await self._emit()

    async def _emit(self):
        try:
            final = await self._stream.get_final_message() if self._stream else None
        except Exception:
            final = getattr(self._stream, "current_message_snapshot", None)
        if final is not None:
            tok = push_llm_trace_id(self._trace_id) if self._trace_id else None
            try:
                _emit_messages_trace(self._kwargs, final, self._start, time.time(), self._tcl)
            finally:
                if tok is not None:
                    pop_llm_trace_id(tok)


def _build_stream_helper_wrapper(original):
    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Anthropic,
            api="messages.stream",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
            system=_to_jsonable(kwargs.get("system")),
        )
        start = time.time()
        try:
            try:
                manager = original(self, *args, **kwargs)
            except Exception as e:
                _emit_messages_error(kwargs, e, start, time.time(), api="messages.stream")
                raise
            return _AnthropicStreamManagerProxy(manager, kwargs, tool_call_latencies, trace_id=trace_id)
        finally:
            pop_llm_trace_id(ctx_tok)
    return wrapped


def _build_async_stream_helper_wrapper(original):
    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Anthropic,
            api="messages.stream",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
            system=_to_jsonable(kwargs.get("system")),
        )
        start = time.time()
        try:
            try:
                manager = original(self, *args, **kwargs)
            except Exception as e:
                _emit_messages_error(kwargs, e, start, time.time(), api="messages.stream")
                raise
            return _AsyncAnthropicStreamManagerProxy(manager, kwargs, tool_call_latencies, trace_id=trace_id)
        finally:
            pop_llm_trace_id(ctx_tok)
    return wrapped


@_fail_open
def _emit_count_tokens_trace(api, kwargs, response, start, end):
    input_tokens = getattr(response, "input_tokens", None)
    payload = LogTrace(
        type="llm",
        integration=TraceType.Anthropic,
        api=api,
        model=kwargs.get("model"),
        messages=_to_jsonable(kwargs.get("messages")),
        system=_to_jsonable(kwargs.get("system")),
        tools=_to_jsonable(kwargs.get("tools")),
        response=_to_jsonable(response),
        latency=end - start,
        parent_id=current_parent_id(),
        tokens={"prompt": input_tokens, "completion": None, "total": input_tokens} if input_tokens is not None else None,
    )
    log_trace(payload.model_dump(mode="json"))


def _build_count_tokens_wrapper(original, api):
    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        start = time.time()
        try:
            response = original(self, *args, **kwargs)
        except Exception as e:
            _emit_messages_error(kwargs, e, start, time.time(), api=api)
            raise
        end = time.time()
        _emit_count_tokens_trace(api, kwargs, response, start, end)
        return response
    return wrapped


def _build_async_count_tokens_wrapper(original, api):
    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)
        start = time.time()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as e:
            _emit_messages_error(kwargs, e, start, time.time(), api=api)
            raise
        end = time.time()
        _emit_count_tokens_trace(api, kwargs, response, start, end)
        return response
    return wrapped


def patch_anthropic():
    from anthropic.resources.messages import Messages
    Messages.create = _build_messages_wrapper(Messages.create)
    if hasattr(Messages, "stream"):
        Messages.stream = _build_stream_helper_wrapper(Messages.stream)
    if hasattr(Messages, "count_tokens"):
        Messages.count_tokens = _build_count_tokens_wrapper(Messages.count_tokens, "messages.count_tokens")


def patch_anthropic_async():
    from anthropic.resources.messages import AsyncMessages
    AsyncMessages.create = _build_async_messages_wrapper(AsyncMessages.create)
    if hasattr(AsyncMessages, "stream"):
        AsyncMessages.stream = _build_async_stream_helper_wrapper(AsyncMessages.stream)
    if hasattr(AsyncMessages, "count_tokens"):
        AsyncMessages.count_tokens = _build_async_count_tokens_wrapper(
            AsyncMessages.count_tokens, "messages.count_tokens"
        )


def patch_anthropic_beta():
    from anthropic.resources.beta.messages import Messages as BetaMessages
    BetaMessages.create = _build_messages_wrapper(BetaMessages.create)
    if hasattr(BetaMessages, "stream"):
        BetaMessages.stream = _build_stream_helper_wrapper(BetaMessages.stream)
    if hasattr(BetaMessages, "count_tokens"):
        BetaMessages.count_tokens = _build_count_tokens_wrapper(
            BetaMessages.count_tokens, "beta.messages.count_tokens"
        )


def patch_anthropic_beta_async():
    from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
    AsyncBetaMessages.create = _build_async_messages_wrapper(AsyncBetaMessages.create)
    if hasattr(AsyncBetaMessages, "stream"):
        AsyncBetaMessages.stream = _build_async_stream_helper_wrapper(AsyncBetaMessages.stream)
    if hasattr(AsyncBetaMessages, "count_tokens"):
        AsyncBetaMessages.count_tokens = _build_async_count_tokens_wrapper(
            AsyncBetaMessages.count_tokens, "beta.messages.count_tokens"
        )
