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
from fluiq.integrations.shared.models import TraceType as _TraceType
from fluiq.integrations.shared.safety import _fail_open
from fluiq.integrations.OpenAI.helper.utils import _to_jsonable, _strip_media
from fluiq.integrations.OpenAI.helper.tool_trace import (
    _extract_tool_calls,
    _finish_reasons,
    _compute_tool_call_latencies,
    _gc_pending_tool_calls,
    _record_dispatched_tool_calls,
)
from fluiq.integrations.OpenAI.helper.thinking_trace import _extract_thinking
from fluiq.integrations.OpenAI.helper.streaming import (
    _ChatStreamAccumulator,
    _ResponsesStreamAccumulator,
    _StreamProxy,
    _AsyncStreamProxy,
)


@_fail_open
def _emit_chat_trace(kwargs, response, start, end, tool_call_latencies):
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
        model=kwargs.get("model") or getattr(response, "model", None),
        messages=kwargs.get("messages"),
        tools=_to_jsonable(kwargs.get("tools")),
        tool_choice=_to_jsonable(kwargs.get("tool_choice")),
        response=text,
        thinking=thinking,
        tool_calls=_extract_tool_calls(choices=choices),
        tool_call_latencies=tool_call_latencies,
        finish_reasons=_finish_reasons(choices=choices),
        latency=end - start,
        parent_id=current_parent_id(),
        tokens={
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens,
        } if usage else None,
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit_chat_stream_trace(kwargs, acc, start, end, tool_call_latencies):
    data = acc.assemble()
    payload = LogTrace(
        type="llm",
        integration=TraceType.OpenAI,
        api="chat.completions.stream",
        model=kwargs.get("model") or data["model"],
        messages=kwargs.get("messages"),
        tools=_to_jsonable(kwargs.get("tools")),
        tool_choice=_to_jsonable(kwargs.get("tool_choice")),
        response=data["response"],
        thinking=data["thinking"],
        tool_calls=data["tool_calls"],
        tool_call_latencies=tool_call_latencies,
        finish_reasons=data["finish_reasons"],
        latency=end - start,
        parent_id=current_parent_id(),
        tokens=data["usage"],
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit_responses_trace(kwargs, response, start, end):
    from fluiq.integrations.OpenAI.helper.mcp_trace import (
        _extract_mcp_servers_from_tools,
        _extract_mcp_calls_from_output,
    )
    output = getattr(response, "output", None)
    usage = getattr(response, "usage", None)
    payload = LogTrace(
        type="llm",
        integration=TraceType.OpenAI,
        api="responses",
        model=kwargs.get("model") or getattr(response, "model", None),
        input=_to_jsonable(kwargs.get("input")),
        tools=_to_jsonable(kwargs.get("tools")),
        mcp_servers=_extract_mcp_servers_from_tools(kwargs.get("tools")),
        mcp_calls=_extract_mcp_calls_from_output(output),
        response=_to_jsonable(output),
        latency=end - start,
        parent_id=current_parent_id(),
        tokens=_to_jsonable(usage),
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit_chat_error(kwargs, error, start, end, api="chat.completions"):
    payload = LogTrace(
        type="llm",
        integration=TraceType.OpenAI,
        api=api,
        model=kwargs.get("model"),
        messages=kwargs.get("messages"),
        tools=_to_jsonable(kwargs.get("tools")),
        tool_choice=_to_jsonable(kwargs.get("tool_choice")),
        output=str(error),
        error_traceback=format_error_traceback(error),
        latency=end - start,
        parent_id=current_parent_id(),
        success=False,
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit_responses_error(kwargs, error, start, end, api="responses"):
    payload = LogTrace(
        type="llm",
        integration=TraceType.OpenAI,
        api=api,
        model=kwargs.get("model"),
        input=_to_jsonable(kwargs.get("input")),
        tools=_to_jsonable(kwargs.get("tools")),
        output=str(error),
        error_traceback=format_error_traceback(error),
        latency=end - start,
        parent_id=current_parent_id(),
        success=False,
    )
    log_trace(payload.model_dump(mode="json"))


@_fail_open
def _emit_responses_stream_trace(kwargs, acc, start, end):
    from fluiq.integrations.OpenAI.helper.mcp_trace import (
        _extract_mcp_servers_from_tools,
        _extract_mcp_calls_from_output,
    )
    data = acc.assemble()
    final = data.get("raw_response")
    output = getattr(final, "output", None) if final is not None else None
    payload = LogTrace(
        type="llm",
        integration=TraceType.OpenAI,
        api="responses.stream",
        model=kwargs.get("model") or data["model"],
        input=_to_jsonable(kwargs.get("input")),
        tools=_to_jsonable(kwargs.get("tools")),
        mcp_servers=_extract_mcp_servers_from_tools(kwargs.get("tools")),
        mcp_calls=_extract_mcp_calls_from_output(output),
        response=data["response"],
        latency=end - start,
        parent_id=current_parent_id(),
        tokens=data["usage"],
    )
    log_trace(payload.model_dump(mode="json"))


def _wrap_chat_stream(stream, kwargs, start, tool_call_latencies, async_=False, trace_id=None):
    acc = _ChatStreamAccumulator()

    def on_chunk(chunk):
        acc.feed(chunk)

    def on_end():
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_chat_stream_trace(kwargs, acc, start, time.time(), tool_call_latencies)
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)

    def on_error(exc):
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_chat_error(kwargs, exc, start, time.time(), api="chat.completions.stream")
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)

    Proxy = _AsyncStreamProxy if async_ else _StreamProxy
    return Proxy(stream, on_chunk, on_end, on_error=on_error)


def _wrap_responses_stream(stream, kwargs, start, async_=False, trace_id=None):
    acc = _ResponsesStreamAccumulator()

    def on_chunk(chunk):
        acc.feed(chunk)

    def on_end():
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_responses_stream_trace(kwargs, acc, start, time.time())
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)

    def on_error(exc):
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_responses_error(kwargs, exc, start, time.time(), api="responses.stream")
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)

    Proxy = _AsyncStreamProxy if async_ else _StreamProxy
    return Proxy(stream, on_chunk, on_end, on_error=on_error)


def patch_openai():
    from openai.resources.chat.completions import Completions

    original = Completions.create

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))

        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="chat.completions",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                response = original(self, *args, **kwargs)
            except Exception as e:
                _emit_chat_error(kwargs, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _wrap_chat_stream(response, kwargs, start, tool_call_latencies, async_=False, trace_id=trace_id)

            end = time.time()
            _emit_chat_trace(kwargs, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    Completions.create = wrapped


def patch_openai_async():
    from openai.resources.chat.completions import AsyncCompletions

    original = AsyncCompletions.create

    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))

        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="chat.completions",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                response = await original(self, *args, **kwargs)
            except Exception as e:
                _emit_chat_error(kwargs, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _wrap_chat_stream(response, kwargs, start, tool_call_latencies, async_=True, trace_id=trace_id)

            end = time.time()
            _emit_chat_trace(kwargs, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    AsyncCompletions.create = wrapped


def patch_openai_responses():
    from openai.resources.responses import Responses
    original = Responses.create

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="responses",
            model=kwargs.get("model"),
            input=_to_jsonable(kwargs.get("input")),
        )
        start = time.time()
        try:
            try:
                response = original(self, *args, **kwargs)
            except Exception as e:
                _emit_responses_error(kwargs, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _wrap_responses_stream(response, kwargs, start, async_=False, trace_id=trace_id)

            end = time.time()
            _emit_responses_trace(kwargs, response, start, end)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    Responses.create = wrapped


def patch_openai_responses_async():
    from openai.resources.responses import AsyncResponses
    original = AsyncResponses.create

    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)
        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="responses",
            model=kwargs.get("model"),
            input=_to_jsonable(kwargs.get("input")),
        )
        start = time.time()
        try:
            try:
                response = await original(self, *args, **kwargs)
            except Exception as e:
                _emit_responses_error(kwargs, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _wrap_responses_stream(response, kwargs, start, async_=True, trace_id=trace_id)

            end = time.time()
            _emit_responses_trace(kwargs, response, start, end)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    AsyncResponses.create = wrapped


class _OpenAIStreamManagerProxy:
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
            final = self._stream.get_final_completion() if self._stream else None
        except Exception:
            final = getattr(self._stream, "current_completion_snapshot", None)
        if final is not None:
            tok = push_llm_trace_id(self._trace_id) if self._trace_id else None
            try:
                _emit_chat_trace(self._kwargs, final, self._start, time.time(), self._tcl)
            finally:
                if tok is not None:
                    pop_llm_trace_id(tok)


class _AsyncOpenAIStreamManagerProxy:
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
            final = await self._stream.get_final_completion() if self._stream else None
        except Exception:
            final = getattr(self._stream, "current_completion_snapshot", None)
        if final is not None:
            tok = push_llm_trace_id(self._trace_id) if self._trace_id else None
            try:
                _emit_chat_trace(self._kwargs, final, self._start, time.time(), self._tcl)
            finally:
                if tok is not None:
                    pop_llm_trace_id(tok)


def patch_openai_parse():
    from openai.resources.chat.completions import Completions
    if not hasattr(Completions, "parse"):
        return
    original = Completions.parse

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))
        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="chat.completions.parse",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                response = original(self, *args, **kwargs)
            except Exception as e:
                _emit_chat_error(kwargs, e, start, time.time(), api="chat.completions.parse")
                raise
            end = time.time()
            _emit_chat_trace(kwargs, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    Completions.parse = wrapped


def patch_openai_parse_async():
    from openai.resources.chat.completions import AsyncCompletions
    if not hasattr(AsyncCompletions, "parse"):
        return
    original = AsyncCompletions.parse

    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))
        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="chat.completions.parse",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                response = await original(self, *args, **kwargs)
            except Exception as e:
                _emit_chat_error(kwargs, e, start, time.time(), api="chat.completions.parse")
                raise
            end = time.time()
            _emit_chat_trace(kwargs, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    AsyncCompletions.parse = wrapped


def patch_openai_stream_helper():
    from openai.resources.chat.completions import Completions
    if not hasattr(Completions, "stream"):
        return
    original = Completions.stream

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))
        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="chat.completions.stream",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                manager = original(self, *args, **kwargs)
            except Exception as e:
                _emit_chat_error(kwargs, e, start, time.time(), api="chat.completions.stream")
                raise
            return _OpenAIStreamManagerProxy(manager, kwargs, tool_call_latencies, trace_id=trace_id)
        finally:
            pop_llm_trace_id(ctx_tok)

    Completions.stream = wrapped


def patch_openai_stream_helper_async():
    from openai.resources.chat.completions import AsyncCompletions
    if not hasattr(AsyncCompletions, "stream"):
        return
    original = AsyncCompletions.stream

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))
        trace_id, ctx_tok = emit_llm_start(
            _TraceType.OpenAI,
            api="chat.completions.stream",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                manager = original(self, *args, **kwargs)
            except Exception as e:
                _emit_chat_error(kwargs, e, start, time.time(), api="chat.completions.stream")
                raise
            return _AsyncOpenAIStreamManagerProxy(manager, kwargs, tool_call_latencies, trace_id=trace_id)
        finally:
            pop_llm_trace_id(ctx_tok)

    AsyncCompletions.stream = wrapped
