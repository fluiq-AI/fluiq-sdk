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
from fluiq.integrations.shared.security_gate import pre_call_guard
from fluiq.integrations.shared.optimize_gate import pre_call_optimize
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

    _details = getattr(usage, "prompt_tokens_details", None) if usage else None
    _cached = getattr(_details, "cached_tokens", None) if _details else None
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
        prompt_cached_tokens=_cached,
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
        prompt_cached_tokens=data.get("prompt_cached_tokens"),
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
    from fluiq.config import _config

    # When secure mode='block', buffer the entire stream before yielding any chunk.
    # This lets the response gate fire before the caller receives any content.
    _needs_gate = _config.get("secure") and _config.get("secure_mode") == "block"

    if _needs_gate and not async_:
        acc = _ChatStreamAccumulator()
        chunks = []
        try:
            for chunk in stream:
                acc.feed(chunk)
                chunks.append(chunk)
        except Exception as e:
            _emit_chat_error(kwargs, e, start, time.time(), api="chat.completions.stream")
            raise
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            # May raise FluiqSecurityError if response gate blocks
            _emit_chat_stream_trace(kwargs, acc, start, time.time(), tool_call_latencies)
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)
        return iter(chunks)

    if _needs_gate and async_:
        async def _buffered():
            acc = _ChatStreamAccumulator()
            chunks = []
            try:
                async for chunk in stream:
                    acc.feed(chunk)
                    chunks.append(chunk)
            except Exception as e:
                _emit_chat_error(kwargs, e, start, time.time(), api="chat.completions.stream")
                raise
            tok = push_llm_trace_id(trace_id) if trace_id else None
            try:
                _emit_chat_stream_trace(kwargs, acc, start, time.time(), tool_call_latencies)
            finally:
                if tok is not None:
                    pop_llm_trace_id(tok)
            for chunk in chunks:
                yield chunk
        return _buffered()

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


def _emit_security_blocked_trace(kwargs: dict, exc: "Exception", start: float, end: float, api: str = "chat.completions") -> None:
    """Emit a trace for a call that was blocked by fluiq.secure(mode='block').

    The pre-call check fires before the LLM API call, so there is no response.
    Security fields are populated directly from the FluiqSecurityError so the
    dashboard's SecurityPanel can render the block without a second /secure call.
    """
    if api == "responses":
        payload = LogTrace(
            type="llm",
            integration=TraceType.OpenAI,
            api=api,
            model=kwargs.get("model"),
            input=_to_jsonable(kwargs.get("input")),
            tools=_to_jsonable(kwargs.get("tools")),
            latency=end - start,
            parent_id=current_parent_id(),
            success=False,
        )
    else:
        payload = LogTrace(
            type="llm",
            integration=TraceType.OpenAI,
            api=api,
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
            tools=_to_jsonable(kwargs.get("tools")),
            tool_choice=_to_jsonable(kwargs.get("tool_choice")),
            latency=end - start,
            parent_id=current_parent_id(),
            success=False,
        )
    attack_types = getattr(exc, "attack_types", [])
    data = payload.model_dump(mode="json")
    data.update({
        "_security_pre_blocked":        True,
        "status":                       "blocked",
        "security_risk_level":          getattr(exc, "risk_level", "high"),
        "security_risk_score":          1.0,
        "should_block":                 True,
        "block_reason":                 getattr(exc, "block_reason", str(exc)),
        "injection_detected":           "prompt_injection" in attack_types,
        "injection_patterns":           ["prompt_injection"] if "prompt_injection" in attack_types else [],
        "jailbreak_detected":           "jailbreak"         in attack_types,
        "jailbreak_patterns":           ["jailbreak"]        if "jailbreak"         in attack_types else [],
        "skeleton_key_detected":        "skeleton_key"      in attack_types,
        "skeleton_key_patterns":        ["skeleton_key"]     if "skeleton_key"      in attack_types else [],
        "secrets_detected":             False,
        "secret_types":                 [],
        "pii_entities_prompt":          [],
        "pii_entities_response":        [],
        "prompt_redacted":              "",
        "response_redacted":            "",
        "indirect_injection_detected":  False,
        "indirect_injection_sources":   [],
        "semantic_attack_score":        0.0,
    })
    log_trace(data)


def patch_openai():
    from openai.resources.chat.completions import Completions

    original = Completions.create

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)

        _gc_pending_tool_calls()
        tool_call_latencies = _compute_tool_call_latencies(kwargs.get("messages"))

        trace_id, ctx_tok = emit_llm_start(
            TraceType.OpenAI,
            api="chat.completions",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                pre_call_guard(kwargs)
            except Exception as sec_exc:
                from fluiq.exceptions import FluiqSecurityError
                if isinstance(sec_exc, FluiqSecurityError):
                    _emit_security_blocked_trace(kwargs, sec_exc, start, time.time(), api="chat.completions")
                raise

            try:
                from fluiq.integrations.shared.tool_cache import learn_from_openai_messages
                learn_from_openai_messages(kwargs.get("messages") or [])
            except Exception:
                pass

            cached = pre_call_optimize(kwargs, "openai")
            if cached is not None:
                end = time.time()
                _payload = getattr(cached, "_fluiq_payload", {})
                log_trace({
                    "type": "llm",
                    "integration": TraceType.OpenAI.value,
                    "api": "chat.completions",
                    "model": kwargs.get("model"),
                    "messages": _to_jsonable(kwargs.get("messages")),
                    "tools": _to_jsonable(kwargs.get("tools")),
                    "response": _payload.get("response"),
                    "tool_calls": _payload.get("tool_calls"),
                    "mcp_calls": _payload.get("mcp_calls"),
                    "mcp_results": _payload.get("mcp_results"),
                    "mcp_servers": _payload.get("mcp_servers"),
                    "latency": end - start,
                    "parent_id": current_parent_id(),
                    "_cache_hit": True,
                    "tokens": None,
                })
                return cached

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
            TraceType.OpenAI,
            api="chat.completions",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                pre_call_guard(kwargs)
            except Exception as sec_exc:
                from fluiq.exceptions import FluiqSecurityError
                if isinstance(sec_exc, FluiqSecurityError):
                    _emit_security_blocked_trace(kwargs, sec_exc, start, time.time(), api="chat.completions")
                raise

            try:
                from fluiq.integrations.shared.tool_cache import learn_from_openai_messages
                learn_from_openai_messages(kwargs.get("messages") or [])
            except Exception:
                pass

            cached = pre_call_optimize(kwargs, "openai")
            if cached is not None:
                end = time.time()
                _payload = getattr(cached, "_fluiq_payload", {})
                log_trace({
                    "type": "llm",
                    "integration": TraceType.OpenAI.value,
                    "api": "chat.completions",
                    "model": kwargs.get("model"),
                    "messages": _to_jsonable(kwargs.get("messages")),
                    "tools": _to_jsonable(kwargs.get("tools")),
                    "response": _payload.get("response"),
                    "tool_calls": _payload.get("tool_calls"),
                    "mcp_calls": _payload.get("mcp_calls"),
                    "mcp_results": _payload.get("mcp_results"),
                    "mcp_servers": _payload.get("mcp_servers"),
                    "latency": end - start,
                    "parent_id": current_parent_id(),
                    "_cache_hit": True,
                    "tokens": None,
                })
                return cached

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
            TraceType.OpenAI,
            api="responses",
            model=kwargs.get("model"),
            input=_to_jsonable(kwargs.get("input")),
        )
        start = time.time()
        try:
            try:
                pre_call_guard(kwargs)
            except Exception as sec_exc:
                from fluiq.exceptions import FluiqSecurityError
                if isinstance(sec_exc, FluiqSecurityError):
                    _emit_security_blocked_trace(kwargs, sec_exc, start, time.time(), api="responses")
                raise

            cached = pre_call_optimize(kwargs, "openai_responses")
            if cached is not None:
                end = time.time()
                _payload = getattr(cached, "_fluiq_payload", {})
                log_trace({
                    "type": "llm",
                    "integration": TraceType.OpenAI.value,
                    "api": "responses",
                    "model": kwargs.get("model"),
                    "input": _to_jsonable(kwargs.get("input")),
                    "tools": _to_jsonable(kwargs.get("tools")),
                    "response": _payload.get("response"),
                    "mcp_calls": _payload.get("mcp_calls"),
                    "mcp_results": _payload.get("mcp_results"),
                    "mcp_servers": _payload.get("mcp_servers"),
                    "latency": end - start,
                    "parent_id": current_parent_id(),
                    "_cache_hit": True,
                    "tokens": None,
                })
                return cached

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
            TraceType.OpenAI,
            api="responses",
            model=kwargs.get("model"),
            input=_to_jsonable(kwargs.get("input")),
        )
        start = time.time()
        try:
            try:
                pre_call_guard(kwargs)
            except Exception as sec_exc:
                from fluiq.exceptions import FluiqSecurityError
                if isinstance(sec_exc, FluiqSecurityError):
                    _emit_security_blocked_trace(kwargs, sec_exc, start, time.time(), api="responses")
                raise

            cached = pre_call_optimize(kwargs, "openai_responses")
            if cached is not None:
                end = time.time()
                _payload = getattr(cached, "_fluiq_payload", {})
                log_trace({
                    "type": "llm",
                    "integration": TraceType.OpenAI.value,
                    "api": "responses",
                    "model": kwargs.get("model"),
                    "input": _to_jsonable(kwargs.get("input")),
                    "tools": _to_jsonable(kwargs.get("tools")),
                    "response": _payload.get("response"),
                    "mcp_calls": _payload.get("mcp_calls"),
                    "mcp_results": _payload.get("mcp_results"),
                    "mcp_servers": _payload.get("mcp_servers"),
                    "latency": end - start,
                    "parent_id": current_parent_id(),
                    "_cache_hit": True,
                    "tokens": None,
                })
                return cached

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
            TraceType.OpenAI,
            api="chat.completions.parse",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                pre_call_guard(kwargs)
            except Exception as sec_exc:
                from fluiq.exceptions import FluiqSecurityError
                if isinstance(sec_exc, FluiqSecurityError):
                    _emit_security_blocked_trace(kwargs, sec_exc, start, time.time(), api="chat.completions.parse")
                raise
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
            TraceType.OpenAI,
            api="chat.completions.parse",
            model=kwargs.get("model"),
            messages=_to_jsonable(kwargs.get("messages")),
        )
        start = time.time()
        try:
            try:
                pre_call_guard(kwargs)
            except Exception as sec_exc:
                from fluiq.exceptions import FluiqSecurityError
                if isinstance(sec_exc, FluiqSecurityError):
                    _emit_security_blocked_trace(kwargs, sec_exc, start, time.time(), api="chat.completions.parse")
                raise
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
            TraceType.OpenAI,
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
            TraceType.OpenAI,
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
