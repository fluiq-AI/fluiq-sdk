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
from fluiq.integrations.Gemini.helper.mcp_trace import (
    _extract_mcp_servers,
    _enrich_mcp_sessions,
)

def _usage_dict(usage):
    if not usage:
        return None
    return {
        "prompt": getattr(usage, "prompt_token_count", None),
        "completion": getattr(usage, "candidates_token_count", None),
        "total": getattr(usage, "total_token_count", None),
    }

def _emit_genai_trace(kwargs, response, start, end):
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
        integration=TraceType.Gemini,
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
        tool_call_latencies=_compute_tool_call_latencies(kwargs.get("contents")),
        finish_reasons=_finish_reasons(candidates),
        latency=end - start,
        parent_id=current_parent_id(),
        tokens=_usage_dict(usage),
    )
    log_trace(payload.model_dump(mode="json"))


def _emit_genai_error(kwargs, error, start, end, api=None, model=None):
    payload = LogTrace(
        type="llm",
        integration=TraceType.Gemini,
        api=api,
        model=model or kwargs.get("model"),
        contents=_to_jsonable(kwargs.get("contents")),
        tools=_to_jsonable(kwargs.get("tools")),
        output=str(error),
        error_traceback=format_error_traceback(error),
        latency=end - start,
        parent_id=current_parent_id(),
        success=False,
    )
    log_trace(payload.model_dump(mode="json"))


def patch_genai():
    from google.genai.models import Models

    original = Models.generate_content

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Gemini,
            api="generate_content",
            model=kwargs.get("model"),
            contents=_to_jsonable(kwargs.get("contents")),
        )
        start = time.time()
        try:
            try:
                response = original(self, *args, **kwargs)
            except Exception as e:
                _emit_genai_error(kwargs, e, start, time.time())
                raise
            end = time.time()
            _emit_genai_trace(kwargs, response, start, end)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    Models.generate_content = wrapped


def patch_genai_async():
    from google.genai.models import AsyncModels

    original = AsyncModels.generate_content

    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)
        _gc_pending_tool_calls()
        await _enrich_mcp_sessions(kwargs)
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Gemini,
            api="generate_content",
            model=kwargs.get("model"),
            contents=_to_jsonable(kwargs.get("contents")),
        )
        start = time.time()
        try:
            try:
                response = await original(self, *args, **kwargs)
            except Exception as e:
                _emit_genai_error(kwargs, e, start, time.time())
                raise
            end = time.time()
            _emit_genai_trace(kwargs, response, start, end)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    AsyncModels.generate_content = wrapped


class _GenaiStreamAggregator:
    def __init__(self):
        self._chunks = []
        self._latest = None

    def feed(self, chunk):
        self._chunks.append(chunk)
        self._latest = chunk

    def assemble(self):
        return self._latest, self._chunks


def patch_genai_stream():
    from google.genai.models import Models
    original = Models.generate_content_stream

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            yield from original(self, *args, **kwargs)
            return
        _gc_pending_tool_calls()
        agg = _GenaiStreamAggregator()
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Gemini,
            api="generate_content_stream",
            model=kwargs.get("model"),
            contents=_to_jsonable(kwargs.get("contents")),
        )
        start = time.time()
        errored = False
        try:
            try:
                for chunk in original(self, *args, **kwargs):
                    agg.feed(chunk)
                    yield chunk
            except Exception as e:
                errored = True
                _emit_genai_error(kwargs, e, start, time.time(), api="generate_content_stream")
                raise
            finally:
                if not errored:
                    end = time.time()
                    latest, _ = agg.assemble()
                    if latest is not None:
                        _emit_genai_trace(kwargs, latest, start, end)
        finally:
            pop_llm_trace_id(ctx_tok)

    Models.generate_content_stream = wrapped


def patch_genai_stream_async():
    from google.genai.models import AsyncModels
    original = AsyncModels.generate_content_stream

    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            async for chunk in original(self, *args, **kwargs):
                yield chunk
            return
        _gc_pending_tool_calls()
        await _enrich_mcp_sessions(kwargs)
        agg = _GenaiStreamAggregator()
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Gemini,
            api="generate_content_stream",
            model=kwargs.get("model"),
            contents=_to_jsonable(kwargs.get("contents")),
        )
        start = time.time()
        errored = False
        try:
            try:
                async for chunk in original(self, *args, **kwargs):
                    agg.feed(chunk)
                    yield chunk
            except Exception as e:
                errored = True
                _emit_genai_error(kwargs, e, start, time.time(), api="generate_content_stream")
                raise
            finally:
                if not errored:
                    end = time.time()
                    latest, _ = agg.assemble()
                    if latest is not None:
                        _emit_genai_trace(kwargs, latest, start, end)
        finally:
            pop_llm_trace_id(ctx_tok)

    AsyncModels.generate_content_stream = wrapped

def _emit_vertex_trace(self, kwargs, request_contents, response, start, end, tool_call_latencies):
    usage = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", None)
    _record_dispatched_tool_calls(candidates)
    thinking = _extract_thinking(candidates)
    text = _strip_media(candidates)
    function_calls = _extract_function_calls(candidates)
    tools, tool_config = _extract_request_tools(kwargs, instance=self)
    mcp_servers = _extract_mcp_servers(kwargs)
    model = getattr(self, "_model_name", None) or getattr(self, "model_name", None)

    payload = LogTrace(
        type="llm",
        integration=TraceType.Gemini,
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
        parent_id=current_parent_id(),
        tokens=_usage_dict(usage),
    )
    log_trace(payload.model_dump(mode="json"))


def _vertex_stream_passthrough(self, kwargs, request_contents, tool_call_latencies, stream, start, trace_id=None):
    agg = _GenaiStreamAggregator()
    errored = False
    try:
        for chunk in stream:
            agg.feed(chunk)
            yield chunk
    except Exception as e:
        errored = True
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_vertex_error(self, kwargs, request_contents, e, start, time.time())
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)
        raise
    finally:
        if not errored:
            end = time.time()
            latest, _ = agg.assemble()
            if latest is not None:
                tok = push_llm_trace_id(trace_id) if trace_id else None
                try:
                    _emit_vertex_trace(self, kwargs, request_contents, latest, start, end, tool_call_latencies)
                finally:
                    if tok is not None:
                        pop_llm_trace_id(tok)


async def _vertex_async_stream_passthrough(self, kwargs, request_contents, tool_call_latencies, stream, start, trace_id=None):
    agg = _GenaiStreamAggregator()
    errored = False
    try:
        async for chunk in stream:
            agg.feed(chunk)
            yield chunk
    except Exception as e:
        errored = True
        tok = push_llm_trace_id(trace_id) if trace_id else None
        try:
            _emit_vertex_error(self, kwargs, request_contents, e, start, time.time())
        finally:
            if tok is not None:
                pop_llm_trace_id(tok)
        raise
    finally:
        if not errored:
            end = time.time()
            latest, _ = agg.assemble()
            if latest is not None:
                tok = push_llm_trace_id(trace_id) if trace_id else None
                try:
                    _emit_vertex_trace(self, kwargs, request_contents, latest, start, end, tool_call_latencies)
                finally:
                    if tok is not None:
                        pop_llm_trace_id(tok)


def _emit_vertex_error(self, kwargs, request_contents, error, start, end):
    model = getattr(self, "_model_name", None) or getattr(self, "model_name", None)
    payload = LogTrace(
        type="llm",
        integration=TraceType.Gemini,
        model=model,
        contents=_to_jsonable(request_contents),
        output=str(error),
        error_traceback=format_error_traceback(error),
        latency=end - start,
        parent_id=current_parent_id(),
        success=False,
    )
    log_trace(payload.model_dump(mode="json"))


def patch_vertexai():
    from vertexai.generative_models import GenerativeModel
    original = GenerativeModel.generate_content

    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)

        _gc_pending_tool_calls()
        request_contents = args[0] if args else kwargs.get("contents")
        tool_call_latencies = _compute_tool_call_latencies(request_contents)

        model = getattr(self, "_model_name", None) or getattr(self, "model_name", None)
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Gemini,
            api="vertex.generate_content",
            model=model,
            contents=_to_jsonable(request_contents),
        )
        start = time.time()
        try:
            try:
                response = original(self, *args, **kwargs)
            except Exception as e:
                _emit_vertex_error(self, kwargs, request_contents, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _vertex_stream_passthrough(self, kwargs, request_contents, tool_call_latencies, response, start, trace_id=trace_id)

            end = time.time()
            _emit_vertex_trace(self, kwargs, request_contents, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    GenerativeModel.generate_content = wrapped


def patch_vertexai_async():
    from vertexai.generative_models import GenerativeModel
    if not hasattr(GenerativeModel, "generate_content_async"):
        return
    original = GenerativeModel.generate_content_async

    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)

        _gc_pending_tool_calls()
        request_contents = args[0] if args else kwargs.get("contents")
        tool_call_latencies = _compute_tool_call_latencies(request_contents)

        model = getattr(self, "_model_name", None) or getattr(self, "model_name", None)
        trace_id, ctx_tok = emit_llm_start(
            TraceType.Gemini,
            api="vertex.generate_content",
            model=model,
            contents=_to_jsonable(request_contents),
        )
        start = time.time()
        try:
            try:
                response = await original(self, *args, **kwargs)
            except Exception as e:
                _emit_vertex_error(self, kwargs, request_contents, e, start, time.time())
                raise

            if kwargs.get("stream"):
                return _vertex_async_stream_passthrough(self, kwargs, request_contents, tool_call_latencies, response, start, trace_id=trace_id)

            end = time.time()
            _emit_vertex_trace(self, kwargs, request_contents, response, start, end, tool_call_latencies)
            return response
        finally:
            pop_llm_trace_id(ctx_tok)

    GenerativeModel.generate_content_async = wrapped


def _emit_count_tokens_trace(integration, kwargs, contents, response, start, end, model=None):
    total = getattr(response, "total_tokens", None)
    payload = LogTrace(
        type="llm",
        integration=integration,
        api="count_tokens",
        model=model or kwargs.get("model"),
        contents=_to_jsonable(contents),
        response=_to_jsonable(response),
        latency=end - start,
        parent_id=current_parent_id(),
        tokens={"prompt": total, "completion": None, "total": total} if total is not None else None,
    )
    log_trace(payload.model_dump(mode="json"))


def _emit_count_tokens_error(integration, kwargs, contents, error, start, end, model=None):
    payload = LogTrace(
        type="llm",
        integration=integration,
        api="count_tokens",
        model=model or kwargs.get("model"),
        contents=_to_jsonable(contents),
        output=str(error),
        error_traceback=format_error_traceback(error),
        latency=end - start,
        parent_id=current_parent_id(),
        success=False,
    )
    log_trace(payload.model_dump(mode="json"))


def patch_genai_count_tokens():
    from google.genai.models import Models
    if not hasattr(Models, "count_tokens"):
        return
    original = Models.count_tokens

    def wrapped(self, *args, **kwargs):
        start = time.time()
        try:
            response = original(self, *args, **kwargs)
        except Exception as e:
            _emit_count_tokens_error(TraceType.Gemini, kwargs, kwargs.get("contents"), e, start, time.time())
            raise
        end = time.time()
        _emit_count_tokens_trace(TraceType.Gemini, kwargs, kwargs.get("contents"), response, start, end)
        return response

    Models.count_tokens = wrapped


def patch_genai_count_tokens_async():
    from google.genai.models import AsyncModels
    if not hasattr(AsyncModels, "count_tokens"):
        return
    original = AsyncModels.count_tokens

    async def wrapped(self, *args, **kwargs):
        start = time.time()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as e:
            _emit_count_tokens_error(TraceType.Gemini, kwargs, kwargs.get("contents"), e, start, time.time())
            raise
        end = time.time()
        _emit_count_tokens_trace(TraceType.Gemini, kwargs, kwargs.get("contents"), response, start, end)
        return response

    AsyncModels.count_tokens = wrapped


def patch_vertexai_count_tokens():
    from vertexai.generative_models import GenerativeModel
    if not hasattr(GenerativeModel, "count_tokens"):
        return
    original = GenerativeModel.count_tokens

    def wrapped(self, *args, **kwargs):
        contents = args[0] if args else kwargs.get("contents")
        model = getattr(self, "_model_name", None) or getattr(self, "model_name", None)
        start = time.time()
        try:
            response = original(self, *args, **kwargs)
        except Exception as e:
            _emit_count_tokens_error(TraceType.Gemini, kwargs, contents, e, start, time.time(), model=model)
            raise
        end = time.time()
        _emit_count_tokens_trace(TraceType.Gemini, kwargs, contents, response, start, end, model=model)
        return response

    GenerativeModel.count_tokens = wrapped


def patch_vertexai_count_tokens_async():
    from vertexai.generative_models import GenerativeModel
    if not hasattr(GenerativeModel, "count_tokens_async"):
        return
    original = GenerativeModel.count_tokens_async

    async def wrapped(self, *args, **kwargs):
        contents = args[0] if args else kwargs.get("contents")
        model = getattr(self, "_model_name", None) or getattr(self, "model_name", None)
        start = time.time()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as e:
            _emit_count_tokens_error(TraceType.Gemini, kwargs, contents, e, start, time.time(), model=model)
            raise
        end = time.time()
        _emit_count_tokens_trace(TraceType.Gemini, kwargs, contents, response, start, end, model=model)
        return response

    GenerativeModel.count_tokens_async = wrapped
