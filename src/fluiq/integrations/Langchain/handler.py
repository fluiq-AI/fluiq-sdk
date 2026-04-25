import time
from langchain_core.callbacks.base import BaseCallbackHandler

from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import enter_langchain_llm, exit_langchain_llm
from fluiq.integrations.Langchain.helper.utils import (
    _to_jsonable,
    _model_name,
    _component_name,
    _extract_tokens,
    _extract_response_text,
    _extract_response_model,
    _extract_finish_reason,
    _extract_tool_calls,
)


class FluiqCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self._runs = {}

    def _start(self, run_id, **state):
        self._runs[run_id] = {"start": time.time(), **state}

    def _end(self, run_id):
        state = self._runs.pop(run_id, None) or {}
        return state, time.time()

    def _emit(self, payload):
        try:
            log_trace(payload.model_dump(mode="json"))
        except Exception:
            pass

    def _parent(self, parent_run_id):
        return str(parent_run_id) if parent_run_id else None

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None,
                     tags=None, metadata=None, **kwargs):
        token = enter_langchain_llm()
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            model=_model_name(serialized, kwargs.get("invocation_params"), metadata),
            prompts=prompts,
            tools=kwargs.get("invocation_params", {}).get("tools"),
            _lc_token=token,
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                            tags=None, metadata=None, **kwargs):
        token = enter_langchain_llm()
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            model=_model_name(serialized, kwargs.get("invocation_params"), metadata),
            messages=messages,
            tools=kwargs.get("invocation_params", {}).get("tools"),
            _lc_token=token,
        )

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        exit_langchain_llm(state.get("_lc_token"))
        self._emit(LogTrace(
            type="llm",
            integration=TraceType.LangChain,
            model=state.get("model") or _extract_response_model(response),
            messages=_to_jsonable(state.get("messages")),
            input=_to_jsonable(state.get("prompts")),
            tools=_to_jsonable(state.get("tools")),
            response=_extract_response_text(response),
            tool_calls=_extract_tool_calls(response),
            tokens=_extract_tokens(response),
            finish_reasons=[_extract_finish_reason(response)] if _extract_finish_reason(response) else None,
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=True,
        ))

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        exit_langchain_llm(state.get("_lc_token"))
        self._emit(LogTrace(
            type="llm",
            integration=TraceType.LangChain,
            model=state.get("model"),
            messages=_to_jsonable(state.get("messages")),
            input=_to_jsonable(state.get("prompts")),
            output=str(error),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=False,
        ))

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       tags=None, metadata=None, **kwargs):
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            name=_component_name(serialized),
            inputs=inputs,
        )

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit(LogTrace(
            type="chain",
            integration=TraceType.LangChain,
            function=state.get("name"),
            input=_to_jsonable(state.get("inputs")),
            output=_to_jsonable(outputs),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=True,
        ))

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit(LogTrace(
            type="chain",
            integration=TraceType.LangChain,
            function=state.get("name"),
            input=_to_jsonable(state.get("inputs")),
            output=str(error),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=False,
        ))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      tags=None, metadata=None, inputs=None, **kwargs):
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            name=_component_name(serialized),
            input=input_str,
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit(LogTrace(
            type="tool",
            integration=TraceType.LangChain,
            function=state.get("name"),
            input=state.get("input"),
            output=_to_jsonable(output),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=True,
        ))

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit(LogTrace(
            type="tool",
            integration=TraceType.LangChain,
            function=state.get("name"),
            input=state.get("input"),
            output=str(error),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=False,
        ))
