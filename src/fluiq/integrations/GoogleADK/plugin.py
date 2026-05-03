import time
import uuid

from google.adk.plugins.base_plugin import BasePlugin

from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import (
    current_parent_id,
    push_trace_id,
    pop_trace_id,
    format_error_traceback,
)
from fluiq.integrations.GoogleADK.helper.utils import (
    _to_jsonable,
    _content_to_text,
    _invocation_id,
    _user_message,
    _agent_model,
    _tool_input_schema,
    _collect_output_keys,
    _state_snapshot,
    _agent_output,
)


class FluiqADKPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="fluiq")
        self._agents = {}
        self._tools = {}

    def _emit(self, payload):
        try:
            log_trace(payload.model_dump(mode="json"))
        except Exception:
            pass

    def _emit_start(self, **kwargs):
        # Lightweight live-progress signal: same trace_id as the eventual
        # completion emission so the frontend can replace the running row in
        # place. No latency / output / tokens — those land on completion.
        kwargs.setdefault("integration", TraceType.GoogleADK)
        kwargs.setdefault("status", "running")
        kwargs.setdefault("started_at", time.time())
        try:
            self._emit(LogTrace(**kwargs))
        except Exception:
            pass

    @staticmethod
    def _agent_key(agent, callback_context):
        # ADK builds a fresh CallbackContext for before_/after_agent_callback
        # (see google.adk.agents.base_agent._handle_*_agent_callback), so
        # id(callback_context) differs between the two calls. Key by the
        # (invocation_id, agent_name) pair which is stable across both and
        # unique within a run (sub-agents have distinct names).
        return (_invocation_id(callback_context), getattr(agent, "name", None))

    async def before_agent_callback(self, *, agent, callback_context):
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        token = push_trace_id(trace_id)
        output_keys = _collect_output_keys(agent)
        invocation_id = _invocation_id(callback_context)
        agent_name = getattr(agent, "name", None)
        model = _agent_model(agent)
        user_input = _user_message(callback_context)
        self._agents[self._agent_key(agent, callback_context)] = {
            "trace_id": trace_id,
            "parent_id": parent_id,
            "start": time.time(),
            "agent_name": agent_name,
            "model": model,
            "input": user_input,
            "_token": token,
            "invocation_id": invocation_id,
            "output_keys": output_keys,
            "state_before": _state_snapshot(callback_context, output_keys),
        }
        self._emit_start(
            type="agent",
            function=agent_name,
            model=model,
            input=user_input,
            trace_id=trace_id,
            parent_id=parent_id,
            invocation_id=invocation_id,
        )
        return None

    async def after_agent_callback(self, *, agent, callback_context):
        state = self._agents.pop(self._agent_key(agent, callback_context), None)
        end = time.time()
        if state is None:
            return None
        pop_trace_id(state.get("_token"))
        output = _agent_output(
            callback_context,
            state.get("output_keys") or [],
            state.get("state_before") or {},
        )
        if output is None:
            # No output_key declared (or none of this agent's descendants
            # write to state): fall back to the last LLM response captured
            # under this agent via after_model_callback.
            output = state.get("last_response")
        self._emit(LogTrace(
            type="agent",
            integration=TraceType.GoogleADK,
            function=state.get("agent_name"),
            model=state.get("model"),
            input=state.get("input"),
            output=output,
            latency=end - state.get("start", end),
            trace_id=state.get("trace_id"),
            parent_id=state.get("parent_id"),
            invocation_id=state.get("invocation_id"),
            success=True,
        ))
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        # Capture the model's reply text as a fallback agent output for
        # agents that don't declare an output_key. Stored on the in-flight
        # entry of the calling LlmAgent and bubbled up to all ancestor
        # wrapper agents in the same invocation so a LoopAgent /
        # SequentialAgent without output_keys still surfaces something.
        text = _content_to_text(getattr(llm_response, "content", None))
        if not text:
            return None
        invocation_id = _invocation_id(callback_context)
        for (inv, _name), st in self._agents.items():
            if inv == invocation_id:
                st["last_response"] = text
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        tool_name = getattr(tool, "name", None)
        tool_input = _to_jsonable(tool_args)
        self._tools[id(tool_context)] = {
            "trace_id": trace_id,
            "parent_id": parent_id,
            "start": time.time(),
            "name": tool_name,
            "description": getattr(tool, "description", None),
            "input": tool_input,
            "schema": _tool_input_schema(tool),
        }
        self._emit_start(
            type="tool",
            function=tool_name,
            input=tool_input,
            trace_id=trace_id,
            parent_id=parent_id,
        )
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        state = self._tools.pop(id(tool_context), None)
        end = time.time()
        if state is None:
            return None
        self._emit(LogTrace(
            type="tool",
            integration=TraceType.GoogleADK,
            function=state.get("name"),
            input=state.get("input"),
            output=_to_jsonable(result),
            tools=[{
                "name": state.get("name"),
                "description": state.get("description"),
                "input_schema": state.get("schema"),
            }] if state.get("name") else None,
            latency=end - state.get("start", end),
            trace_id=state.get("trace_id"),
            parent_id=state.get("parent_id"),
            success=True,
        ))
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        state = self._tools.pop(id(tool_context), None)
        end = time.time()
        if state is None:
            return None
        self._emit(LogTrace(
            type="tool",
            integration=TraceType.GoogleADK,
            function=state.get("name"),
            input=state.get("input"),
            output=str(error),
            error_traceback=format_error_traceback(error),
            latency=end - state.get("start", end),
            trace_id=state.get("trace_id"),
            parent_id=state.get("parent_id"),
            success=False,
        ))
        return None

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        # Inner google.genai LLM call failed during an ADK agent run. The
        # Gemini patch wraps generate_content/generate_content_stream and emits
        # its own failure trace; this callback is a safety net for cases where
        # the error is raised before the patch can capture it (e.g. the ADK
        # flow short-circuits on LlmRequest assembly). Emits a minimal failed
        # llm trace nested under the active agent so the failure shows up in
        # the dashboard immediately.
        key = (_invocation_id(callback_context), getattr(callback_context, "agent_name", None))
        agent_state = self._agents.get(key)
        parent_id = agent_state.get("trace_id") if agent_state else current_parent_id()
        self._emit(LogTrace(
            type="llm",
            integration=TraceType.GoogleADK,
            model=getattr(llm_request, "model", None),
            output=str(error),
            error_traceback=format_error_traceback(error),
            trace_id=str(uuid.uuid4()),
            parent_id=parent_id,
            success=False,
        ))
        return None

    async def after_run_callback(self, *, invocation_context):
        # ADK's BasePlugin has no on_agent_error_callback. When an agent
        # raises mid-run, after_agent_callback is skipped and the entry stays
        # in self._agents. after_run_callback is invoked for every Runner.run
        # regardless of outcome, so we sweep any unclosed agent entries
        # belonging to this invocation and emit them as failed traces.
        invocation_id = getattr(invocation_context, "invocation_id", None)
        if invocation_id is None:
            return None
        end = time.time()
        stale = [
            (key, st) for key, st in list(self._agents.items())
            if st.get("invocation_id") == str(invocation_id)
        ]
        for key, state in stale:
            self._agents.pop(key, None)
            pop_trace_id(state.get("_token"))
            self._emit(LogTrace(
                type="agent",
                integration=TraceType.GoogleADK,
                function=state.get("agent_name"),
                model=state.get("model"),
                input=state.get("input"),
                output="agent did not complete (no after_agent_callback fired)",
                latency=end - state.get("start", end),
                trace_id=state.get("trace_id"),
                parent_id=state.get("parent_id"),
                invocation_id=state.get("invocation_id"),
                success=False,
            ))
        return None
