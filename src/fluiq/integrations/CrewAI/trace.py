import time
import uuid

from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import (
    push_trace_id,
    pop_trace_id,
    current_parent_id,
    format_error_traceback,
)
from fluiq.integrations.CrewAI.helper.utils import (
    _to_jsonable,
    _crew_name,
    _agent_name,
    _task_description,
    _tool_name,
    _extract_crew_output,
    _extract_task_output,
    _extract_token_usage,
)

_patched = False


# ---------------------------------------------------------------------------
# Internal emit helpers
# ---------------------------------------------------------------------------

def _emit(*, type_, trace_id, parent_id, function, input_=None, output=None,
          latency=None, success, tokens=None, error=None, status=None):
    try:
        kwargs = dict(
            integration=TraceType.CrewAI,
            type=type_,
            trace_id=trace_id,
            parent_id=parent_id,
            function=function,
            input=input_,
            output=output if error is None else str(error),
            latency=latency,
            success=success,
            tokens=tokens,
        )
        if status is not None:
            kwargs["status"] = status
        if error is not None:
            kwargs["error_traceback"] = format_error_traceback(error)
        log_trace(LogTrace(**kwargs).model_dump(mode="json"))
    except Exception:
        pass


def _emit_start(*, type_, trace_id, parent_id, function, input_=None):
    """Lightweight running signal so the frontend can show in-progress rows."""
    try:
        log_trace(LogTrace(
            integration=TraceType.CrewAI,
            type=type_,
            trace_id=trace_id,
            parent_id=parent_id,
            function=function,
            input=input_,
            status="running",
            started_at=time.time(),
        ).model_dump(mode="json"))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Crew.kickoff patch
# ---------------------------------------------------------------------------

def _patch_crew(Crew):
    original_kickoff = Crew.kickoff
    original_kickoff_async = getattr(Crew, "kickoff_async", None)

    def wrapped_kickoff(self, inputs=None, input_files=None, **kwargs):
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        name = _crew_name(self)
        start = time.time()

        _emit_start(type_="crew", trace_id=trace_id, parent_id=parent_id,
                    function=name, input_=_to_jsonable(inputs))

        ctx_tok = push_trace_id(trace_id)
        try:
            result = original_kickoff(self, inputs=inputs, input_files=input_files, **kwargs)
            _emit(
                type_="crew",
                trace_id=trace_id,
                parent_id=parent_id,
                function=name,
                input_=_to_jsonable(inputs),
                output=_extract_crew_output(result),
                latency=time.time() - start,
                success=True,
                tokens=_extract_token_usage(result),
            )
            return result
        except Exception as e:
            _emit(
                type_="crew",
                trace_id=trace_id,
                parent_id=parent_id,
                function=name,
                input_=_to_jsonable(inputs),
                latency=time.time() - start,
                success=False,
                error=e,
            )
            return None
        finally:
            pop_trace_id(ctx_tok)

    Crew.kickoff = wrapped_kickoff

    if original_kickoff_async is not None:
        async def wrapped_kickoff_async(self, inputs=None, input_files=None, **kwargs):
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            name = _crew_name(self)
            start = time.time()

            _emit_start(type_="crew", trace_id=trace_id, parent_id=parent_id,
                        function=name, input_=_to_jsonable(inputs))

            ctx_tok = push_trace_id(trace_id)
            try:
                result = await original_kickoff_async(self, inputs=inputs, input_files=input_files, **kwargs)
                _emit(
                    type_="crew",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=name,
                    input_=_to_jsonable(inputs),
                    output=_extract_crew_output(result),
                    latency=time.time() - start,
                    success=True,
                    tokens=_extract_token_usage(result),
                )
                return result
            except Exception as e:
                _emit(
                    type_="crew",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=name,
                    input_=_to_jsonable(inputs),
                    latency=time.time() - start,
                    success=False,
                    error=e,
                )
                return None
            finally:
                pop_trace_id(ctx_tok)

        Crew.kickoff_async = wrapped_kickoff_async


# ---------------------------------------------------------------------------
# Task.execute_sync / execute_async patch
# ---------------------------------------------------------------------------

def _patch_task(Task):
    original_execute_sync = getattr(Task, "execute_sync", None)
    original_execute_async = getattr(Task, "execute_async", None)

    if original_execute_sync is not None:
        def wrapped_execute_sync(self, agent=None, context=None, tools=None):
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            desc = _task_description(self)
            start = time.time()

            _emit_start(type_="task", trace_id=trace_id, parent_id=parent_id,
                        function=desc, input_=context)

            ctx_tok = push_trace_id(trace_id)
            try:
                result = original_execute_sync(self, agent=agent, context=context, tools=tools)
                _emit(
                    type_="task",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=desc,
                    input_=context,
                    output=_extract_task_output(result),
                    latency=time.time() - start,
                    success=True,
                )
                return result
            except Exception as e:
                _emit(
                    type_="task",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=desc,
                    input_=context,
                    latency=time.time() - start,
                    success=False,
                    error=e,
                )
                return None
            finally:
                pop_trace_id(ctx_tok)

        Task.execute_sync = wrapped_execute_sync

    if original_execute_async is not None:
        async def wrapped_execute_async(self, agent=None, context=None, tools=None):
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            desc = _task_description(self)
            start = time.time()

            _emit_start(type_="task", trace_id=trace_id, parent_id=parent_id,
                        function=desc, input_=context)

            ctx_tok = push_trace_id(trace_id)
            try:
                result = await original_execute_async(self, agent=agent, context=context, tools=tools)
                _emit(
                    type_="task",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=desc,
                    input_=context,
                    output=_extract_task_output(result),
                    latency=time.time() - start,
                    success=True,
                )
                return result
            except Exception as e:
                _emit(
                    type_="task",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=desc,
                    input_=context,
                    latency=time.time() - start,
                    success=False,
                    error=e,
                )
                return None
            finally:
                pop_trace_id(ctx_tok)

        Task.execute_async = wrapped_execute_async


# ---------------------------------------------------------------------------
# Agent.execute_task patch
# ---------------------------------------------------------------------------

def _patch_agent(Agent):
    original_execute_task = getattr(Agent, "execute_task", None)
    if original_execute_task is None:
        return

    def wrapped_execute_task(self, task, context=None, tools=None):
        trace_id = str(uuid.uuid4())
        parent_id = current_parent_id()
        name = _agent_name(self)
        desc = _task_description(task)
        start = time.time()

        _emit_start(type_="agent", trace_id=trace_id, parent_id=parent_id,
                    function=name, input_=desc)

        ctx_tok = push_trace_id(trace_id)
        try:
            result = original_execute_task(self, task, context=context, tools=tools)
            _emit(
                type_="agent",
                trace_id=trace_id,
                parent_id=parent_id,
                function=name,
                input_=desc,
                output=result if isinstance(result, str) else _to_jsonable(result),
                latency=time.time() - start,
                success=True,
            )
            return result
        except Exception as e:
            _emit(
                type_="agent",
                trace_id=trace_id,
                parent_id=parent_id,
                function=name,
                input_=desc,
                latency=time.time() - start,
                success=False,
                error=e,
            )
            return None
        finally:
            pop_trace_id(ctx_tok)

    Agent.execute_task = wrapped_execute_task


# ---------------------------------------------------------------------------
# BaseTool._run / _run_async patch
# ---------------------------------------------------------------------------

def _patch_tool(BaseTool):
    original_run = getattr(BaseTool, "_run", None)
    original_run_async = getattr(BaseTool, "_run_async", None)

    if original_run is not None:
        def wrapped_run(self, *args, **kwargs):
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            name = _tool_name(self)
            input_ = kwargs if kwargs else (args[0] if args else None)
            start = time.time()

            _emit_start(type_="tool", trace_id=trace_id, parent_id=parent_id,
                        function=name, input_=_to_jsonable(input_))

            try:
                result = original_run(self, *args, **kwargs)
                _emit(
                    type_="tool",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=name,
                    input_=_to_jsonable(input_),
                    output=result if isinstance(result, str) else _to_jsonable(result),
                    latency=time.time() - start,
                    success=True,
                )
                return result
            except Exception as e:
                _emit(
                    type_="tool",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=name,
                    input_=_to_jsonable(input_),
                    latency=time.time() - start,
                    success=False,
                    error=e,
                )
                return None

        BaseTool._run = wrapped_run

    if original_run_async is not None:
        async def wrapped_run_async(self, *args, **kwargs):
            trace_id = str(uuid.uuid4())
            parent_id = current_parent_id()
            name = _tool_name(self)
            input_ = kwargs if kwargs else (args[0] if args else None)
            start = time.time()

            _emit_start(type_="tool", trace_id=trace_id, parent_id=parent_id,
                        function=name, input_=_to_jsonable(input_))

            try:
                result = await original_run_async(self, *args, **kwargs)
                _emit(
                    type_="tool",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=name,
                    input_=_to_jsonable(input_),
                    output=result if isinstance(result, str) else _to_jsonable(result),
                    latency=time.time() - start,
                    success=True,
                )
                return result
            except Exception as e:
                _emit(
                    type_="tool",
                    trace_id=trace_id,
                    parent_id=parent_id,
                    function=name,
                    input_=_to_jsonable(input_),
                    latency=time.time() - start,
                    success=False,
                    error=e,
                )
                return None

        BaseTool._run_async = wrapped_run_async


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def patch_crewai():
    global _patched
    if _patched:
        return

    try:
        from crewai import Crew
        _patch_crew(Crew)
    except Exception:
        pass

    try:
        from crewai import Task
        _patch_task(Task)
    except Exception:
        pass

    try:
        from crewai import Agent
        _patch_agent(Agent)
    except Exception:
        pass

    try:
        from crewai.tools import BaseTool
        _patch_tool(BaseTool)
    except Exception:
        pass

    _patched = True