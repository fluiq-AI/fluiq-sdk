import time
from langchain_core.callbacks.base import BaseCallbackHandler

from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import (
    enter_langchain_llm,
    exit_langchain_llm,
    current_parent_id,
    format_error_traceback,
)
from fluiq.integrations.shared.security_gate import pre_call_guard
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
from fluiq.integrations.Langchain.helper.langgraph_edges import (
    current_graph_name,
    predecessor_names,
    resolve_join_parents,
    resolve_join_parents_by_edges,
)


_LANGGRAPH_META_KEYS = (
    "langgraph_node",
    "langgraph_step",
    "langgraph_path",
    "langgraph_triggers",
    "langgraph_checkpoint_ns",
    "thread_id",
)


def _langgraph_meta(metadata):
    if not isinstance(metadata, dict):
        return None
    extracted = {k: metadata[k] for k in _LANGGRAPH_META_KEYS if k in metadata}
    return extracted or None


def _integration_for(metadata):
    return TraceType.LangGraph if _langgraph_meta(metadata) else TraceType.LangChain


class FluiqCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self._runs = {}
        # Per-graph-run registry of {node_name: run_id}, used to turn a join
        # node's langgraph_triggers back into predecessor run_ids (parent_ids).
        self._lg_nodes = {}

    def _start(self, run_id, **state):
        self._runs[run_id] = {"start": time.time(), **state}

    # ── LangGraph DAG (fan-in / join) tracking ────────────────────────────────
    def _lg_thread_key(self, metadata, parent_run_id):
        """Scope the node registry to a single graph invocation. thread_id is
        stable per run; else the shared container run_id groups sibling nodes."""
        if isinstance(metadata, dict) and metadata.get("thread_id"):
            return f"t:{metadata['thread_id']}"
        return f"p:{parent_run_id}" if parent_run_id else "p:root"

    def _register_lg_node(self, tkey, node_name, run_id):
        if not tkey or not node_name:
            return
        reg = self._lg_nodes.get(tkey)
        if reg is None:
            if len(self._lg_nodes) > 256:  # bound memory; fail-open
                self._lg_nodes.clear()
            reg = {}
            self._lg_nodes[tkey] = reg
        reg[node_name] = str(run_id)

    def _lg_join_parents(self, metadata, run_id, parent_run_id):
        """Resolve multi-parent ids for a LangGraph join node (or None)."""
        lg = _langgraph_meta(metadata)
        node_name = lg.get("langgraph_node") if lg else None
        if not node_name:
            return None
        tkey = self._lg_thread_key(metadata, parent_run_id)
        registry = self._lg_nodes.get(tkey) or {}
        # Prefer the static edge graph captured at compile (reliable regardless
        # of LangGraph's trigger encoding); fall back to trigger-name matching
        # for other frameworks / versions that name source nodes in triggers.
        # Resolve against predecessors already registered, then record self.
        parent_ids = resolve_join_parents_by_edges(registry, node_name)
        if parent_ids is None:
            parent_ids = resolve_join_parents(
                registry, node_name, lg.get("langgraph_triggers"),
            )
        self._register_lg_node(tkey, node_name, run_id)
        return parent_ids

    def _lg_predecessors(self, metadata):
        """Static predecessor node names for this LangGraph node (or None)."""
        lg = _langgraph_meta(metadata)
        node_name = lg.get("langgraph_node") if lg else None
        if not node_name:
            return None
        return predecessor_names(node_name) or None

    def _end(self, run_id):
        state = self._runs.pop(run_id, None) or {}
        return state, time.time()

    def _emit(self, **fields):
        try:
            log_trace(LogTrace(**fields).model_dump(mode="json"))
        except Exception:
            pass

    def _emit_kwargs(self, state, **kwargs):
        meta = state.get("metadata")
        lg = _langgraph_meta(meta)
        kwargs.setdefault("integration", _integration_for(meta))
        if lg is not None:
            # Stamp the node's static predecessors (node names) so the dashboard
            # can draw the real DAG, including the fan-out edges that parent_ids
            # (fan-in only) can't express.
            preds = state.get("lg_predecessors")
            if preds:
                lg = {**lg, "predecessors": list(preds)}
            kwargs.setdefault("langgraph", lg)
        self._emit(**kwargs)

    def _emit_start(self, *, run_id, parent_run_id, metadata, type, **kwargs):
        # Lightweight live-progress signal: same trace_id as the eventual
        # `_end` emission so the frontend can replace the running row in
        # place. No latency / output / tokens — those land on completion.
        lg = _langgraph_meta(metadata)
        payload_kwargs = {
            "type": type,
            "trace_id": str(run_id),
            "parent_id": self._parent(parent_run_id),
            "integration": _integration_for(metadata),
            "status": "running",
            "started_at": time.time(),
            **kwargs,
        }
        if lg is not None:
            payload_kwargs.setdefault("langgraph", lg)
        self._emit(**payload_kwargs)

    def _parent(self, parent_run_id):
        if parent_run_id:
            return str(parent_run_id)
        return current_parent_id()

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None,
                     tags=None, metadata=None, **kwargs):
        if prompts:
            pre_call_guard({"prompt": "\n".join(p for p in prompts if isinstance(p, str))})
        token = enter_langchain_llm()
        model = _model_name(serialized, kwargs.get("invocation_params"), metadata)
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            model=model,
            prompts=prompts,
            tools=kwargs.get("invocation_params", {}).get("tools"),
            metadata=metadata,
            _lc_token=token,
        )
        self._emit_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            type="llm",
            model=model,
            input=_to_jsonable(prompts),
        )

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                            tags=None, metadata=None, **kwargs):
        flat = []
        for msg_list in messages:
            for msg in (msg_list if isinstance(msg_list, list) else [msg_list]):
                content = getattr(msg, "content", "") or ""
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                    )
                flat.append({"role": "user", "content": str(content)})
        if flat:
            pre_call_guard({"messages": flat})
        token = enter_langchain_llm()
        model = _model_name(serialized, kwargs.get("invocation_params"), metadata)
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            model=model,
            messages=messages,
            tools=kwargs.get("invocation_params", {}).get("tools"),
            metadata=metadata,
            _lc_token=token,
        )
        self._emit_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            type="llm",
            model=model,
            messages=_to_jsonable(messages),
        )

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        exit_langchain_llm(state.get("_lc_token"))
        self._emit_kwargs(
            state,
            type="llm",
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
        )

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        exit_langchain_llm(state.get("_lc_token"))
        self._emit_kwargs(
            state,
            type="llm",
            model=state.get("model"),
            messages=_to_jsonable(state.get("messages")),
            input=_to_jsonable(state.get("prompts")),
            output=str(error),
            error_traceback=format_error_traceback(error),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=False,
        )

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       tags=None, metadata=None, **kwargs):
        name = _component_name(serialized)
        # LangGraph's top-level graph run arrives with no serialized name — give
        # the parent-less container the graph's name so it has an agent identity
        # (else it's invisible in the Agents view). Only the outermost chain.
        if not name and parent_run_id is None:
            name = current_graph_name()
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            parent_ids=self._lg_join_parents(metadata, run_id, parent_run_id),
            lg_predecessors=self._lg_predecessors(metadata),
            name=name,
            inputs=inputs,
            metadata=metadata,
        )
        self._emit_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            type="chain",
            function=name,
            input=_to_jsonable(inputs),
        )

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit_kwargs(
            state,
            type="chain",
            function=state.get("name"),
            input=_to_jsonable(state.get("inputs")),
            output=_to_jsonable(outputs),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            parent_ids=state.get("parent_ids"),
            success=True,
        )

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit_kwargs(
            state,
            type="chain",
            function=state.get("name"),
            input=_to_jsonable(state.get("inputs")),
            output=str(error),
            error_traceback=format_error_traceback(error),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            parent_ids=state.get("parent_ids"),
            success=False,
        )

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      tags=None, metadata=None, inputs=None, **kwargs):
        name = _component_name(serialized)
        self._start(
            run_id,
            parent_id=self._parent(parent_run_id),
            name=name,
            input=input_str,
            metadata=metadata,
        )
        self._emit_start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            type="tool",
            function=name,
            input=input_str,
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit_kwargs(
            state,
            type="tool",
            function=state.get("name"),
            input=state.get("input"),
            output=_to_jsonable(output),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=True,
        )

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        state, end = self._end(run_id)
        self._emit_kwargs(
            state,
            type="tool",
            function=state.get("name"),
            input=state.get("input"),
            output=str(error),
            error_traceback=format_error_traceback(error),
            latency=end - state.get("start", end),
            trace_id=str(run_id),
            parent_id=state.get("parent_id") or self._parent(parent_run_id),
            success=False,
        )
