from fluiq.integrations.Langchain.trace import patch_langchain


def patch_langgraph():
    # LangGraph executes nodes through LangChain Core's callback infrastructure.
    # The same FluiqCallbackHandler captures node/tool/llm events; the handler
    # detects LangGraph metadata (langgraph_node, langgraph_step, ...) and
    # tags those traces with TraceType.LangGraph. Registration is idempotent.
    patch_langchain()
    _patch_state_graph_compile()


def _patch_state_graph_compile():
    """Capture each StateGraph's static edges at compile time.

    LangGraph's per-node ``langgraph_triggers`` name the *destination* channel
    (e.g. ``branch:to:synthesize``), not the source nodes, so trigger parsing
    can't detect a fan-in. Recording the declared edges here lets the callback
    handler resolve a join node to its real predecessor run_ids. Idempotent and
    fail-open: if LangGraph isn't installed or its API differs, tracing simply
    falls back to the trigger-based path.
    """
    try:
        from langgraph.graph import StateGraph
    except Exception:
        return
    if getattr(StateGraph, "_fluiq_compile_patched", False):
        return
    from fluiq.integrations.Langchain.helper.langgraph_edges import register_graph_edges

    _orig_compile = StateGraph.compile

    def _compile(self, *args, **kwargs):
        try:
            register_graph_edges(getattr(self, "edges", None))
        except Exception:
            pass
        compiled = _orig_compile(self, *args, **kwargs)
        try:
            _tag_graph_name(compiled, self)
        except Exception:
            pass
        return compiled

    StateGraph.compile = _compile
    StateGraph._fluiq_compile_patched = True


_EDGE_SENTINELS = ("__start__", "__end__")


def _graph_display_name(graph, compiled):
    """Build ``LangGraph(node_a, node_b, ...)`` from the graph's node names.

    Mirrors CrewAI's ``Crew(agent1, agent2, ...)`` identity so the Agents view
    shows *which* graph ran, not a generic label. Node order follows the user's
    declaration (dict insertion order); START/END sentinels are dropped. Falls
    back to the compiled graph's own ``name`` (``"LangGraph"``) when the node
    list can't be read.
    """
    nodes = None
    for src in (graph, compiled):
        raw = getattr(src, "nodes", None)
        if isinstance(raw, dict) and raw:
            nodes = raw
            break
    fallback = getattr(compiled, "name", None) or "LangGraph"
    if not nodes:
        return fallback
    names = [str(n) for n in nodes.keys() if str(n) not in _EDGE_SENTINELS]
    if not names:
        return fallback
    return "LangGraph(" + ", ".join(names) + ")"


def _tag_graph_name(compiled, graph=None):
    """Wrap a compiled graph's invoke/stream to publish its name during the run.

    The top-level graph container reaches the callback handler with no
    ``serialized`` name, so on its own it has no agent identity and never shows
    in the Agents view. We publish ``LangGraph(node names)`` (like CrewAI's
    ``Crew(...)``) into a ContextVar for the duration of each invocation; the
    handler names the parent-less container chain from it. Nested sub-graph runs
    are unaffected — the handler only reads the name for the outermost chain.
    """
    from fluiq.integrations.Langchain.helper.langgraph_edges import (
        set_current_graph_name,
        reset_current_graph_name,
    )

    if getattr(compiled, "_fluiq_name_tagged", False):
        return
    name = _graph_display_name(graph, compiled)

    def _wrap_sync(method):
        def inner(*args, **kwargs):
            token = set_current_graph_name(name)
            try:
                return method(*args, **kwargs)
            finally:
                reset_current_graph_name(token)
        return inner

    def _wrap_stream(method):
        def inner(*args, **kwargs):
            token = set_current_graph_name(name)
            try:
                yield from method(*args, **kwargs)
            finally:
                reset_current_graph_name(token)
        return inner

    def _wrap_async(method):
        async def inner(*args, **kwargs):
            token = set_current_graph_name(name)
            try:
                return await method(*args, **kwargs)
            finally:
                reset_current_graph_name(token)
        return inner

    def _wrap_astream(method):
        async def inner(*args, **kwargs):
            token = set_current_graph_name(name)
            try:
                async for item in method(*args, **kwargs):
                    yield item
            finally:
                reset_current_graph_name(token)
        return inner

    for attr, wrapper in (
        ("invoke", _wrap_sync),
        ("stream", _wrap_stream),
        ("ainvoke", _wrap_async),
        ("astream", _wrap_astream),
    ):
        orig = getattr(compiled, attr, None)
        if orig is None:
            continue
        try:
            object.__setattr__(compiled, attr, wrapper(orig))
        except Exception:
            pass
    try:
        object.__setattr__(compiled, "_fluiq_name_tagged", True)
    except Exception:
        pass
