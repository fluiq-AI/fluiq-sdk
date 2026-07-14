"""Resolve DAG fan-in (join) parents for LangGraph nodes.

A LangGraph node exposes ``langgraph_triggers`` in its metadata — the channels
that fired it, which are named after the upstream nodes that wrote them. When a
node is triggered by *two or more* upstream nodes it is a **join / fan-in** node
with multiple logical parents, which the single ``parent_id`` (the graph
container) can't express.

These are pure functions (no LangChain import) so they're cheap to unit-test.
The handler keeps a per-graph-run registry of ``node_name -> run_id`` and calls
:func:`resolve_join_parents` to turn the trigger channel names back into the
predecessor run_ids that become ``parent_ids`` on the emitted trace.
"""
from __future__ import annotations

import contextvars
import re
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[^A-Za-z0-9_]+")

# ── Current graph name (set around a compiled-graph invocation) ───────────────
# LangGraph's top-level Pregel run reaches the callback handler with an empty
# ``serialized`` — so the graph *container* span has no name of its own and,
# without help, ends up with an empty agent identity (invisible in the Agents
# view). The compile patch wraps each compiled graph's invoke/stream to publish
# its name here for the duration of the run; the handler reads it to name the
# parent-less container chain. A ContextVar keeps it correct under async and
# concurrent graph runs.
_current_graph_name: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "fluiq_lg_graph_name", default=None
)


def set_current_graph_name(name: Optional[str]):
    """Publish the running graph's name; returns a token to reset() with."""
    return _current_graph_name.set(name or None)


def reset_current_graph_name(token) -> None:
    try:
        _current_graph_name.reset(token)
    except Exception:
        pass


def current_graph_name() -> Optional[str]:
    """The name of the graph currently being invoked, if any."""
    return _current_graph_name.get()

# ── Static edge graph (captured at StateGraph.compile) ───────────────────────
# Modern LangGraph names a node's ``langgraph_triggers`` after the destination
# channel (e.g. ``branch:to:synthesize``), NOT the source nodes — so trigger
# parsing alone can't see a fan-in. We instead capture the graph's declared
# edges at compile time and resolve a join node's parents from its static
# predecessors. ``target_node -> {source_nodes}``, merged across every compiled
# graph in the process; join resolution always intersects this with the
# per-run registry of nodes that actually ran, so cross-graph node-name
# collisions can never invent a false parent.
_STATIC_PREDS: Dict[str, set] = {}

# LangGraph's START / END sentinels — never real join parents.
_EDGE_SENTINELS = {"__start__", "__end__"}


def register_graph_edges(edges: Any) -> None:
    """Record static ``(source, target)`` edges as ``target -> {sources}``.

    Called from the LangGraph compile patch with ``StateGraph.edges`` (a set of
    2-tuples). Ignores START/END sentinels and malformed entries. Fail-open.
    """
    try:
        for edge in edges or ():
            try:
                src, dst = edge
            except (TypeError, ValueError):
                continue
            src, dst = str(src), str(dst)
            if not src or not dst or src in _EDGE_SENTINELS or dst in _EDGE_SENTINELS:
                continue
            _STATIC_PREDS.setdefault(dst, set()).add(src)
    except Exception:
        pass


def predecessor_names(node_name: Optional[str]) -> List[str]:
    """Static predecessor node names for ``node_name`` (sorted), or ``[]``.

    Stamped onto each LangGraph node's trace so the dashboard can draw the real
    DAG — both fan-in (join) and fan-out edges. ``parent_ids`` only carries
    fan-in joins, so the single-predecessor edges (e.g. synthesize → writer)
    would otherwise be invisible.
    """
    if not node_name:
        return []
    return sorted(_STATIC_PREDS.get(node_name, ()))


def resolve_join_parents_by_edges(
    registry: Dict[str, str],
    node_name: Optional[str],
    min_parents: int = 2,
) -> Optional[List[str]]:
    """Join parents from the static edge graph, intersected with nodes that ran.

    A node's parents are its declared predecessors that have already registered
    a run_id in the current graph run. Reliable regardless of the LangGraph
    trigger encoding. Returns the predecessor run_ids only for a genuine fan-in
    (``>= min_parents``); ``None`` otherwise so the single ``parent_id`` stands.
    """
    if not node_name or not registry:
        return None
    preds = _STATIC_PREDS.get(node_name)
    if not preds:
        return None
    out: List[str] = []
    seen: set = set()
    for name in preds:
        run_id = registry.get(name)
        if run_id and run_id not in seen:
            out.append(run_id)
            seen.add(run_id)
    return out if len(out) >= min_parents else None


def trigger_tokens(triggers: Any) -> set:
    """Split trigger strings into node-name tokens, format-agnostically.

    LangGraph trigger encodings vary by version (``"node_a"``,
    ``"branch:to:node_x"``, channel names, ...). Rather than parse a specific
    grammar we tokenize on non-identifier chars and match tokens against known
    node names — robust across versions.
    """
    if triggers is None:
        return set()
    if isinstance(triggers, (list, tuple, set)):
        text = " ".join(str(t) for t in triggers)
    else:
        text = str(triggers)
    return {t for t in _TOKEN_RE.split(text) if t}


def resolve_join_parents(
    registry: Dict[str, str],
    node_name: Optional[str],
    triggers: Any,
    min_parents: int = 2,
) -> Optional[List[str]]:
    """Return the predecessor run_ids for a join node, or ``None``.

    ``registry`` maps predecessor ``node_name -> run_id`` for the current graph
    run. A list is returned only when ``>= min_parents`` distinct predecessors
    are named in ``triggers`` (a genuine fan-in) — single-parent nodes return
    ``None`` so their existing ``parent_id`` is used unchanged.
    """
    tokens = trigger_tokens(triggers)
    if not tokens or not registry:
        return None
    out: List[str] = []
    seen: set = set()
    for name, run_id in registry.items():
        if not name or name == node_name:
            continue
        if name in tokens and run_id not in seen:
            out.append(run_id)
            seen.add(run_id)
    return out if len(out) >= min_parents else None
