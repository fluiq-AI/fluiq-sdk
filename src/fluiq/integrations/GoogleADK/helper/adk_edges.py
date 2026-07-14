"""Resolve DAG fan-in (join) parents for Google ADK agents.

ADK agents pass data through session ``state``: an agent writes its result to
``state[output_key]``, and a downstream agent's ``instruction`` injects upstream
results via ``{state_key}`` placeholders. So an agent whose instruction reads the
outputs of two or more upstream agents (typically the children of a
``ParallelAgent``) is a **join / fan-in** node. Given a registry of
``output_key -> trace_id`` for agents already completed in the invocation, we turn
the ``{key}`` references back into predecessor run_ids (``parent_ids``).

Pure functions (no google.adk import) so they're cheap to unit-test.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ADK injects state as {key} (required) or {key?} (optional); keys may be dotted
# (e.g. {user.name}) or namespaced. Match the leading identifier path.
_KEY_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.-]*)\??\}")


def instruction_state_keys(instruction: Any) -> List[str]:
    """Ordered, de-duplicated ``{state_key}`` names referenced by an instruction.
    Non-string instructions (InstructionProvider callables) yield no keys."""
    if not isinstance(instruction, str):
        return []
    out: List[str] = []
    seen: set = set()
    for m in _KEY_RE.finditer(instruction):
        key = m.group(1)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def resolve_adk_predecessors(
    registry: Dict[str, str],
    instruction: Any,
) -> Optional[List[str]]:
    """ALL upstream outputs the instruction reads, as run_ids (>= 1), or ``None``.

    Unlike :func:`resolve_adk_join_parents` (fan-in joins only, >= 2), this
    returns every resolved predecessor so the dashboard can draw the full DAG —
    including single-upstream fan-out edges. Stamped as ``predecessors``.
    """
    keys = instruction_state_keys(instruction)
    if not keys or not registry:
        return None
    out: List[str] = []
    seen: set = set()
    for key in keys:
        trace_id = registry.get(key)
        if trace_id and trace_id not in seen:
            out.append(trace_id)
            seen.add(trace_id)
    return out or None


def resolve_adk_join_parents(
    registry: Dict[str, str],
    instruction: Any,
    min_parents: int = 2,
) -> Optional[List[str]]:
    """Return predecessor run_ids for a join agent, or ``None``.

    ``registry`` maps ``output_key -> trace_id`` for agents already completed in
    this invocation. A list is returned only when the instruction references
    ``>= min_parents`` distinct upstream outputs — a genuine fan-in.
    """
    keys = instruction_state_keys(instruction)
    if not keys or not registry:
        return None
    out: List[str] = []
    seen: set = set()
    for key in keys:
        trace_id = registry.get(key)
        if trace_id and trace_id not in seen:
            out.append(trace_id)
            seen.add(trace_id)
    return out if len(out) >= min_parents else None
