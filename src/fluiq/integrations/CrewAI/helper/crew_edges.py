"""Resolve DAG fan-in (join) parents for CrewAI tasks.

In CrewAI a Task declares the upstream tasks it depends on via ``task.context``
(a list of other Task objects whose outputs feed it). A task with two or more
context tasks is a **join / fan-in** node — the single ``parent_id`` (the crew /
enclosing scope) can't express that. Given a registry of ``id(task) -> trace_id``
for tasks already executed in this run, we turn ``task.context`` back into the
predecessor run_ids that become ``parent_ids`` on the emitted trace.

Pure functions (no crewai import) so they're cheap to unit-test.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def resolve_task_predecessors(
    registry: Dict[int, str],
    context: Any,
    self_trace_id: Optional[str] = None,
) -> Optional[List[str]]:
    """ALL executed ``context`` dependencies as run_ids (>= 1), or ``None``.

    Unlike :func:`resolve_task_parents` (fan-in joins only, >= 2), this returns
    every resolved predecessor so the dashboard can draw the full DAG — the
    single-dependency fan-out edges too. Stamped on the task as ``predecessors``.
    """
    if not isinstance(context, (list, tuple)) or not context:
        return None
    out: List[str] = []
    seen: set = set()
    for dep in context:
        trace_id = registry.get(id(dep))
        if trace_id and trace_id != self_trace_id and trace_id not in seen:
            out.append(trace_id)
            seen.add(trace_id)
    return out or None


def resolve_task_parents(
    registry: Dict[int, str],
    context: Any,
    self_trace_id: Optional[str],
    min_parents: int = 2,
) -> Optional[List[str]]:
    """Return predecessor run_ids for a join task, or ``None``.

    ``registry`` maps ``id(task) -> trace_id`` for already-executed tasks;
    ``context`` is ``task.context`` (a list of dependency Task objects). A list
    is returned only when ``>= min_parents`` distinct dependencies resolve — a
    genuine fan-in — so single-dependency tasks keep their existing ``parent_id``.
    """
    if not isinstance(context, (list, tuple)) or not context:
        return None
    out: List[str] = []
    seen: set = set()
    for dep in context:
        trace_id = registry.get(id(dep))
        if trace_id and trace_id != self_trace_id and trace_id not in seen:
            out.append(trace_id)
            seen.add(trace_id)
    return out if len(out) >= min_parents else None
