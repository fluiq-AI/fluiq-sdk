"""parent_ids fan-in emission for CrewAI + the generic A2A/custom seam.

Offline (no crewai run, no LLM):
  1. CrewAI: the pure resolver maps a task's context dependencies to predecessor
     run_ids only for genuine joins (>= 2 deps).
  2. Generic: `fluiq.join_parents(...)` makes the next @fluiq.trace call emit
     parent_ids — the seam A2A / custom multi-agent code uses to record joins.

Run:  ../.venv/Scripts/python.exe tests/multiagent_joins_test.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))


# ── 1. CrewAI task-context resolver (pure) ───────────────────────────────────

def test_crewai_resolver():
    from fluiq.integrations.CrewAI.helper.crew_edges import resolve_task_parents

    class Task:  # stand-in for a crewai Task object
        pass

    a, b, c = Task(), Task(), Task()
    registry = {id(a): "run-A", id(b): "run-B", id(c): "run-C"}

    # join: task depends on a and b → both predecessor run_ids
    assert resolve_task_parents(registry, [a, b], "run-J") == ["run-A", "run-B"]
    # single dependency → None (keep existing parent_id)
    assert resolve_task_parents(registry, [a], "run-J") is None
    # no context → None
    assert resolve_task_parents(registry, None, "run-J") is None
    # duplicate dep collapses to one → not a join
    assert resolve_task_parents(registry, [a, a], "run-J") is None
    # unknown deps (not yet run) are ignored
    assert resolve_task_parents(registry, [a, Task()], "run-J") is None


# ── 2. generic join_parents seam via @fluiq.trace ────────────────────────────

def test_join_parents_decorator():
    import fluiq
    import fluiq.decorator as dec

    emitted = []
    dec.log_trace = lambda payload: emitted.append(payload)  # capture

    @fluiq.trace
    def branch(x):
        return x

    @fluiq.trace
    def synthesize():
        return "merged"

    def run_and_trace_id(fn, *a):
        emitted.clear()
        fn(*a)
        ends = [e for e in emitted if e.get("status") != "running"]
        return ends[-1]["trace_id"]

    tid_a = run_and_trace_id(branch, "A")
    tid_b = run_and_trace_id(branch, "B")

    # branches themselves declare no parents
    emitted.clear()
    branch("C")
    assert [e for e in emitted if e.get("status") != "running"][-1]["parent_ids"] is None

    # the join node declares both branch run_ids as parents
    emitted.clear()
    with fluiq.join_parents(tid_a, tid_b):
        synthesize()
    join_payload = [e for e in emitted if e.get("status") != "running"][-1]
    assert join_payload["parent_ids"] == [tid_a, tid_b]

    # declaration is consumed once — a later traced call is unaffected
    emitted.clear()
    synthesize()
    assert [e for e in emitted if e.get("status") != "running"][-1]["parent_ids"] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
