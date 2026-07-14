"""SDK-side DAG fan-in emission: a LangGraph join node emits parent_ids.

Two checks, both offline (no real graph / LLM):
  1. The pure resolver (`resolve_join_parents`) maps trigger channel names to
     predecessor run_ids only for genuine joins (>= 2 parents).
  2. The real FluiqCallbackHandler, driven with LangGraph-style metadata,
     emits parent_ids = [A, B] on the join node and None on single-parent nodes.

Run:  ../../.venv/Scripts/python.exe tests/langgraph/test_join_parent_ids.py
"""
import os
import sys

# Make the SDK importable when run as a plain script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from fluiq.integrations.Langchain.helper.langgraph_edges import (
    resolve_join_parents,
    trigger_tokens,
)


# ── 1. pure resolver ─────────────────────────────────────────────────────────

def test_resolver_join_and_non_join():
    registry = {"a": "run-A", "b": "run-B", "c": "run-C"}
    # join: triggered by a and b → both predecessor run_ids
    parents = resolve_join_parents(registry, "join", ["a", "b"])
    assert parents == ["run-A", "run-B"]
    # single parent → None (keep existing parent_id)
    assert resolve_join_parents(registry, "join", ["a"]) is None
    # no triggers → None
    assert resolve_join_parents(registry, "join", None) is None
    # self-reference excluded
    assert resolve_join_parents({"join": "run-J", "a": "run-A"}, "join", ["join", "a"]) is None


def test_trigger_tokens_formats():
    assert trigger_tokens(["branch:to:node_x", "node_a"]) == {"branch", "to", "node_x", "node_a"}
    assert trigger_tokens("node_a") == {"node_a"}
    assert trigger_tokens(None) == set()


# ── 2. handler emits parent_ids on a join node ───────────────────────────────

def _drive_handler():
    import fluiq.integrations.Langchain.handler as handler_mod
    from fluiq.integrations.Langchain.handler import FluiqCallbackHandler

    emitted = []
    handler_mod.log_trace = lambda payload: emitted.append(payload)  # capture

    h = FluiqCallbackHandler()
    meta = lambda node, trig: {
        "thread_id": "thread-1", "langgraph_node": node, "langgraph_triggers": trig,
    }

    # Two parallel branches off the graph container "G", then a join over both.
    h.on_chain_start({"name": "a"}, {}, run_id="A", parent_run_id="G", metadata=meta("a", ["__start__"]))
    h.on_chain_end({}, run_id="A", parent_run_id="G")
    h.on_chain_start({"name": "b"}, {}, run_id="B", parent_run_id="G", metadata=meta("b", ["__start__"]))
    h.on_chain_end({}, run_id="B", parent_run_id="G")
    h.on_chain_start({"name": "join"}, {}, run_id="J", parent_run_id="G", metadata=meta("join", ["a", "b"]))
    h.on_chain_end({}, run_id="J", parent_run_id="G")

    return {p["trace_id"]: p for p in emitted}


def test_handler_emits_join_parent_ids():
    by_id = _drive_handler()
    # single-parent branches: no parent_ids
    assert by_id["A"].get("parent_ids") is None
    assert by_id["B"].get("parent_ids") is None
    # join node: both predecessor run_ids, order-independent
    assert set(by_id["J"].get("parent_ids") or []) == {"A", "B"}
    # single parent_id still the container
    assert by_id["J"]["parent_id"] == "G"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
