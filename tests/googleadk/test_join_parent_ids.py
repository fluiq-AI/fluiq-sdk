"""Google ADK DAG fan-in emission: a join agent emits parent_ids.

  1. Pure resolver maps instruction {state_key} refs to upstream output_key
     writers, only for genuine joins (>= 2 upstream outputs).
  2. The real FluiqADKPlugin, driven with two writer agents then a synthesizer
     whose instruction reads both outputs, emits parent_ids = [rx, ry].

Run:  ../../.venv/Scripts/python.exe tests/googleadk/test_join_parent_ids.py
"""
import asyncio
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from fluiq.integrations.GoogleADK.helper.adk_edges import (
    instruction_state_keys,
    resolve_adk_join_parents,
)


# ── 1. pure resolver ─────────────────────────────────────────────────────────

def test_adk_edges():
    assert instruction_state_keys("Combine {x_out} and {y_out?} now.") == ["x_out", "y_out"]
    assert instruction_state_keys(None) == []          # callable instruction → no keys
    reg = {"x_out": "run-X", "y_out": "run-Y", "z": "run-Z"}
    assert resolve_adk_join_parents(reg, "use {x_out} and {y_out}") == ["run-X", "run-Y"]
    assert resolve_adk_join_parents(reg, "just {x_out}") is None       # single → not a join
    assert resolve_adk_join_parents(reg, "no refs here") is None
    assert resolve_adk_join_parents({}, "use {x_out} and {y_out}") is None


# ── 2. plugin emits parent_ids on the join agent ─────────────────────────────

def _stub_adk_baseplugin():
    """Inject a minimal google.adk.plugins.base_plugin so the plugin module
    imports without the real google.adk (which has a broken opentelemetry-semconv
    dep in this venv). The plugin's ONLY google import is BasePlugin."""
    import types
    for name in ("google.adk", "google.adk.plugins", "google.adk.plugins.base_plugin"):
        sys.modules.setdefault(name, types.ModuleType(name))

    class BasePlugin:
        def __init__(self, name=None):
            self.name = name

    sys.modules["google.adk.plugins.base_plugin"].BasePlugin = BasePlugin


def test_adk_plugin_emits_join_parent_ids():
    _stub_adk_baseplugin()
    import fluiq.integrations.GoogleADK.plugin as pm

    emitted = []
    pm.log_trace = lambda payload: emitted.append(payload)
    # Neutralize the ADK-object helpers so the callbacks run on plain mocks.
    pm._collect_output_keys = lambda a: []
    pm._agent_model = lambda a: None
    pm._user_message = lambda cc: ""
    pm._state_snapshot = lambda cc, keys: {}
    pm._agent_output = lambda cc, keys, before: None
    pm._invocation_id = lambda cc: getattr(cc, "invocation_id", None)

    plugin = pm.FluiqADKPlugin()
    cc = SimpleNamespace(invocation_id="inv1")

    def agent(name, instruction=None, output_key=None):
        return SimpleNamespace(name=name, instruction=instruction, output_key=output_key)

    async def run():
        for a in (agent("rx", output_key="x_out"), agent("ry", output_key="y_out")):
            await plugin.before_agent_callback(agent=a, callback_context=cc)
            await plugin.after_agent_callback(agent=a, callback_context=cc)
        synth = agent("synth", instruction="Combine {x_out} and {y_out} into a summary.")
        await plugin.before_agent_callback(agent=synth, callback_context=cc)
        await plugin.after_agent_callback(agent=synth, callback_context=cc)

    asyncio.run(run())

    done = {}
    for p in emitted:
        if p.get("status") != "running":
            done[p.get("function")] = p

    assert done["rx"].get("parent_ids") is None
    assert done["ry"].get("parent_ids") is None
    join_parents = done["synth"].get("parent_ids")
    assert join_parents is not None and len(join_parents) == 2
    # they are the writer agents' trace_ids
    assert set(join_parents) == {done["rx"]["trace_id"], done["ry"]["trace_id"]}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
