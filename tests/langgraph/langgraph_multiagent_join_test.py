"""Multi-agent tracing: a fan-in join that then fans back out.

Real LangGraph run (like langgraph_react_agent_test), instrumented against a
local Fluiq. Exercises the DAG shape the SDK's join detection cares about:

        ┌─ research_weather ─┐
START ──┼─ research_market ──┼──► synthesize ──┬─► write_report ─┐
        └─ research_culture ─┘   (JOIN: 3       └─► critique ─────┴─► END
                                  parents)         (FAN-OUT: 2
                                                    branches)

  • START fans out to three researcher agents that run in parallel.
  • ``synthesize`` is triggered by all three → the FluiqCallbackHandler emits
    parent_ids = [weather, market, culture] on that node (a genuine join).
  • ``synthesize`` then fans out to two different downstream agents, each its
    own branch to END.

So the trace shows a single run whose tree carries multi-parent join edges on
the synthesizer and a subsequent fan-out — the multiagent case to eyeball in
the Traces/Architecture views (and for the agentic evaluator's DAG analysis).

Concurrent writes to shared state need reducers, hence the Annotated list
fields (parallel branches append instead of clobbering).

Run:  ../../.venv/Scripts/python.exe tests/langgraph/langgraph_multiagent_join_test.py
"""
import operator
from typing import Annotated, TypedDict

from ..keys import FLUIQ_API_KEY
from fluiq import instrument
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")


class State(TypedDict):
    topic: str
    # Written concurrently by the three researchers → reducer appends them.
    findings: Annotated[list[str], operator.add]
    synthesis: str
    # Written concurrently by write_report + critique → reducer appends them.
    outputs: Annotated[list[str], operator.add]


llm = ChatOpenAI(model="gpt-4o-mini")


def _agent(system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content


# ── Parallel researchers (fan-out from START) ────────────────────────────────

def research_weather(state: State) -> dict:
    out = _agent(
        "You are a weather analyst. One short sentence.",
        f"What's the typical weather in {state['topic']}?",
    )
    return {"findings": [f"[weather] {out}"]}


def research_market(state: State) -> dict:
    out = _agent(
        "You are a market analyst. One short sentence.",
        f"Describe the local economy of {state['topic']}.",
    )
    return {"findings": [f"[market] {out}"]}


def research_culture(state: State) -> dict:
    out = _agent(
        "You are a culture analyst. One short sentence.",
        f"Describe the culture of {state['topic']}.",
    )
    return {"findings": [f"[culture] {out}"]}


# ── Join: synthesize consumes all three researchers ──────────────────────────

def synthesize(state: State) -> dict:
    out = _agent(
        "You merge analyst notes into a single tight paragraph.",
        "Merge these notes:\n" + "\n".join(state["findings"]),
    )
    return {"synthesis": out}


# ── Fan-out again: two different downstream agents ───────────────────────────

def write_report(state: State) -> dict:
    out = _agent(
        "You are a report writer. Two sentences, upbeat.",
        f"Write a mini travel blurb from: {state['synthesis']}",
    )
    return {"outputs": [f"[report] {out}"]}


def critique(state: State) -> dict:
    out = _agent(
        "You are a skeptical editor. One sentence naming the weakest point.",
        f"Critique this synthesis: {state['synthesis']}",
    )
    return {"outputs": [f"[critique] {out}"]}


graph = StateGraph(State)
graph.add_node("research_weather", research_weather)
graph.add_node("research_market", research_market)
graph.add_node("research_culture", research_culture)
graph.add_node("synthesize", synthesize)
graph.add_node("write_report", write_report)
graph.add_node("critique", critique)

# Fan-out from START to the three researchers (parallel).
graph.add_edge(START, "research_weather")
graph.add_edge(START, "research_market")
graph.add_edge(START, "research_culture")

# Fan-in JOIN: synthesize runs once, triggered by all three researchers.
graph.add_edge("research_weather", "synthesize")
graph.add_edge("research_market", "synthesize")
graph.add_edge("research_culture", "synthesize")

# Fan-out again to two different downstream agents, each to END.
graph.add_edge("synthesize", "write_report")
graph.add_edge("synthesize", "critique")
graph.add_edge("write_report", END)
graph.add_edge("critique", END)

app = graph.compile()


def run():
    result = app.invoke({
        "topic": "Lisbon, Portugal",
        "findings": [],
        "synthesis": "",
        "outputs": [],
    })
    lines = [
        f"researchers: {len(result['findings'])} findings joined",
        f"synthesis: {result['synthesis']}",
        *result["outputs"],
    ]
    return "\n".join(lines)


print(run())
