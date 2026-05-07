"""LangGraph realtime-tracking demo.

Multi-step pipeline with intentional sleeps between nodes so the running-state
spinner stays visible long enough to observe in the dashboard. Open
/dashboard/traces in another tab before running this — a placeholder row
should appear immediately, the architecture view should show a pulsing root
that grows new descendant nodes as each LangGraph node fires, and the row
should swap to the completed cost / latency view at the end.
"""

import time
from typing import TypedDict
from ..keys import FLUIQ_API_KEY
from fluiq import instrument
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

class State(TypedDict):
    topic: str
    research: str
    outline: str
    draft: str
    review: str
    summary: str


llm = ChatOpenAI(model="gpt-4o-mini")


@tool
def slow_research(topic: str) -> str:
    """Simulate a slow knowledge-base lookup so the tool node's spinner
    is observable in the dashboard for ~3 seconds."""
    time.sleep(3)
    return (
        f"Reference notes on '{topic}': distributed tracing reveals "
        "agent decision paths, cost hotspots, and latency tail."
    )


def research_node(state: State) -> State:
    notes = slow_research.invoke({"topic": state["topic"]})
    return {**state, "research": notes}


def outline_node(state: State) -> State:
    time.sleep(1)
    resp = llm.invoke([
        SystemMessage(content="Produce a concise 3-bullet outline."),
        HumanMessage(content=f"Topic: {state['topic']}\nNotes: {state['research']}"),
    ])
    return {**state, "outline": resp.content}


def draft_node(state: State) -> State:
    time.sleep(1)
    resp = llm.invoke([
        SystemMessage(content="Expand this outline into a short paragraph."),
        HumanMessage(content=state["outline"]),
    ])
    return {**state, "draft": resp.content}


def review_node(state: State) -> State:
    time.sleep(1)
    resp = llm.invoke([
        SystemMessage(content="Critique this paragraph in one bullet."),
        HumanMessage(content=state["draft"]),
    ])
    return {**state, "review": resp.content}


def summary_node(state: State) -> State:
    time.sleep(1)
    resp = llm.invoke([
        SystemMessage(content="Summarize the paragraph in one sentence."),
        HumanMessage(content=state["draft"]),
    ])
    return {**state, "summary": resp.content}


graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("outline", outline_node)
graph.add_node("draft", draft_node)
graph.add_node("review", review_node)
graph.add_node("summary", summary_node)
graph.add_edge(START, "research")
graph.add_edge("research", "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "review")
graph.add_edge("review", "summary")
graph.add_edge("summary", END)
app = graph.compile()


def run():
    print("Open /dashboard/traces now — pipeline starts in 3s")
    time.sleep(3)
    result = app.invoke({
        "topic": "Why distributed tracing matters for AI agents",
        "research": "",
        "outline": "",
        "draft": "",
        "review": "",
        "summary": "",
    })
    return result["summary"]


print(run())
