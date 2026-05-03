from typing import TypedDict
from keys import FLUIQ_API_KEY
from fluiq import instrument
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

instrument(api_key=FLUIQ_API_KEY)


class State(TypedDict):
    topic: str
    outline: str
    draft: str
    summary: str


llm = ChatOpenAI(model="gpt-4o-mini")


def outline_node(state: State) -> State:
    resp = llm.invoke([
        SystemMessage(content="Produce a 3-bullet outline for the given topic. Be concise."),
        HumanMessage(content=state["topic"]),
    ])
    return {**state, "outline": resp.content}


def draft_node(state: State) -> State:
    resp = llm.invoke([
        SystemMessage(content="Expand this outline into a short paragraph (3-4 sentences)."),
        HumanMessage(content=state["outline"]),
    ])
    return {**state, "draft": resp.content}


def summary_node(state: State) -> State:
    resp = llm.invoke([
        SystemMessage(content="Summarize this paragraph in a single sentence."),
        HumanMessage(content=state["draft"]),
    ])
    return {**state, "summary": resp.content}


graph = StateGraph(State)
graph.add_node("outline", outline_node)
graph.add_node("draft", draft_node)
graph.add_node("summary", summary_node)
graph.add_edge(START, "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "summary")
graph.add_edge("summary", END)
app = graph.compile()


def run():
    result = app.invoke({
        "topic": "Why distributed tracing matters for AI agents",
        "outline": "",
        "draft": "",
        "summary": "",
    })
    return result["summary"]


print(run())
