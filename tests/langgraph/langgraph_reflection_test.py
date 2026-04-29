from typing import TypedDict
from dotenv import load_dotenv
from fluiq import instrument
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")
load_dotenv()


class State(TypedDict):
    task: str
    research: str
    draft: str
    critique: str
    final: str
    iteration: int


llm = ChatOpenAI(model="gpt-4o-mini")


@tool
def lookup_fact(topic: str) -> str:
    """Look up a quick fact about a topic."""
    facts = {
        "tracing": "OpenTelemetry is the most widely adopted distributed tracing standard.",
        "agents": "Agentic systems often need observability across multiple LLM calls.",
        "default": "No specific fact found; rely on general knowledge.",
    }
    key = next((k for k in facts if k in topic.lower()), "default")
    return facts[key]


researcher = create_react_agent(llm, tools=[lookup_fact])


def research_node(state: State) -> State:
    result = researcher.invoke({
        "messages": [
            {"role": "user", "content": f"Research: {state['task']}. Use lookup_fact then summarize in 2 sentences."}
        ]
    })
    return {**state, "research": result["messages"][-1].content}


def generator_node(state: State) -> State:
    resp = llm.invoke([
        SystemMessage(content="You write short technical paragraphs."),
        HumanMessage(content=f"Task: {state['task']}\nResearch: {state['research']}\nCritique: {state['critique']}\nWrite a 3-sentence answer."),
    ])
    return {**state, "draft": resp.content, "iteration": state["iteration"] + 1}


def critic_node(state: State) -> State:
    resp = llm.invoke([
        SystemMessage(content="You are a strict critic. Reply with 'APPROVED' if the draft is clear and factual, otherwise give one short improvement."),
        HumanMessage(content=state["draft"]),
    ])
    return {**state, "critique": resp.content}


def finalize_node(state: State) -> State:
    return {**state, "final": state["draft"]}


def route(state: State) -> str:
    if "APPROVED" in state["critique"].upper() or state["iteration"] >= 2:
        return "finalize"
    return "revise"


graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("generator", generator_node)
graph.add_node("critic", critic_node)
graph.add_node("finalize", finalize_node)
graph.add_edge(START, "research")
graph.add_edge("research", "generator")
graph.add_edge("generator", "critic")
graph.add_conditional_edges("critic", route, {"revise": "generator", "finalize": "finalize"})
graph.add_edge("finalize", END)
app = graph.compile()


def run():
    result = app.invoke({
        "task": "Explain why distributed tracing matters for agentic AI systems.",
        "research": "",
        "draft": "",
        "critique": "",
        "final": "",
        "iteration": 0,
    })
    return f"iterations={result['iteration']}\n{result['final']}"


print(run())
