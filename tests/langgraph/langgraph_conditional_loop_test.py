from typing import TypedDict
from keys import FLUIQ_API_KEY
from fluiq import instrument
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

instrument(api_key=FLUIQ_API_KEY)


class State(TypedDict):
    question: str
    answer: str
    attempts: int
    accepted: bool


llm = ChatOpenAI(model="gpt-4o-mini")


def answer_node(state: State) -> State:
    style = "very brief (under 15 words)" if state["attempts"] == 0 else "extremely concise (under 10 words)"
    resp = llm.invoke([
        SystemMessage(content=f"Answer the question in a {style} sentence."),
        HumanMessage(content=state["question"]),
    ])
    return {**state, "answer": resp.content, "attempts": state["attempts"] + 1}


def critic_node(state: State) -> State:
    accepted = len(state["answer"].split()) <= 12
    return {**state, "accepted": accepted}


def route(state: State) -> str:
    if state["accepted"] or state["attempts"] >= 3:
        return "end"
    return "retry"


graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_node("critic", critic_node)
graph.add_edge(START, "answer")
graph.add_edge("answer", "critic")
graph.add_conditional_edges("critic", route, {"retry": "answer", "end": END})
app = graph.compile()


def run():
    result = app.invoke({
        "question": "Explain how a transformer model works.",
        "answer": "",
        "attempts": 0,
        "accepted": False,
    })
    return f"attempts={result['attempts']} accepted={result['accepted']} answer={result['answer']}"


print(run())
