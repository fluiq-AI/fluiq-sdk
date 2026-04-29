from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import fluiq

fluiq.instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")

load_dotenv()

class State(TypedDict):
    question: str
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini")


def answer_node(state: State) -> State:
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer in one short sentence."),
        HumanMessage(content=state["question"]),
    ])
    return {"question": state["question"], "answer": response.content}


graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
app = graph.compile()


def run():
    result = app.invoke({"question": "What is the capital of France?", "answer": ""})
    return result["answer"]


print(run())
