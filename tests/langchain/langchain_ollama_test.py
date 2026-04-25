from fluiq import instrument, trace
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

instrument(api_key="YOUR_KEY")

llm = ChatOllama(model="llama3.1:8b")


@trace
def run():
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that answers in one short sentence."),
        HumanMessage(content="What is the capital of France?"),
    ])
    return response.content


print(run())
