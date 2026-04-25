from dotenv import load_dotenv
from fluiq import instrument, trace
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

instrument(api_key="YOUR_KEY")
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")


@trace
def run():
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ])
    return response.content


print(run())
