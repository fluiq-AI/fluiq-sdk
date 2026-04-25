from dotenv import load_dotenv
from fluiq import instrument, trace
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

instrument(api_key="YOUR_KEY")
load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 21C."


llm = ChatOpenAI(model="gpt-4o").bind_tools([get_weather])


@trace
def run():
    messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed."),
        HumanMessage(content="What's the weather in Paris?"),
    ]
    first = llm.invoke(messages)
    messages.append(first)

    for tc in first.tool_calls or []:
        result = get_weather.invoke(tc["args"])
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })

    second = llm.invoke(messages)
    return second.content


print(run())
