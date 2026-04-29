from dotenv import load_dotenv
from fluiq import instrument
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")
load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 21C."


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {"paris": "2.1M", "berlin": "3.7M", "tokyo": "13.9M"}
    return populations.get(city.lower(), "unknown")


llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[get_weather, get_population])


def run():
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "What's the weather and population of Paris?"}
        ]
    })
    return result["messages"][-1].content


print(run())
