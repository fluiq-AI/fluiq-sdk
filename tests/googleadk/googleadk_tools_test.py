import asyncio
from keys import FLUIQ_API_KEY, GEMINI_KEY
from fluiq import instrument
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
import os

instrument(api_key=FLUIQ_API_KEY)

os.environ["GOOGLE_API_KEY"] = GEMINI_KEY


APP_NAME = "fluiq_tools_demo"
USER_ID = "u1"


def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {"city": city, "temp_c": 21, "condition": "sunny"}


def get_population(city: str) -> dict:
    """Get the population of a city."""
    populations = {"paris": "2.1M", "berlin": "3.7M", "tokyo": "13.9M"}
    return {"city": city, "population": populations.get(city.lower(), "unknown")}


agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a city info assistant. Use the available tools to answer "
        "weather and population questions, then summarize the results."
    ),
    tools=[FunctionTool(get_weather), FunctionTool(get_population)],
)


async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What's the weather and population of Paris?")],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    return final_text


print(asyncio.run(run()))
