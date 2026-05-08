from ..keys import FLUIQ_API_KEY, OPENAI_API_KEY
from fluiq import instrument
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class WeatherInput(BaseModel):
    city: str = Field(description="The city to get weather for.")


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get the current weather for a city."
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, city: str) -> str:
        return f"The weather in {city} is sunny, 21C."


class PopulationInput(BaseModel):
    city: str = Field(description="The city to get population for.")


class PopulationTool(BaseTool):
    name: str = "get_population"
    description: str = "Get the population of a city."
    args_schema: type[BaseModel] = PopulationInput

    def _run(self, city: str) -> str:
        populations = {"paris": "2.1M", "berlin": "3.7M", "tokyo": "13.9M"}
        return populations.get(city.lower(), "unknown")


weather_tool = WeatherTool()
population_tool = PopulationTool()

city_agent = Agent(
    role="City Info Specialist",
    goal="Answer questions about cities using available tools.",
    backstory="You are a city information expert. Always use tools to look up data.",
    tools=[weather_tool, population_tool],
    llm="gpt-4o-mini",
    verbose=False,
)

task = Task(
    description="What is the weather and population of Paris? Use the tools provided.",
    expected_output="A short summary of Paris's current weather and population.",
    agent=city_agent,
)

crew = Crew(agents=[city_agent], tasks=[task], verbose=False)


def run():
    result = crew.kickoff()
    return result.raw


print(run())
