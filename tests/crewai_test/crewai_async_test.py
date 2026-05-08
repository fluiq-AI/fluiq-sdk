import asyncio
from ..keys import FLUIQ_API_KEY, OPENAI_API_KEY
from fluiq import instrument
from crewai import Agent, Task, Crew
import os

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


analyst = Agent(
    role="Data Analyst",
    goal="Answer analytical questions clearly and concisely.",
    backstory="You are a data analyst with expertise in interpreting information.",
    llm="gpt-4o-mini",
    verbose=False,
)

task = Task(
    description="Explain what a large language model is in two sentences.",
    expected_output="A two-sentence explanation of what a large language model is.",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task], verbose=False)


async def run():
    result = await crew.kickoff_async()
    return result


print(asyncio.run(run()))
