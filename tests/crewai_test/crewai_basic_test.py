from ..keys import FLUIQ_API_KEY, OPENAI_API_KEY
from fluiq import instrument
from crewai import Agent, Task, Crew
import os

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


researcher = Agent(
    role="Research Analyst",
    goal="Answer factual questions accurately and concisely.",
    backstory="You are an expert research analyst with broad general knowledge.",
    llm="gpt-4o-mini",
    verbose=False,
)

task = Task(
    description="What is the capital of France? Answer in one sentence.",
    expected_output="A single sentence stating the capital of France.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task], verbose=False)


def run():
    result = crew.kickoff()
    return result.raw


print(run())
