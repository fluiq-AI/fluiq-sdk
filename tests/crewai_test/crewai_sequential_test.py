from ..keys import FLUIQ_API_KEY, OPENAI_API_KEY
from fluiq import instrument
from crewai import Agent, Task, Crew, Process
import os

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


researcher = Agent(
    role="Researcher",
    goal="Gather key facts and insights on a given topic.",
    backstory="You are a thorough researcher who finds the most relevant information.",
    llm="gpt-4o-mini",
    verbose=False,
)

writer = Agent(
    role="Technical Writer",
    goal="Turn research notes into a clear, concise paragraph.",
    backstory="You are a technical writer who excels at making complex topics accessible.",
    llm="gpt-4o-mini",
    verbose=False,
)

reviewer = Agent(
    role="Editor",
    goal="Summarize the final paragraph into a single sentence.",
    backstory="You are a sharp editor who distills content to its essential point.",
    llm="gpt-4o-mini",
    verbose=False,
)

research_task = Task(
    description="Research why distributed tracing matters for AI agents. Produce 3 bullet points.",
    expected_output="Three concise bullet points on the importance of distributed tracing for AI agents.",
    agent=researcher,
)

write_task = Task(
    description="Using the research bullet points, write a short paragraph (3-4 sentences).",
    expected_output="A 3-4 sentence paragraph on distributed tracing for AI agents.",
    agent=writer,
    context=[research_task],
)

review_task = Task(
    description="Summarize the paragraph into a single sentence.",
    expected_output="One sentence summarizing why distributed tracing matters for AI agents.",
    agent=reviewer,
    context=[write_task],
)

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, write_task, review_task],
    process=Process.sequential,
    verbose=False,
)


def run():
    result = crew.kickoff()
    return result.raw


print(run())
