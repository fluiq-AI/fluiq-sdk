"""Multi-agent tracing (CrewAI): a fan-in join that then fans back out.

Real CrewAI run (like crewai_sequential_test), instrumented against a local
Fluiq. Exercises the DAG shape the SDK's CrewAI join detection cares about:

        ┌─ weather_task ─┐                       ┌─ report_task ─┐
START ──┼─ market_task ──┼──► synthesize_task ──►┤               ├──► END
        └─ culture_task ─┘   (JOIN: context =     └─ critique_task┘
                             [weather, market,       (FAN-OUT: both
                              culture])               context = [synthesize])

  • Three research tasks run first (no context / no upstream dependency).
  • ``synthesize_task`` declares ``context=[weather_task, market_task,
    culture_task]`` — CrewAI's explicit data-dependency list. The SDK's
    resolve_task_parents turns those three executed tasks into
    parent_ids = [weather, market, culture], a genuine fan-in.
  • ``report_task`` and ``critique_task`` each declare ``context=[synthesize_task]``
    — single-dependency, so they're fan-out edges, not joins.

Process is sequential (tasks run in list order); ``context`` defines the DAG the
trace records. So the run's tree carries a multi-parent join on the synthesizer
and a subsequent fan-out — the CrewAI multi-agent case for the
Traces/Architecture views and the agentic evaluator's DAG analysis.

Run:  ../../.venv/Scripts/python.exe -m tests.crewai_test.crewai_multiagent_join_test
"""
import os

from ..keys import FLUIQ_API_KEY, OPENAI_API_KEY
from fluiq import instrument
from crewai import Agent, Task, Crew, Process

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def _analyst(role: str, goal: str) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=f"You are a concise {role.lower()} who answers in a single sentence.",
        llm="gpt-4o-mini",
        verbose=False,
    )


weather_analyst = _analyst("Weather Analyst", "Describe a city's typical weather.")
market_analyst = _analyst("Market Analyst", "Describe a city's local economy.")
culture_analyst = _analyst("Culture Analyst", "Describe a city's culture.")
synthesizer = _analyst("Synthesizer", "Merge analyst notes into one paragraph.")
writer = _analyst("Travel Writer", "Turn a synthesis into an upbeat blurb.")
critic = _analyst("Editor", "Name the weakest point of a synthesis.")


# ── Three researchers (no context = no upstream dependency) ──────────────────
weather_task = Task(
    description="In one sentence, describe the typical weather of Lisbon, Portugal.",
    expected_output="One sentence on Lisbon's weather.",
    agent=weather_analyst,
)
market_task = Task(
    description="In one sentence, describe the local economy of Lisbon, Portugal.",
    expected_output="One sentence on Lisbon's economy.",
    agent=market_analyst,
)
culture_task = Task(
    description="In one sentence, describe the culture of Lisbon, Portugal.",
    expected_output="One sentence on Lisbon's culture.",
    agent=culture_analyst,
)

# ── Join: synthesize consumes all three researchers (fan-in) ─────────────────
synthesize_task = Task(
    description="Merge the weather, market, and culture notes into one tight paragraph.",
    expected_output="A 3-4 sentence paragraph merging the three notes.",
    agent=synthesizer,
    context=[weather_task, market_task, culture_task],
)

# ── Fan-out: two downstream tasks, each depending only on the synthesis ──────
report_task = Task(
    description="Write an upbeat two-sentence travel blurb from the synthesis.",
    expected_output="A two-sentence travel blurb.",
    agent=writer,
    context=[synthesize_task],
)
critique_task = Task(
    description="In one sentence, name the weakest point of the synthesis.",
    expected_output="One sentence naming the weakest point.",
    agent=critic,
    context=[synthesize_task],
)

crew = Crew(
    agents=[weather_analyst, market_analyst, culture_analyst, synthesizer, writer, critic],
    tasks=[weather_task, market_task, culture_task, synthesize_task, report_task, critique_task],
    process=Process.sequential,
    verbose=False,
)


def run():
    result = crew.kickoff()
    return result.raw


print(run())
