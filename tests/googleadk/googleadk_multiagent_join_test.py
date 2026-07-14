"""Multi-agent tracing (Google ADK): a fan-in join that then fans back out.

Real ADK run (like googleadk_parallel_test), instrumented against a local Fluiq.
Exercises the DAG shape the SDK's ADK join detection cares about:

        ┌─ research_weather ─┐                    ┌─ write_report ─┐
START ──┼─ research_market ──┼──► synthesize ──►──┤                ├──► END
        └─ research_culture ─┘   (JOIN: reads     └─ critique ─────┘
                                  {weather}{market}   (FAN-OUT: two
                                  {culture})           agents read {verdict})

  • A ParallelAgent fans out to three researchers, each writing its own
    session-state output_key (weather / market / culture).
  • ``synthesize``'s instruction injects all three via {placeholders}; the SDK's
    resolve_adk_join_parents turns those into parent_ids = [weather, market,
    culture] — a genuine fan-in. (Prose like "the pros key" would NOT trigger
    it; the join is detected from the {key} placeholders ADK actually injects.)
  • A second ParallelAgent fans back out to two agents that each read only
    {verdict} — single-parent nodes, so the fan-out edges, not joins.

So the trace shows one run whose tree carries a multi-parent join on the
synthesizer and a subsequent fan-out — the ADK multi-agent case to eyeball in
the Traces/Architecture views (and for the agentic evaluator's DAG analysis).

Run:  ../../.venv/Scripts/python.exe -m tests.googleadk.googleadk_multiagent_join_test
"""
import asyncio
import os

from ..keys import FLUIQ_API_KEY, GEMINI_KEY
from fluiq import instrument
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
os.environ["GOOGLE_API_KEY"] = GEMINI_KEY


APP_NAME = "fluiq_multiagent_join_demo"
USER_ID = "u1"

# ── Fan-out: three researchers in parallel, each writing its own state key ────
research_weather = LlmAgent(
    name="research_weather",
    model="gemini-2.5-flash",
    instruction="In one sentence, describe the typical weather of the user's city.",
    output_key="weather",
)
research_market = LlmAgent(
    name="research_market",
    model="gemini-2.5-flash",
    instruction="In one sentence, describe the local economy of the user's city.",
    output_key="market",
)
research_culture = LlmAgent(
    name="research_culture",
    model="gemini-2.5-flash",
    instruction="In one sentence, describe the culture of the user's city.",
    output_key="culture",
)
fanout = ParallelAgent(
    name="research_fanout",
    sub_agents=[research_weather, research_market, research_culture],
)

# ── Join: synthesizer reads all three upstream outputs via {placeholders} ─────
synthesize = LlmAgent(
    name="synthesize",
    model="gemini-2.5-flash",
    instruction=(
        "Merge these notes into one tight paragraph.\n"
        "Weather: {weather}\nMarket: {market}\nCulture: {culture}"
    ),
    output_key="verdict",
)

# ── Fan-out again: two downstream agents, each reading only {verdict} ─────────
write_report = LlmAgent(
    name="write_report",
    model="gemini-2.5-flash",
    instruction="Write an upbeat two-sentence travel blurb from: {verdict}",
    output_key="report",
)
critique = LlmAgent(
    name="critique",
    model="gemini-2.5-flash",
    instruction="In one sentence, name the weakest point of this synthesis: {verdict}",
    output_key="review",
)
downstream = ParallelAgent(
    name="downstream_fanout",
    sub_agents=[write_report, critique],
)

pipeline = SequentialAgent(
    name="multiagent_join_pipeline",
    sub_agents=[fanout, synthesize, downstream],
)


async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(app_name=APP_NAME, agent=pipeline, session_service=session_service)
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Lisbon, Portugal")],
    )
    last_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            last_text = "".join(p.text or "" for p in event.content.parts)
    return last_text


print(asyncio.run(run()))
