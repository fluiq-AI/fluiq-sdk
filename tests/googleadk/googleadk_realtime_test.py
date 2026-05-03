"""Google ADK realtime-tracking demo.

Sequential agent with sub-agents + a slow function tool, so the running-state
spinner stays visible long enough to observe in the dashboard. Open
/dashboard/traces in another tab before running this — placeholder rows
should appear immediately as each agent / tool starts, and the architecture
view should show pulsing nodes that swap to completed state once each step
finishes.
"""

import asyncio
import os
import time

from keys import FLUIQ_API_KEY, GEMINI_KEY
from fluiq import instrument
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

instrument(api_key=FLUIQ_API_KEY)

os.environ["GOOGLE_API_KEY"] = GEMINI_KEY


APP_NAME = "fluiq_realtime_demo"
USER_ID = "u1"


def slow_research(topic: str) -> str:
    """Simulate a slow knowledge-base lookup so the tool node's spinner is
    observable in the dashboard for ~3 seconds.

    Args:
      topic: subject to research.

    Returns:
      A short reference note.
    """
    time.sleep(3)
    return (
        f"Reference notes on '{topic}': distributed tracing reveals "
        "agent decision paths, cost hotspots, and latency tail."
    )


researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction=(
        "Call the slow_research tool with the user's topic, then return the "
        "tool's notes verbatim."
    ),
    tools=[FunctionTool(func=slow_research)],
    output_key="research",
)

outliner = LlmAgent(
    name="outliner",
    model="gemini-2.5-flash",
    instruction=(
        "Read 'research' from session state and produce a concise 3-bullet "
        "outline. Be brief."
    ),
    output_key="outline",
)

drafter = LlmAgent(
    name="drafter",
    model="gemini-2.5-flash",
    instruction=(
        "Read 'outline' from session state and expand it into a short "
        "paragraph (3-4 sentences)."
    ),
    output_key="draft",
)

reviewer = LlmAgent(
    name="reviewer",
    model="gemini-2.5-flash",
    instruction=(
        "Read 'draft' from session state and provide a single-bullet "
        "critique."
    ),
    output_key="review",
)

summarizer = LlmAgent(
    name="summarizer",
    model="gemini-2.5-flash",
    instruction=(
        "Read 'draft' from session state and summarize it in a single "
        "sentence."
    ),
    output_key="summary",
)

pipeline = SequentialAgent(
    name="writing_pipeline",
    sub_agents=[researcher, outliner, drafter, reviewer, summarizer],
)


async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(app_name=APP_NAME, agent=pipeline, session_service=session_service)

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Why distributed tracing matters for AI agents")],
    )

    print("Open /dashboard/traces now — pipeline starts in 3s")
    await asyncio.sleep(3)

    last_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            last_text = "".join(p.text or "" for p in event.content.parts)

    return last_text


print(asyncio.run(run()))
