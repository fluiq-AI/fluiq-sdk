import asyncio
from dotenv import load_dotenv
from fluiq import instrument
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import os

instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_KEY"]


APP_NAME = "fluiq_sequential_demo"
USER_ID = "u1"


outliner = LlmAgent(
    name="outliner",
    model="gemini-2.5-flash",
    instruction="Produce a 3-bullet outline for the user's topic. Be concise.",
    output_key="outline",
)

drafter = LlmAgent(
    name="drafter",
    model="gemini-2.5-flash",
    instruction=(
        "Expand the outline in session state under key 'outline' into a "
        "short paragraph (3-4 sentences)."
    ),
    output_key="draft",
)

summarizer = LlmAgent(
    name="summarizer",
    model="gemini-2.5-flash",
    instruction=(
        "Summarize the paragraph in session state under key 'draft' "
        "into a single sentence."
    ),
    output_key="summary",
)

pipeline = SequentialAgent(
    name="writing_pipeline",
    sub_agents=[outliner, drafter, summarizer],
)


async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(app_name=APP_NAME, agent=pipeline, session_service=session_service)

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Why distributed tracing matters for AI agents")],
    )

    last_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            last_text = "".join(p.text or "" for p in event.content.parts)

    return last_text


print(asyncio.run(run()))
