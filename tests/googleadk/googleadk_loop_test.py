import asyncio
from fluiq import instrument
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types
import os
from keys import FLUIQ_API_KEY, GEMINI_KEY

instrument(api_key=FLUIQ_API_KEY)
os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

APP_NAME = "fluiq_loop_demo"
USER_ID = "u1"


def stop_loop(reason: str, tool_context: ToolContext) -> dict:
    """Stop the surrounding LoopAgent when the answer is acceptable."""
    tool_context.actions.escalate = True
    return {"stopped": True, "reason": reason}


writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction=(
        "You are a concise writer. If session state has a 'critique' key, "
        "rewrite the previous 'draft' incorporating the critique. Otherwise, "
        "write a first short answer (under 25 words) to the user's question."
    ),
    output_key="draft",
)

critic = LlmAgent(
    name="critic",
    model="gemini-2.5-flash",
    instruction=(
        "Read the draft in session state under key 'draft'. If it is under "
        "20 words and clear, call the stop_loop tool with reason='ok'. "
        "Otherwise, output a one-line critique describing what to shorten."
    ),
    tools=[FunctionTool(stop_loop)],
    output_key="critique",
)

revision_loop = LoopAgent(
    name="revision_loop",
    sub_agents=[writer, critic],
    max_iterations=3,
)

pipeline = SequentialAgent(
    name="loop_pipeline",
    sub_agents=[revision_loop],
)


async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(app_name=APP_NAME, agent=pipeline, session_service=session_service)

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Explain how a transformer model works.")],
    )

    last_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            last_text = "".join(p.text or "" for p in event.content.parts)

    final_state = (await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session.id
    )).state
    return f"draft={final_state.get('draft', '')}\nlast_event={last_text}"


print(asyncio.run(run()))
