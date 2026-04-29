import asyncio
import os
from dotenv import load_dotenv
from fluiq import instrument
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_KEY"]


APP_NAME = "fluiq_basic_demo"
USER_ID = "u1"


agent = LlmAgent(
    name="basic_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Answer in one short sentence.",
)


async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the capital of France?")],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    return final_text


print(asyncio.run(run()))
