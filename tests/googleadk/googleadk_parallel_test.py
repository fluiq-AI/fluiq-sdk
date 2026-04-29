import asyncio
from dotenv import load_dotenv
from fluiq import instrument
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import os

instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_KEY"]


APP_NAME = "fluiq_parallel_demo"
USER_ID = "u1"


pros_agent = LlmAgent(
    name="pros_agent",
    model="gemini-2.5-flash",
    instruction="List 3 short bullet points of PROS for the user's topic. Be concise.",
    output_key="pros",
)

cons_agent = LlmAgent(
    name="cons_agent",
    model="gemini-2.5-flash",
    instruction="List 3 short bullet points of CONS for the user's topic. Be concise.",
    output_key="cons",
)

risks_agent = LlmAgent(
    name="risks_agent",
    model="gemini-2.5-flash",
    instruction="List 3 short bullet points of RISKS for the user's topic. Be concise.",
    output_key="risks",
)

fanout = ParallelAgent(
    name="analysis_fanout",
    sub_agents=[pros_agent, cons_agent, risks_agent],
)

synthesizer = LlmAgent(
    name="synthesizer",
    model="gemini-2.5-flash",
    instruction=(
        "Combine the bullet lists in session state under keys 'pros', 'cons', "
        "and 'risks' into a single short verdict paragraph (3-4 sentences)."
    ),
    output_key="verdict",
)

pipeline = SequentialAgent(
    name="parallel_pipeline",
    sub_agents=[fanout, synthesizer],
)


async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(app_name=APP_NAME, agent=pipeline, session_service=session_service)

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Adopting agentic AI inside an enterprise.")],
    )

    last_text = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message
    ):
        if event.content and event.content.parts:
            last_text = "".join(p.text or "" for p in event.content.parts)

    return last_text


print(asyncio.run(run()))
