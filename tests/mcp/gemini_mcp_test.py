import asyncio
import os
from dotenv import load_dotenv
import fluiq

fluiq.instrument(api_key="YOUR_KEY")
load_dotenv()

from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))


async def run():
    async with streamablehttp_client("https://mcp.deepwiki.com/mcp") as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents="What transport protocols does the MCP spec support?",
                config=types.GenerateContentConfig(tools=[session]),
            )
            return response.text


print(asyncio.run(run()))
