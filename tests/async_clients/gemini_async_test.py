import asyncio
import os
from google import genai
from keys import FLUIQ_API_KEY, GEMINI_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

client = genai.Client(api_key=GEMINI_KEY)


async def main():
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is the capital of France?",
    )
    print(response.text)


asyncio.run(main())
