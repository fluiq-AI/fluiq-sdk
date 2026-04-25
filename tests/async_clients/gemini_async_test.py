import asyncio
import os
from google import genai
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))


async def main():
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is the capital of France?",
    )
    print(response.text)


asyncio.run(main())
