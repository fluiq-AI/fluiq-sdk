import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = AsyncOpenAI()


async def main():
    response = await client.responses.create(
        model="gpt-4o-mini",
        input="What is the capital of France?",
    )
    print(response.output_text)


asyncio.run(main())
