import asyncio
from openai import AsyncOpenAI
from fluiq import instrument
from tests.keys import FLUIQ_API_KEY

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = AsyncOpenAI()


async def main():
    response = await client.responses.create(
        model="gpt-4o-mini",
        input="What is the capital of France?",
    )
    print(response.output_text)


asyncio.run(main())
