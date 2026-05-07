import asyncio
from openai import AsyncOpenAI
from ..keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = AsyncOpenAI()


async def main():
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(response.choices[0].message.content)


asyncio.run(main())
