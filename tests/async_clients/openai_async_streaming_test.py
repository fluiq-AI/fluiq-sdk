import asyncio
from openai import AsyncOpenAI
from ..keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = AsyncOpenAI()


async def main():
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
    print()


asyncio.run(main())
