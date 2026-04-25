import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = AsyncOpenAI()


async def main():
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(response.choices[0].message.content)


asyncio.run(main())
