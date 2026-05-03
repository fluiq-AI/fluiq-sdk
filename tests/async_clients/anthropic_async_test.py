import asyncio
import anthropic
from keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

client = anthropic.AsyncAnthropic()


async def main():
    msg = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(msg.content[0].text)


asyncio.run(main())
