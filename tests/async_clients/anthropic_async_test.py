import asyncio
import anthropic
from tests.keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = anthropic.AsyncAnthropic()


async def main():
    msg = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(msg.content[0].text)


asyncio.run(main())
