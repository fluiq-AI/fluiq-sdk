import asyncio
import anthropic
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = anthropic.AsyncAnthropic()


async def main():
    msg = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(msg.content[0].text)


asyncio.run(main())
