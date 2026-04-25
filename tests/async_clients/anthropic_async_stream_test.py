import asyncio
import anthropic
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = anthropic.AsyncAnthropic()


async def main():
    async with client.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
        final = await stream.get_final_message()
    print()
    print("STOP_REASON:", final.stop_reason)


asyncio.run(main())
