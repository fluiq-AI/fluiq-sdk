import anthropic
from ..keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = anthropic.Anthropic()

stream = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)

for event in stream:
    etype = getattr(event, "type", None)
    if etype == "content_block_delta":
        delta = getattr(event, "delta", None)
        text = getattr(delta, "text", None)
        if text:
            print(text, end="", flush=True)
print()
