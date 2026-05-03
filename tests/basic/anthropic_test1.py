import anthropic
import fluiq
from keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY)

client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    messages=[ 
        {"role": "user", "content": "What is capital of France?"}
    ],
)

print(msg)