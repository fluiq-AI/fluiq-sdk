import anthropic
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 2000},
    messages=[
        {"role": "user", "content": "What is 27 * 43? Show your reasoning step by step."}
    ],
)

print(msg)
