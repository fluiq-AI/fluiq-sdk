import anthropic
import fluiq
from dotenv import load_dotenv

fluiq.instrument(api_key="your-fluiq-key")

load_dotenv()

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
