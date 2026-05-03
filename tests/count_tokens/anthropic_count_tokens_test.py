import anthropic
from keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

client = anthropic.Anthropic()

result = client.messages.count_tokens(
    model="claude-sonnet-4-5",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print("INPUT_TOKENS:", result.input_tokens)
