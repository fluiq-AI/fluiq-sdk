import anthropic
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = anthropic.Anthropic()

result = client.messages.count_tokens(
    model="claude-sonnet-4-5",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print("INPUT_TOKENS:", result.input_tokens)
