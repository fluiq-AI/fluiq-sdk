import anthropic
import fluiq
from dotenv import load_dotenv

fluiq.instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")

load_dotenv()

client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    messages=[ 
        {"role": "user", "content": "What is capital of France?"}
    ],
)

print(msg)