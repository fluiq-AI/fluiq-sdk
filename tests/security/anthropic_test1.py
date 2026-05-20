"""
fluiq.secure() — warn mode with Anthropic.

In warn mode the SDK runs a post-call security scan and logs a warning for
HIGH-risk content.  The LLM call is never interrupted.

Run:  python -m tests.security.anthropic_test1
"""
import anthropic
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.secure(mode="warn")

client = anthropic.Anthropic()


def respond(user_message: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": user_message}],
    )
    return msg.content[0].text


print(respond("Book tickets to Paris. My name is Alice Chen and my card number is 4111111111111111."))
# print(respond("What are the best travel tips for visiting France?"))
