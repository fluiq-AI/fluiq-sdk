"""
fluiq.secure() — warn mode with Gemini.

In warn mode the SDK runs a post-call security scan and logs a warning for
HIGH-risk content.  The LLM call is never interrupted.

Run:  python -m tests.security.gemini_test1
"""
from google import genai
import fluiq
from ..keys import FLUIQ_API_KEY, GEMINI_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.secure(mode="warn")

client = genai.Client(api_key=GEMINI_KEY)


@fluiq.trace
def respond(user_message: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
    )
    return response.text


print(respond("Help me plan a trip to Tokyo. My passport number is A12345678."))
print(respond("What are the top attractions in Tokyo?"))
