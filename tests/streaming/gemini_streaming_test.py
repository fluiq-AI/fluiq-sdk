import os
from google import genai
from ..keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Count from 1 to 5.",
)

for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()
