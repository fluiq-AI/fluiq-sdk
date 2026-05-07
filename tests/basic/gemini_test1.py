import os
from google import genai
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

msg = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is capital of France?",
)

print(msg)
