import os
from google import genai
from keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

result = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents="What is the capital of France?",
)

print("TOTAL_TOKENS:", result.total_tokens)
