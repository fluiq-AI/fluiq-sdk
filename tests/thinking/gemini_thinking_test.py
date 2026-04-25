import os
from google import genai
from google.genai import types
import fluiq
from dotenv import load_dotenv

fluiq.instrument(api_key="your-fluiq-key")

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

msg = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 27 * 43? Show your reasoning step by step.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
    ),
)

print(msg)
