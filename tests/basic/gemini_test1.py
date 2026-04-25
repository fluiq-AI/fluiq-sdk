import os
from google import genai
import fluiq
from dotenv import load_dotenv

fluiq.instrument(api_key="your-fluiq-key")

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

msg = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is capital of France?",
)

print(msg)
