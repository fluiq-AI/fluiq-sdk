import os
from google import genai
import fluiq
from dotenv import load_dotenv

fluiq.instrument(api_key="fl_zLBKMj9NVmILlOUN32awEuYNso_t45u48ggMPZ-Kkqk")

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

msg = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is capital of France?",
)

print(msg)
