from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = OpenAI()

out_path = Path(__file__).parent / "speech.mp3"

response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="The capital of France is Paris.",
)
out_path.write_bytes(response.read())

print("WROTE:", out_path, out_path.stat().st_size, "bytes")
