from pathlib import Path
from openai import OpenAI
from ..keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

client = OpenAI()

out_path = Path(__file__).parent / "speech.mp3"

response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="The capital of France is Paris.",
)
out_path.write_bytes(response.read())

print("WROTE:", out_path, out_path.stat().st_size, "bytes")
