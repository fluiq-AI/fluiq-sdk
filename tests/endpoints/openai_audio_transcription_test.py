from pathlib import Path
from openai import OpenAI
from keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

client = OpenAI()

audio_path = Path(__file__).parent / "speech.mp3"
if not audio_path.exists():
    raise SystemExit(f"Run openai_audio_speech_test.py first — missing {audio_path}")

with audio_path.open("rb") as f:
    response = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=f,
    )

print("TEXT:", response.text)
