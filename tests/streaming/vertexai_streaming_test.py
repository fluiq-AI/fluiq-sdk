import os
import vertexai
from vertexai.generative_models import GenerativeModel
from keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

vertexai.init(project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION", "us-central1"))

model = GenerativeModel("gemini-2.0-flash")

stream = model.generate_content("Count from 1 to 5.", stream=True)

for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()
