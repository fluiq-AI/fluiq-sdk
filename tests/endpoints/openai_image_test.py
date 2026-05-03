from openai import OpenAI
from keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

client = OpenAI()

response = client.images.generate(
    model="gpt-image-1",
    prompt="A small red apple on a white plate, photorealistic.",
    size="1024x1024",
    n=1,
)

print("IMAGE_COUNT:", len(response.data))
