from openai import OpenAI
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = OpenAI()

response = client.images.generate(
    model="gpt-image-1",
    prompt="A small red apple on a white plate, photorealistic.",
    size="1024x1024",
    n=1,
)

print("IMAGE_COUNT:", len(response.data))
