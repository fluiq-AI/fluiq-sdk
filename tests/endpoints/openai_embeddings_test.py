from openai import OpenAI
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["The capital of France is Paris.", "Berlin is the capital of Germany."],
)

print("DIMENSIONS:", len(response.data[0].embedding))
print("USAGE:", response.usage)
