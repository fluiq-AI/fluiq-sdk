from openai import OpenAI
from keys import FLUIQ_API_KEY
from fluiq import instrument

instrument(api_key=FLUIQ_API_KEY)

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["The capital of France is Paris.", "Berlin is the capital of Germany."],
)

print("DIMENSIONS:", len(response.data[0].embedding))
print("USAGE:", response.usage)
