from openai import OpenAI
from dotenv import load_dotenv
from fluiq import instrument

instrument(api_key="your-fluiq-key")
load_dotenv()

client = OpenAI()

with client.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)
    final = stream.get_final_completion()

print()
print("FINAL:", final.choices[0].message.content)
