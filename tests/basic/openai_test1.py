from openai import OpenAI
from dotenv import load_dotenv
from fluiq import instrument, trace

instrument(api_key="YOUR_KEY")

load_dotenv()

client = OpenAI()

@trace
def res():

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is capital of France?"}
        ]
    )

    return response.choices[0].message.content

print(res())