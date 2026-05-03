from openai import OpenAI
from keys import FLUIQ_API_KEY
from fluiq import instrument, trace

instrument(api_key=FLUIQ_API_KEY)

client = OpenAI()


@trace
def ask_llm(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


@trace
def workflow():
    a = ask_llm("What is the capital of France?")
    b = ask_llm("What is the capital of Germany?")
    return f"{a} | {b}"


print(workflow())
