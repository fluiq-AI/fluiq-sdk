from openai import OpenAI
from dotenv import load_dotenv
from fluiq import instrument, trace

instrument(api_key="YOUR_KEY")

load_dotenv()

client = OpenAI()


@trace
def res():
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "user", "content": "What is 27 * 43? Show your reasoning step by step."}
        ],
        reasoning_effort="medium",
    )

    return response.choices[0].message.content


print(res())
