"""
fluiq.optimize() — observe mode with OpenAI.

In observe mode the SDK records what *would* have been a cache hit without
intercepting any calls.  Use this to review potential savings before enabling
full caching.

Run:  python -m tests.optimize.openai_test2
"""
from openai import OpenAI
from ..keys import FLUIQ_API_KEY
import fluiq

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize(mode="observe")

client = OpenAI()


@fluiq.trace
def summarise(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarise in one sentence."},
            {"role": "user", "content": topic},
        ],
    )
    return response.choices[0].message.content


print(summarise("the history of the internet")) #
print(summarise("the history of the internet"))  # 
