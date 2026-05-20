"""
fluiq.eval() — warn mode with OpenAI.

After each LLM call the SDK sends the prompt and response to Fluiq's
LLM-as-judge in the background.  Scores are stored in your dashboard.
A warning is logged if any metric falls below its threshold; the call
is never interrupted.

Run:  python -m tests.evals.openai_test1
"""
from openai import OpenAI
from ..keys import FLUIQ_API_KEY
import fluiq

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.eval(
    metrics=["hallucination", "relevance"],
    thresholds={"hallucination": 0.7, "relevance": 0.6},
    mode="warn",
)

client = OpenAI()


@fluiq.trace
def answer(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful, factual assistant."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


print(answer("What year did the first moon landing happen?"))
print(answer("Who invented the telephone?"))
