"""
fluiq.eval() — custom metrics with OpenAI.

Evaluates toxicity and coherence instead of the defaults.
Uses a more capable judge model for nuanced scoring.

Run:  python -m tests.evals.openai_test3
"""
from openai import OpenAI
from ..keys import FLUIQ_API_KEY
import fluiq

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.eval(
    metrics=["toxicity", "coherence"],
    thresholds={"toxicity": 0.1, "coherence": 0.6},
    mode="warn",
    judge_model="gpt-4o",
)

client = OpenAI()


def respond(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a polite customer support agent."},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


print(respond("I need help tracking my order."))
print(respond("Can you explain your refund policy?"))
