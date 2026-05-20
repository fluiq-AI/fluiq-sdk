"""
fluiq.eval() — warn mode with Anthropic.

Run:  python -m tests.evals.anthropic_test1
"""
import anthropic
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.eval(
    metrics=["hallucination", "relevance"],
    thresholds={"hallucination": 0.7, "relevance": 0.6},
    mode="warn",
)

client = anthropic.Anthropic()


def answer(question: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text


print(answer("What is the speed of light?"))
print(answer("Who wrote Romeo and Juliet?"))
