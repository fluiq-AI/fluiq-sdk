"""
fluiq.eval() — block mode with Anthropic.

Run:  python -m tests.evals.anthropic_test2
"""
import anthropic
import fluiq
from fluiq.exceptions import FluiqEvalError
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.eval(
    metrics=["hallucination", "relevance"],
    thresholds={"hallucination": 0.7, "relevance": 0.6},
    mode="block",
)

client = anthropic.Anthropic()


@fluiq.trace
def answer(question: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text


try:
    print(answer("What is the capital of Germany?"))
except FluiqEvalError as e:
    print(f"Eval blocked — failures: {e.failures}")
    print(f"All scores: {e.scores}")
