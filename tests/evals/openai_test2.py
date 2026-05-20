"""
fluiq.eval() — block mode with OpenAI.

In block mode evaluation runs synchronously after each LLM call.
If any metric falls below its threshold a FluiqEvalError is raised,
stopping the response from reaching the application.

Run:  python -m tests.evals.openai_test2
"""
from openai import OpenAI
from ..keys import FLUIQ_API_KEY
import fluiq
from fluiq.exceptions import FluiqEvalError

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.eval(
    metrics=["hallucination", "relevance"],
    thresholds={"hallucination": 0.7, "relevance": 0.6},
    mode="block",
    judge_model="gpt-4o-mini",
)

client = OpenAI()


def answer(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful, factual assistant."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


try:
    print(answer("What is the capital of France?"))
except FluiqEvalError as e:
    print(f"Eval blocked response — failures: {e.failures}")
    print(f"All scores: {e.scores}")
