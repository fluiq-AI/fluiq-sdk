"""
fluiq.eval() — warn mode with Gemini.

Run:  python -m tests.evals.gemini_test1
"""
from google import genai
import fluiq
from ..keys import FLUIQ_API_KEY, GEMINI_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.eval(
    metrics=["hallucination", "relevance"],
    thresholds={"hallucination": 0.7, "relevance": 0.6},
    mode="warn",
)

client = genai.Client(api_key=GEMINI_KEY)


@fluiq.trace
def answer(question: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
    )
    return response.text


print(answer("What is the largest planet in our solar system?"))
print(answer("Who painted the Mona Lisa?"))
