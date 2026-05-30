"""
fluiq.optimize() — cache mode with OpenAI.

The first call is a real LLM request whose response gets stored in Redis.
The second identical call is served from the cache (zero latency, no LLM cost).

Run:  python -m tests.optimize.openai_test1
"""
import time
from openai import OpenAI
from ..keys import FLUIQ_API_KEY
import fluiq

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

client = OpenAI()

MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is the boiling point of water in Celsius?"},
]


def ask():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=MESSAGES,
    )
    return response.choices[0].message.content


# First call — cache miss, real LLM request
t0 = time.perf_counter()
answer1 = ask()
t1 = time.perf_counter()
print(f"[call 1] {answer1}  ({(t1 - t0) * 1000:.0f} ms)") # 1845 ms

# Second identical call — served from cache if backend has provisioned Redis
t2 = time.perf_counter()
answer2 = ask()
t3 = time.perf_counter()
print(f"[call 2] {answer2}  ({(t3 - t2) * 1000:.0f} ms)") # 767 ms
