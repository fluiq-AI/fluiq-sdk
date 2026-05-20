"""
fluiq.optimize() — cache mode with Anthropic.

Run:  python -m tests.optimize.anthropic_test1
"""
import time
import anthropic
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

client = anthropic.Anthropic()

def ask():
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is the speed of light in km/s?"}],
    )
    return msg.content[0].text


t0 = time.perf_counter()
r1 = ask()
t1 = time.perf_counter()
print(f"[call 1] {r1}  ({(t1 - t0) * 1000:.0f} ms)") # 1520

t2 = time.perf_counter()
r2 = ask()
t3 = time.perf_counter()
print(f"[call 2] {r2}  ({(t3 - t2) * 1000:.0f} ms)") # 30
