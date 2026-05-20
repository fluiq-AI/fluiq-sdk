"""
fluiq.optimize() — cache mode with Gemini.

Run:  python -m tests.optimize.gemini_test1
"""
import os
import time
from google import genai
import fluiq
from ..keys import FLUIQ_API_KEY, GEMINI_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

client = genai.Client(api_key=GEMINI_KEY)


@fluiq.trace
def ask():
    msg = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is the chemical formula for water?",
    )
    return msg.text


t0 = time.perf_counter()
r1 = ask()
t1 = time.perf_counter()
print(f"[call 1] {r1}  ({(t1 - t0) * 1000:.0f} ms)")  # 864

t2 = time.perf_counter()
r2 = ask()
t3 = time.perf_counter()
print(f"[call 2] {r2}  ({(t3 - t2) * 1000:.0f} ms)") # 790
