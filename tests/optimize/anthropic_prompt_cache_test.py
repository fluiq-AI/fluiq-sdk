"""
fluiq.optimize() — Anthropic provider prompt caching.

When optimize() is active the SDK automatically injects
cache_control: {"type": "ephemeral"} on the system prompt and the last tool
definition before every messages.create() call.  Anthropic caches the matching
prefix server-side; subsequent calls with the same system share the cached
prefix, paying only ~10% of normal input-token cost for those tokens.

Two trace fields are populated on every Anthropic call (visible in the
Optimize dashboard ▸ Prompt Caching card):
  prompt_cache_read_tokens    — tokens served from Anthropic's prefix cache
  prompt_cache_creation_tokens — tokens written to create the cache entry

Run:  python -m tests.optimize.anthropic_prompt_cache_test
"""
import time
import anthropic
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

client = anthropic.Anthropic()

# A large system prompt makes the prefix worth caching.
# Anthropic requires ~1 024 tokens minimum; this exceeds that threshold.
SYSTEM = (
    "You are an expert Python tutor with 20 years of experience teaching "
    "software engineering. You specialise in clean code, design patterns, "
    "and performance optimisation. You always provide concrete examples, "
    "explain the 'why' behind every recommendation, and tailor your answers "
    "to the student's level. You are patient, thorough, and encouraging. "
    "When reviewing code you identify bugs, suggest improvements, and explain "
    "each change. You are familiar with the entire Python standard library, "
    "popular third-party packages (NumPy, Pandas, FastAPI, SQLAlchemy, Pydantic, "
    "Pytest, etc.), and modern best practices including type hints, dataclasses, "
    "async/await, and context managers. "
    * 6  # repeat to push well past 1 024 tokens
)


def ask(question: str) -> tuple[str, int, int]:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    read = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
    creation = getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
    return msg.content[0].text, read, creation


# ── Call 1: cache miss, Anthropic writes the prefix to its cache ─────────────
t0 = time.perf_counter()
answer1, read1, creation1 = ask("What is a Python generator?")
t1 = time.perf_counter()
print(f"[call 1] {(t1 - t0) * 1000:.0f} ms")
print(f"         cache_read={read1}  cache_creation={creation1}")
print(f"         {answer1[:120]}")

# ── Call 2: same system prompt — Anthropic serves tokens from its prefix cache
t2 = time.perf_counter()
answer2, read2, creation2 = ask("What is the difference between a list and a tuple?")
t3 = time.perf_counter()
print(f"[call 2] {(t3 - t2) * 1000:.0f} ms")
print(f"         cache_read={read2}  cache_creation={creation2}")
print(f"         {answer2[:120]}")

if read2 > 0:
    print(f"\nOK — {read2} tokens served from Anthropic's prefix cache on call 2.")
else:
    print("\nNOTE — cache_read_tokens=0 on call 2; the cache entry may still be warming.")
