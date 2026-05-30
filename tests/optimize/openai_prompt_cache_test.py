"""
fluiq.optimize() — OpenAI provider prompt caching.

OpenAI automatically caches the prompt prefix for requests whose prompt
is >= 1 024 tokens.  No injection is needed; Fluiq captures the saved token
count from every response and stores it as prompt_cached_tokens in the trace.

The Optimize dashboard ▸ Prompt Caching card aggregates these counts across
all calls in the selected time window.

Run:  python -m tests.optimize.openai_prompt_cache_test
"""
import time
from openai import OpenAI
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

client = OpenAI()

# System message must push the full prompt past ~1 024 tokens for OpenAI to
# cache the prefix automatically.
SYSTEM = (
    "You are a world-class software architect with deep expertise in distributed "
    "systems, microservices, event-driven architectures, and cloud-native "
    "infrastructure.  You have designed systems that process billions of events "
    "per day and have led engineering teams at top-tier technology companies. "
    "When answering questions you provide precise, actionable advice backed by "
    "real-world experience.  You cite trade-offs honestly and never oversimplify "
    "complex engineering decisions.  You are fluent in Python, Go, Java, and "
    "TypeScript and have deep knowledge of AWS, GCP, and Azure services, "
    "Kubernetes, Kafka, Redis, PostgreSQL, and modern observability tooling. "
    * 5  # push past 1 024 tokens
)

MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "What is the CAP theorem and how does it apply to distributed databases?"},
]


def ask(user_content: str) -> tuple[str, int]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=128,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return response.choices[0].message.content, cached


# ── Call 1: cache miss ────────────────────────────────────────────────────────
t0 = time.perf_counter()
answer1, cached1 = ask("What is the CAP theorem?")
t1 = time.perf_counter()
print(f"[call 1] {(t1 - t0) * 1000:.0f} ms  cached_tokens={cached1}")
print(f"         {answer1[:120]}")

# ── Call 2: same system prefix — OpenAI may serve prefix tokens from cache ───
t2 = time.perf_counter()
answer2, cached2 = ask("How does the CAP theorem affect database design choices?")
t3 = time.perf_counter()
print(f"[call 2] {(t3 - t2) * 1000:.0f} ms  cached_tokens={cached2}")
print(f"         {answer2[:120]}")

if cached2 > 0:
    print(f"\nOK — {cached2} prompt tokens served from OpenAI's cache on call 2.")
else:
    print("\nNOTE — cached_tokens=0; OpenAI's cache may need the prompt to reach the "
          "minimum token threshold or a brief warm-up window.")
