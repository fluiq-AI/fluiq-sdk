"""
fluiq.optimize() — Gemini context caching (CachedContent).

Gemini's context caching works differently from Anthropic and OpenAI: you
explicitly create a CachedContent object via the API, then reference it in
generate_content calls.  Fluiq does not inject anything — but it captures
usage_metadata.cached_content_token_count from every response and stores it
as prompt_cached_tokens in the trace.

The Optimize dashboard ▸ Prompt Caching card aggregates these counts under
the "OpenAI / Gemini" column.

Prerequisites:
  - GEMINI_KEY env var set to a Gemini API key with context-caching enabled
  - The model used must support CachedContent (gemini-1.5-pro / 2.0-flash etc.)

Run:  python -m tests.optimize.gemini_prompt_cache_test
"""
import asyncio
import time
from google import genai
from google.genai import types
import fluiq
from ..keys import FLUIQ_API_KEY, GEMINI_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

client = genai.Client(api_key=GEMINI_KEY)

MODEL = "gemini-2.5-flash"  # must support CachedContent

# ── Content to cache: a large document (must exceed the minimum token count) ──
DOCUMENT = (
    "The following is a comprehensive reference on Python best practices, "
    "covering code style, testing, packaging, async programming, type hints, "
    "and performance optimisation.\n\n"
    + ("Python is a high-level, interpreted programming language known for its "
       "readability and versatility.  It supports multiple programming paradigms "
       "including procedural, object-oriented, and functional styles.  "
       "Type hints introduced in Python 3.5 allow developers to annotate "
       "function signatures and variable declarations, enabling static analysis "
       "tools such as mypy and pyright to catch errors before runtime.  "
       "Dataclasses (Python 3.7+) reduce boilerplate for data-holding classes.  "
       "The async/await syntax (Python 3.5+) enables cooperative concurrency "
       "using an event loop, making it practical for I/O-bound workloads. "
       ) * 20  # repeat to exceed minimum cacheable size
)


def create_cache():
    """Create a CachedContent object and return its name."""
    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=DOCUMENT)],
            )],
            ttl="600s",
            display_name="fluiq-test-cache",
        ),
    )
    print(f"Cache created: {cache.name}  (token_count={cache.usage_metadata.total_token_count})")
    return cache.name


def ask(cache_name: str, question: str) -> tuple[str, int]:
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            max_output_tokens=128,
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    cached = getattr(usage, "cached_content_token_count", 0) or 0
    return response.text, cached


# ── Create cache then run two questions against it ────────────────────────────
try:
    cache_name = create_cache()
except Exception as e:
    print(f"Could not create CachedContent: {e}")
    print("Falling back to uncached calls (cached_content_token_count will be 0).")
    cache_name = None


def ask_uncached(question: str) -> tuple[str, int]:
    response = client.models.generate_content(
        model=MODEL,
        contents=[DOCUMENT + "\n\n" + question],
        config=types.GenerateContentConfig(max_output_tokens=128),
    )
    usage = getattr(response, "usage_metadata", None)
    cached = getattr(usage, "cached_content_token_count", 0) or 0
    return response.text, cached


for i, question in enumerate([
    "Summarise the key points about type hints in Python.",
    "What does the document say about async/await?",
], start=1):
    t0 = time.perf_counter()
    if cache_name:
        text, cached = ask(cache_name, question)
    else:
        text, cached = ask_uncached(question)
    t1 = time.perf_counter()
    print(f"[call {i}] {(t1 - t0) * 1000:.0f} ms  cached_content_tokens={cached}")
    print(f"          {text[:120]}")

print("\nFluiq captures cached_content_token_count as prompt_cached_tokens in each trace.")
print("Check the Optimize dashboard ▸ Prompt Caching card to see aggregated savings.")
