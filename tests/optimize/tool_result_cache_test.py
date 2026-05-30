"""
fluiq.optimize() — tool result caching via @fluiq.trace and fluiq.lookup_tool_result.

There are two ways to cache tool execution results:

1. @fluiq.trace decorator — the decorator automatically checks the cache before
   executing the function and stores the result after a real execution.
   No changes to your tool code are needed beyond adding the decorator.

2. fluiq.lookup_tool_result() — a manual cache check you can insert at the
   start of any tool function, giving you explicit control over the cache path.

Both approaches use the same Redis backend (keyed by tool_name + sorted args)
and appear in the Optimize dashboard ▸ By cache type ▸ function.

Run:  python -m tests.optimize.tool_result_cache_test
"""
import time
import fluiq
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()


# ── Approach 1: @fluiq.trace — fully automatic ────────────────────────────────

@fluiq.trace
def fetch_weather(city: str) -> dict:
    """Simulate an expensive external API call."""
    print(f"  [real call] fetching weather for {city!r} …")
    time.sleep(0.05)  # simulate network latency
    return {"city": city, "temp_c": 18, "condition": "partly cloudy"}


print("=== Approach 1: @fluiq.trace ===")

t0 = time.perf_counter()
w1 = fetch_weather(city="London")
t1 = time.perf_counter()
print(f"[call 1] {(t1 - t0) * 1000:.0f} ms  result={w1}")

t2 = time.perf_counter()
w2 = fetch_weather(city="London")   # same args → served from cache
t3 = time.perf_counter()
print(f"[call 2] {(t3 - t2) * 1000:.0f} ms  result={w2}")

assert w1 == w2, "Cache returned different result!"
print("OK — results match; call 2 skipped real execution.\n")


# ── Approach 2: fluiq.lookup_tool_result — manual ────────────────────────────

def fetch_stock_price(ticker: str) -> dict:
    """Manually checks the cache before executing the expensive call."""
    cached = fluiq.lookup_tool_result("fetch_stock_price", {"ticker": ticker})
    if cached is not None:
        print(f"  [cache hit] serving cached result for {ticker!r}")
        return cached

    print(f"  [real call] fetching stock price for {ticker!r} …")
    time.sleep(0.05)
    result = {"ticker": ticker, "price_usd": 182.34, "change_pct": 0.42}

    # Populate the cache so the next call is served from Redis.
    from fluiq.optimization.client import populate_tool_cache
    populate_tool_cache("fetch_stock_price", {"ticker": ticker}, result)
    return result


print("=== Approach 2: fluiq.lookup_tool_result ===")

t4 = time.perf_counter()
s1 = fetch_stock_price(ticker="AAPL")
t5 = time.perf_counter()
print(f"[call 1] {(t5 - t4) * 1000:.0f} ms  result={s1}")

t6 = time.perf_counter()
s2 = fetch_stock_price(ticker="AAPL")   # cache hit
t7 = time.perf_counter()
print(f"[call 2] {(t7 - t6) * 1000:.0f} ms  result={s2}")

assert s1 == s2, "Cache returned different result!"
print("OK — results match; call 2 served from cache.\n")

print("Check Optimize dashboard ▸ By cache type ▸ function for hit/miss counts.")
