"""
fluiq.optimize() — embedding cache with FAISS (in-memory).

FAISS search returns (distances, indices) numpy arrays.  On a cache hit the
arrays are reconstructed from Redis and returned with ~0 ms latency.

Run:  python -m tests.optimize.faiss_test1
"""
import time

import numpy as np
import faiss
import fluiq

from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

# ── Build a small flat L2 index ──────────────────────────────────────────────
DIM = 64
N_VECTORS = 100

rng = np.random.default_rng(42)
vectors = rng.random((N_VECTORS, DIM)).astype(np.float32)

index = faiss.IndexFlatL2(DIM)
index.add(vectors)

QUERY = rng.random((1, DIM)).astype(np.float32)
K = 5


def search():
    return index.search(QUERY, K)


# First call — cache miss, real FAISS search
t0 = time.perf_counter()
D1, I1 = search()
t1 = time.perf_counter()
print(f"[call 1] top-{K} indices={I1[0].tolist()}  ({(t1 - t0) * 1000:.0f} ms)")

# Second identical call — served from Redis cache
t2 = time.perf_counter()
D2, I2 = search()
t3 = time.perf_counter()
print(f"[call 2] top-{K} indices={I2[0].tolist()}  ({(t3 - t2) * 1000:.0f} ms)")

assert np.array_equal(I1, I2), "Cache returned different indices!"
assert np.allclose(D1, D2, atol=1e-5), "Cache returned different distances!"
print("OK — results match")
