"""
fluiq.optimize() — embedding cache with Pinecone.

Requires PINECONE_API_KEY and PINECONE_INDEX_HOST in .env pointing at a
pre-existing index.  The first query hits Pinecone; the second identical
query is served from Redis.

Run:  python -m tests.optimize.pinecone_test1
"""
import time

import fluiq
from pinecone import Pinecone

from ..keys import FLUIQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_HOST

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

# A fixed query vector — adjust dimension to match your index.
import numpy as np
rng = np.random.default_rng(0)
QUERY_VECTOR = rng.random(index.describe_index_stats()["dimension"]).tolist()
TOP_K = 3


@fluiq.trace
def search():
    return index.query(vector=QUERY_VECTOR, top_k=TOP_K, include_metadata=True)


# First call — cache miss
t0 = time.perf_counter()
r1 = search()
t1 = time.perf_counter()
ids1 = [m.id for m in r1.matches]
print(f"[call 1] ids={ids1}  ({(t1 - t0) * 1000:.0f} ms)")

# Second identical call — cache hit
t2 = time.perf_counter()
r2 = search()
t3 = time.perf_counter()
ids2 = [m.id for m in r2.matches]
print(f"[call 2] ids={ids2}  ({(t3 - t2) * 1000:.0f} ms)")

assert ids1 == ids2, "Cache returned different results!"
print("OK — results match")
