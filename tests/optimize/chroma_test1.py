"""
fluiq.optimize() — embedding cache with ChromaDB (in-memory).

ChromaDB computes embeddings internally on query_texts calls.  The first
query hits the collection; the second identical query is served from Redis
without touching ChromaDB at all.

Run:  python -m tests.optimize.chroma_test1
"""
import time

import chromadb
import fluiq

from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

# ── Build an in-memory collection with a few documents ──────────────────────
client = chromadb.Client()
collection = client.get_or_create_collection("fluiq_test")

collection.upsert(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "The capital of France is Paris.",
        "Python is a high-level programming language.",
        "Machine learning models learn from data.",
    ],
)

QUERY = ["What is the capital of France?"]


@fluiq.trace
def search():
    return collection.query(query_texts=QUERY, n_results=2)


# First call — cache miss, real ChromaDB query
t0 = time.perf_counter()
result1 = search()
t1 = time.perf_counter()
ids1 = result1["ids"][0]
print(f"[call 1] ids={ids1}  ({(t1 - t0) * 1000:.0f} ms)")

# Second identical call — served from Redis cache
t2 = time.perf_counter()
result2 = search()
t3 = time.perf_counter()
ids2 = result2["ids"][0]
print(f"[call 2] ids={ids2}  ({(t3 - t2) * 1000:.0f} ms)")

assert ids1 == ids2, "Cache returned different results!"
print("OK — results match")
