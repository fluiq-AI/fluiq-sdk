"""
fluiq.optimize() — embedding cache with Qdrant.

Spins up against a local Qdrant instance (docker run -p 6333:6333 qdrant/qdrant)
or a remote one via QDRANT_URL / QDRANT_API_KEY in .env.

Run:  python -m tests.optimize.qdrant_test1
"""
import time
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import fluiq
from ..keys import FLUIQ_API_KEY, QDRANT_API_KEY, QDRANT_URL

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

DIM = 64
COLLECTION = "fluiq_test"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Recreate the collection each run so the test is idempotent.
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)

rng = np.random.default_rng(7)
points = [
    PointStruct(id=str(uuid.uuid4()), vector=rng.random(DIM).tolist(), payload={"text": f"doc{i}"})
    for i in range(50)
]
client.upsert(collection_name=COLLECTION, points=points)

QUERY_VECTOR = rng.random(DIM).tolist()
TOP_K = 3


def search():
    result = client.query_points(collection_name=COLLECTION, query=QUERY_VECTOR, limit=TOP_K)
    return result.points


# First call — cache miss
t0 = time.perf_counter()
r1 = search()
t1 = time.perf_counter()
ids1 = [str(p.id) for p in r1]
print(f"[call 1] ids={ids1}  ({(t1 - t0) * 1000:.0f} ms)")

# Second identical call — cache hit
t2 = time.perf_counter()
r2 = search()
t3 = time.perf_counter()
ids2 = [str(p.id) for p in r2]
print(f"[call 2] ids={ids2}  ({(t3 - t2) * 1000:.0f} ms)")

assert ids1 == ids2, "Cache returned different results!"
print("OK — results match")
