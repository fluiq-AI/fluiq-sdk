"""
fluiq.optimize() - embedding cache with Weaviate.

Requires a local Weaviate instance:
  docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest

Or point WEAVIATE_URL / WEAVIATE_API_KEY at a Weaviate Cloud cluster.

Run:  python -m tests.optimize.weaviate_test1
"""
import time

import numpy as np
import weaviate
import weaviate.classes as wvc
import fluiq

from ..keys import FLUIQ_API_KEY, WEAVIATE_API_KEY, WEAVIATE_URL

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
fluiq.optimize()

# -- Connect ------------------------------------------------------------------
if WEAVIATE_API_KEY:
    import weaviate.auth as wv_auth
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=wv_auth.AuthApiKey(WEAVIATE_API_KEY),
    )
else:
    client = weaviate.connect_to_local()

COLLECTION_NAME = "FluiqTest"
DIM = 32
VEC_NAME = "fluiq_vec"

rng = np.random.default_rng(42)

# Recreate the collection each run so the test is idempotent.
if client.collections.exists(COLLECTION_NAME):
    client.collections.delete(COLLECTION_NAME)

# self_provided creates a named-vector slot; no external vectorizer module needed.
collection = client.collections.create(
    name=COLLECTION_NAME,
    vector_config=wvc.config.Configure.Vectors.self_provided(name=VEC_NAME),
    properties=[wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT)],
)

docs = [
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Neural networks are inspired by the brain.",
]
collection.data.insert_many([
    wvc.data.DataObject(properties={"text": t}, vector={VEC_NAME: rng.random(DIM).tolist()})
    for t in docs
])
time.sleep(0.5)  # allow Weaviate Cloud to index newly inserted objects

# near_vector takes a plain list; the named target is selected via target_vector=.
QUERY_VECTOR = rng.random(DIM).tolist()


def search():
    return collection.query.near_vector(
        near_vector=QUERY_VECTOR,
        target_vector=VEC_NAME,
        limit=2,
    )


# First call - cache miss, real Weaviate query
t0 = time.perf_counter()
r1 = search()
t1 = time.perf_counter()
uuids1 = [str(o.uuid) for o in r1.objects]
print(f"[call 1] uuids={uuids1}  ({(t1 - t0) * 1000:.0f} ms)")

# Second identical call - served from Redis cache
t2 = time.perf_counter()
r2 = search()
t3 = time.perf_counter()
uuids2 = [str(o.uuid) for o in r2.objects]
print(f"[call 2] uuids={uuids2}  ({(t3 - t2) * 1000:.0f} ms)")

client.close()

assert uuids1 == uuids2, "Cache returned different results!"
assert len(uuids1) > 0, "Query returned no results - check collection setup"
print("OK - results match")
