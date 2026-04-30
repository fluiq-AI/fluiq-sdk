from typing import Any

from fluiq.integrations.shared.models import TraceType
from fluiq.integrations.Vectorstores.shared.utils import (
    _build_match,
    _capture_matches,
    _capture_query_texts,
    _safe_jsonable,
    _truncate_ids,
    _vector_count_and_dim,
    make_sync_wrapper,
)


def _target(instance) -> dict:
    return {"collection": getattr(instance, "name", None)}


def _summarize_query(args, kwargs, instance, response=None) -> dict:
    qe = kwargs.get("query_embeddings")
    qt = kwargs.get("query_texts")
    vc = _vector_count_and_dim(qe)
    out = {
        "target": _target(instance),
        "query": {
            "top_k": kwargs.get("n_results"),
            "vector_count": vc["count"],
            "vector_dim": vc["dim"],
            "texts": _capture_query_texts(qt),
            "filter": _safe_jsonable(kwargs.get("where")),
            "where_document": _safe_jsonable(kwargs.get("where_document")),
            "include": kwargs.get("include"),
        },
    }
    if isinstance(response, dict):
        ids = response.get("ids") or []
        distances = response.get("distances") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        matches: list = []
        flat_d: list[float] = []
        for qi, sub_ids in enumerate(ids if isinstance(ids, list) else []):
            if not isinstance(sub_ids, list):
                continue
            sub_d = distances[qi] if qi < len(distances) and isinstance(distances[qi], list) else []
            sub_doc = documents[qi] if qi < len(documents) and isinstance(documents[qi], list) else []
            sub_md = metadatas[qi] if qi < len(metadatas) and isinstance(metadatas[qi], list) else []
            for j, mid in enumerate(sub_ids):
                dist = sub_d[j] if j < len(sub_d) else None
                if isinstance(dist, (int, float)):
                    flat_d.append(dist)
                matches.append(_build_match(
                    id=mid,
                    score=dist,
                    text=sub_doc[j] if j < len(sub_doc) else None,
                    metadata=sub_md[j] if j < len(sub_md) else None,
                ))
        out["result"] = {
            "matches": _capture_matches(matches),
            "distance_min": min(flat_d) if flat_d else None,
            "distance_max": max(flat_d) if flat_d else None,
        }
    return out


def _summarize_mutation(args, kwargs, instance, response=None) -> dict:
    ids = kwargs.get("ids")
    embeddings = kwargs.get("embeddings")
    documents = kwargs.get("documents")
    vc = _vector_count_and_dim(embeddings)
    return {
        "target": _target(instance),
        "mutation": {
            "ids": _truncate_ids(ids),
            "vector_count": vc["count"],
            "vector_dim": vc["dim"],
            "document_count": len(documents) if isinstance(documents, list) else None,
        },
    }


def _summarize_get(args, kwargs, instance, response=None) -> dict:
    out = {
        "target": _target(instance),
        "query": {
            "ids": _truncate_ids(kwargs.get("ids")),
            "filter": _safe_jsonable(kwargs.get("where")),
            "where_document": _safe_jsonable(kwargs.get("where_document")),
            "limit": kwargs.get("limit"),
            "offset": kwargs.get("offset"),
        },
    }
    if isinstance(response, dict):
        out["result"] = {"ids": _truncate_ids(response.get("ids"))}
    return out


def _summarize_count(args, kwargs, instance, response=None) -> dict:
    out = {"target": _target(instance)}
    if isinstance(response, int):
        out["result"] = {"count": response}
    return out


def _summarize_delete(args, kwargs, instance, response=None) -> dict:
    return {
        "target": _target(instance),
        "mutation": {
            "ids": _truncate_ids(kwargs.get("ids")),
            "filter": _safe_jsonable(kwargs.get("where")),
        },
    }


def patch_chromadb():
    try:
        from chromadb.api.models.Collection import Collection
    except Exception:
        return

    _wrap = lambda method, api, summarize: make_sync_wrapper(  # noqa: E731
        method, TraceType.ChromaDB, api, summarize,
    )

    if hasattr(Collection, "query"):
        Collection.query = _wrap(Collection.query, "query", _summarize_query)
    if hasattr(Collection, "add"):
        Collection.add = _wrap(Collection.add, "add", _summarize_mutation)
    if hasattr(Collection, "upsert"):
        Collection.upsert = _wrap(Collection.upsert, "upsert", _summarize_mutation)
    if hasattr(Collection, "update"):
        Collection.update = _wrap(Collection.update, "update", _summarize_mutation)
    if hasattr(Collection, "get"):
        Collection.get = _wrap(Collection.get, "get", _summarize_get)
    if hasattr(Collection, "count"):
        Collection.count = _wrap(Collection.count, "count", _summarize_count)
    if hasattr(Collection, "delete"):
        Collection.delete = _wrap(Collection.delete, "delete", _summarize_delete)


def patch_chromadb_async():
    try:
        from chromadb.api.models.AsyncCollection import AsyncCollection
    except Exception:
        return
    from fluiq.integrations.Vectorstores.shared.utils import make_async_wrapper

    _wrap = lambda method, api, summarize: make_async_wrapper(  # noqa: E731
        method, TraceType.ChromaDB, api, summarize,
    )

    for attr, api, summarize in (
        ("query", "query", _summarize_query),
        ("add", "add", _summarize_mutation),
        ("upsert", "upsert", _summarize_mutation),
        ("update", "update", _summarize_mutation),
        ("get", "get", _summarize_get),
        ("count", "count", _summarize_count),
        ("delete", "delete", _summarize_delete),
    ):
        if hasattr(AsyncCollection, attr):
            setattr(AsyncCollection, attr, _wrap(getattr(AsyncCollection, attr), api, summarize))
