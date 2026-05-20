from typing import Any

from types import SimpleNamespace

from fluiq.integrations.shared.models import TraceType
from fluiq.integrations.Vectorstores.shared.utils import (
    _build_match,
    _capture_matches,
    _safe_jsonable,
    _truncate_ids,
    _vector_dim,
    make_async_cached_wrapper,
    make_async_invalidating_wrapper,
    make_async_wrapper,
    make_sync_cached_wrapper,
    make_sync_invalidating_wrapper,
    make_sync_wrapper,
)
from fluiq.optimization.client import vectorstore_cache_key


def _pinecone_target(args, kwargs, instance) -> str:
    return (
        getattr(instance, "name", None)
        or getattr(instance, "_index_name", None)
        or getattr(getattr(instance, "config", None), "name", None)
        or ""
    )


def _target(instance, kwargs) -> dict:
    name = (
        getattr(instance, "name", None)
        or getattr(instance, "_index_name", None)
        or getattr(getattr(instance, "config", None), "name", None)
    )
    return {"index": name, "namespace": kwargs.get("namespace")}


def _vectors_summary(vectors: Any) -> dict:
    if vectors is None:
        return {"count": None, "dim": None}
    try:
        seq = list(vectors)
    except TypeError:
        return {"count": None, "dim": None}
    count = len(seq)
    dim = None
    if seq:
        first = seq[0]
        values = None
        if isinstance(first, dict):
            values = first.get("values") or first.get("sparse_values")
        else:
            values = getattr(first, "values", None)
        dim = _vector_dim(values)
    return {"count": count, "dim": dim}


def _summarize_query(args, kwargs, instance, response=None) -> dict:
    vec = kwargs.get("vector")
    out = {
        "target": _target(instance, kwargs),
        "query": {
            "top_k": kwargs.get("top_k"),
            "vector_dim": _vector_dim(vec),
            "has_vector": vec is not None,
            "id": kwargs.get("id"),
            "filter": _safe_jsonable(kwargs.get("filter")),
            "include_values": kwargs.get("include_values"),
            "include_metadata": kwargs.get("include_metadata"),
        },
    }
    raw_matches = None
    if response is not None:
        raw_matches = (
            response.get("matches")
            if isinstance(response, dict)
            else getattr(response, "matches", None)
        )
    if raw_matches is not None:
        scores: list = []
        items: list = []
        for m in raw_matches:
            is_dict = isinstance(m, dict)
            mid = m.get("id") if is_dict else getattr(m, "id", None)
            sc = m.get("score") if is_dict else getattr(m, "score", None)
            md = m.get("metadata") if is_dict else getattr(m, "metadata", None)
            if isinstance(sc, (int, float)):
                scores.append(sc)
            text = None
            if isinstance(md, dict):
                for key in ("text", "content", "page_content", "chunk"):
                    val = md.get(key)
                    if isinstance(val, str):
                        text = val
                        break
            items.append(_build_match(id=mid, score=sc, text=text, metadata=md))
        out["result"] = {
            "matches": _capture_matches(items),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }
    return out


def _summarize_upsert(args, kwargs, instance, response=None) -> dict:
    vectors = kwargs.get("vectors")
    if vectors is None and args:
        vectors = args[0]
    summary = _vectors_summary(vectors)
    out = {
        "target": _target(instance, kwargs),
        "mutation": {
            "vector_count": summary["count"],
            "vector_dim": summary["dim"],
        },
    }
    if response is not None:
        upserted = (
            response.get("upserted_count")
            if isinstance(response, dict)
            else getattr(response, "upserted_count", None)
        )
        if upserted is not None:
            out["mutation"]["upserted_count"] = upserted
    return out


def _summarize_fetch(args, kwargs, instance, response=None) -> dict:
    ids = kwargs.get("ids")
    if ids is None and args:
        ids = args[0]
    out = {
        "target": _target(instance, kwargs),
        "query": {"ids": _truncate_ids(ids)},
    }
    vectors = None
    if response is not None:
        vectors = (
            response.get("vectors")
            if isinstance(response, dict)
            else getattr(response, "vectors", None)
        )
    if vectors is not None:
        try:
            out["result"] = {"count": len(vectors)}
        except TypeError:
            pass
    return out


def _summarize_delete(args, kwargs, instance, response=None) -> dict:
    ids = kwargs.get("ids")
    if ids is None and args:
        ids = args[0]
    return {
        "target": _target(instance, kwargs),
        "mutation": {
            "ids": _truncate_ids(ids),
            "filter": _safe_jsonable(kwargs.get("filter")),
            "delete_all": kwargs.get("delete_all"),
        },
    }


def _summarize_update(args, kwargs, instance, response=None) -> dict:
    vec = kwargs.get("values")
    return {
        "target": _target(instance, kwargs),
        "mutation": {
            "ids": _truncate_ids([kwargs.get("id")] if kwargs.get("id") else None),
            "vector_dim": _vector_dim(vec),
            "has_metadata": kwargs.get("set_metadata") is not None,
        },
    }


def _query_cache_key(args, kwargs, instance) -> str:
    idx = (
        getattr(instance, "name", None)
        or getattr(instance, "_index_name", None)
        or getattr(getattr(instance, "config", None), "name", None)
        or ""
    )
    return vectorstore_cache_key(
        "pinecone",
        idx,
        kwargs.get("vector") or kwargs.get("id"),
        kwargs.get("top_k"),
        kwargs.get("filter"),
    )


def _query_mock(cached_result: dict, args, kwargs, instance):
    items = (cached_result.get("matches") or {}).get("items") or []
    matches = [
        SimpleNamespace(
            id=m.get("id", ""),
            score=m.get("score"),
            metadata=m.get("metadata"),
            values=[],
            sparse_values=None,
        )
        for m in items
    ]
    return SimpleNamespace(
        matches=matches,
        namespace=kwargs.get("namespace", ""),
        usage=None,
    )


# upsert/update/delete mutate the index → invalidate query cache
_INVALIDATING_OPS = (
    ("upsert", "upsert", _summarize_upsert),
    ("update", "update", _summarize_update),
    ("delete", "delete", _summarize_delete),
)
# fetch is read-only — no invalidation needed
_PLAIN_OPS = (("fetch", "fetch", _summarize_fetch),)


def _patch_class_sync(cls):
    if hasattr(cls, "query"):
        setattr(
            cls, "query",
            make_sync_cached_wrapper(
                cls.query, TraceType.Pinecone, "query",
                _summarize_query, _query_cache_key, _query_mock,
            ),
        )
    for attr, api, summarize in _INVALIDATING_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_sync_invalidating_wrapper(
                getattr(cls, attr), TraceType.Pinecone, api, summarize, _pinecone_target,
            ))
    for attr, api, summarize in _PLAIN_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_sync_wrapper(getattr(cls, attr), TraceType.Pinecone, api, summarize))


def _patch_class_async(cls):
    if hasattr(cls, "query"):
        setattr(
            cls, "query",
            make_async_cached_wrapper(
                cls.query, TraceType.Pinecone, "query",
                _summarize_query, _query_cache_key, _query_mock,
            ),
        )
    for attr, api, summarize in _INVALIDATING_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_async_invalidating_wrapper(
                getattr(cls, attr), TraceType.Pinecone, api, summarize, _pinecone_target,
            ))
    for attr, api, summarize in _PLAIN_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_async_wrapper(getattr(cls, attr), TraceType.Pinecone, api, summarize))


def patch_pinecone():
    try:
        from pinecone.index import Index
    except Exception:
        try:
            from pinecone.data.index import Index  # type: ignore[no-redef]
        except Exception:
            try:
                from pinecone import Index  # type: ignore[no-redef]
            except Exception:
                return
    _patch_class_sync(Index)


def patch_pinecone_async():
    try:
        from pinecone.async_client import AsyncIndex
    except Exception:
        try:
            from pinecone.data.index_asyncio import _IndexAsyncio as AsyncIndex  # type: ignore[no-redef]
        except Exception:
            try:
                from pinecone import AsyncIndex  # type: ignore[no-redef]
            except Exception:
                return
    _patch_class_async(AsyncIndex)
