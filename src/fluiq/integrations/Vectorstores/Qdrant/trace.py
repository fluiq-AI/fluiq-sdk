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


def _qdrant_target(args, kwargs, instance) -> str:
    return kwargs.get("collection_name") or (args[0] if args else "") or ""


def _payload_text(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    for key in ("text", "content", "page_content", "chunk", "document"):
        val = payload.get(key)
        if isinstance(val, str):
            return val
    return None


def _build_matches_from_points(points: list) -> tuple:
    items: list = []
    scores: list = []
    for p in points:
        is_dict = isinstance(p, dict)
        pid = p.get("id") if is_dict else getattr(p, "id", None)
        sc = p.get("score") if is_dict else getattr(p, "score", None)
        payload = p.get("payload") if is_dict else getattr(p, "payload", None)
        if isinstance(sc, (int, float)):
            scores.append(sc)
        items.append(_build_match(
            id=pid, score=sc, text=_payload_text(payload), metadata=payload,
        ))
    return items, scores


def _coll(args, kwargs) -> Any:
    return kwargs.get("collection_name") or (args[0] if args else None)


def _target(args, kwargs) -> dict:
    return {"collection": _coll(args, kwargs)}


def _extract_points(response) -> list:
    if response is None:
        return []
    pts = (
        response.get("points")
        if isinstance(response, dict)
        else getattr(response, "points", None)
    )
    if pts is None and isinstance(response, list):
        pts = response
    return list(pts) if pts is not None else []


def _summarize_search(args, kwargs, instance, response=None) -> dict:
    vec = kwargs.get("query_vector")
    out = {
        "target": _target(args, kwargs),
        "query": {
            "top_k": kwargs.get("limit"),
            "vector_dim": _vector_dim(vec) if not isinstance(vec, str) else None,
            "vector_name": vec if isinstance(vec, str) else None,
            "filter": _safe_jsonable(kwargs.get("query_filter")),
            "with_payload": kwargs.get("with_payload"),
            "with_vectors": kwargs.get("with_vectors"),
            "score_threshold": kwargs.get("score_threshold"),
        },
    }
    if response is not None:
        pts = response if isinstance(response, list) else _extract_points(response)
        items, scores = _build_matches_from_points(pts)
        out["result"] = {
            "matches": _capture_matches(items),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }
    return out


def _summarize_query_points(args, kwargs, instance, response=None) -> dict:
    q = kwargs.get("query")
    vector_dim = None
    if isinstance(q, (list, tuple)):
        vector_dim = _vector_dim(q)
    out = {
        "target": _target(args, kwargs),
        "query": {
            "top_k": kwargs.get("limit"),
            "vector_dim": vector_dim,
            "using": kwargs.get("using"),
            "filter": _safe_jsonable(kwargs.get("query_filter")),
            "with_payload": kwargs.get("with_payload"),
            "with_vectors": kwargs.get("with_vectors"),
        },
    }
    pts = _extract_points(response)
    if pts:
        items, scores = _build_matches_from_points(pts)
        out["result"] = {
            "matches": _capture_matches(items),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }
    return out


def _summarize_upsert(args, kwargs, instance, response=None) -> dict:
    points = kwargs.get("points")
    count = None
    dim = None
    ids: list = []
    if points is not None:
        try:
            seq = list(points)
            count = len(seq)
            for p in seq:
                pid = p.get("id") if isinstance(p, dict) else getattr(p, "id", None)
                if pid is not None:
                    ids.append(pid)
                vec = (
                    p.get("vector")
                    if isinstance(p, dict)
                    else getattr(p, "vector", None)
                )
                if dim is None and isinstance(vec, (list, tuple)):
                    dim = _vector_dim(vec)
        except TypeError:
            pass
    return {
        "target": _target(args, kwargs),
        "mutation": {
            "vector_count": count,
            "vector_dim": dim,
            "ids": _truncate_ids(ids) if ids else None,
        },
    }


def _summarize_retrieve(args, kwargs, instance, response=None) -> dict:
    ids = kwargs.get("ids")
    out = {
        "target": _target(args, kwargs),
        "query": {
            "ids": _truncate_ids(ids),
            "with_payload": kwargs.get("with_payload"),
            "with_vectors": kwargs.get("with_vectors"),
        },
    }
    if isinstance(response, list):
        out["result"] = {"count": len(response)}
    return out


def _summarize_delete(args, kwargs, instance, response=None) -> dict:
    sel = kwargs.get("points_selector")
    return {
        "target": _target(args, kwargs),
        "mutation": {"points_selector": _safe_jsonable(sel)},
    }


def _summarize_scroll(args, kwargs, instance, response=None) -> dict:
    out = {
        "target": _target(args, kwargs),
        "query": {
            "limit": kwargs.get("limit"),
            "filter": _safe_jsonable(kwargs.get("scroll_filter")),
            "offset": kwargs.get("offset"),
        },
    }
    if isinstance(response, tuple) and response:
        pts = response[0] if isinstance(response[0], list) else []
        out["result"] = {"count": len(pts)}
    return out


def _search_cache_key(args, kwargs, instance) -> str:
    coll = kwargs.get("collection_name") or (args[0] if args else "") or ""
    vec = kwargs.get("query_vector")
    return vectorstore_cache_key(
        "qdrant", coll, vec, kwargs.get("limit"), kwargs.get("query_filter"),
    )


def _query_points_cache_key(args, kwargs, instance) -> str:
    coll = kwargs.get("collection_name") or (args[0] if args else "") or ""
    return vectorstore_cache_key(
        "qdrant", coll, kwargs.get("query"), kwargs.get("limit"), kwargs.get("query_filter"),
    )


def _make_point_list(cached_result: dict) -> list:
    items = (cached_result.get("matches") or {}).get("items") or []
    return [
        SimpleNamespace(
            id=m.get("id"),
            score=m.get("score"),
            payload=m.get("metadata"),
            vector=None,
        )
        for m in items
    ]


def _points_mock(cached_result: dict, args, kwargs, instance) -> list:
    # search() returns a bare list of ScoredPoint
    return _make_point_list(cached_result)


def _query_points_mock(cached_result: dict, args, kwargs, instance):
    # query_points() returns a QueryResponse with a .points attribute
    return SimpleNamespace(points=_make_point_list(cached_result))


_CACHED_OPS = (
    ("search", "search", _summarize_search, _search_cache_key, _points_mock),
    ("query_points", "query_points", _summarize_query_points, _query_points_cache_key, _query_points_mock),
)
_INVALIDATING_OPS = (
    ("upsert", "upsert", _summarize_upsert),
    ("delete", "delete", _summarize_delete),
)
_PLAIN_OPS = (
    ("retrieve", "retrieve", _summarize_retrieve),
    ("scroll", "scroll", _summarize_scroll),
)


def _patch_sync(cls):
    for attr, api, summarize, key_fn, mock_fn in _CACHED_OPS:
        if hasattr(cls, attr):
            setattr(
                cls, attr,
                make_sync_cached_wrapper(getattr(cls, attr), TraceType.Qdrant, api, summarize, key_fn, mock_fn),
            )
    for attr, api, summarize in _INVALIDATING_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_sync_invalidating_wrapper(
                getattr(cls, attr), TraceType.Qdrant, api, summarize, _qdrant_target,
            ))
    for attr, api, summarize in _PLAIN_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_sync_wrapper(getattr(cls, attr), TraceType.Qdrant, api, summarize))


def _patch_async(cls):
    for attr, api, summarize, key_fn, mock_fn in _CACHED_OPS:
        if hasattr(cls, attr):
            setattr(
                cls, attr,
                make_async_cached_wrapper(getattr(cls, attr), TraceType.Qdrant, api, summarize, key_fn, mock_fn),
            )
    for attr, api, summarize in _INVALIDATING_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_async_invalidating_wrapper(
                getattr(cls, attr), TraceType.Qdrant, api, summarize, _qdrant_target,
            ))
    for attr, api, summarize in _PLAIN_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, make_async_wrapper(getattr(cls, attr), TraceType.Qdrant, api, summarize))


def patch_qdrant():
    try:
        from qdrant_client import QdrantClient
    except Exception:
        return
    _patch_sync(QdrantClient)


def patch_qdrant_async():
    try:
        from qdrant_client import AsyncQdrantClient
    except Exception:
        return
    _patch_async(AsyncQdrantClient)
