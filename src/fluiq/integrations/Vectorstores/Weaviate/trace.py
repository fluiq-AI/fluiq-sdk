from typing import Any

from fluiq.integrations.shared.models import TraceType
from fluiq.integrations.Vectorstores.shared.utils import (
    _build_match,
    _capture_matches,
    _capture_query_texts,
    _safe_jsonable,
    _truncate_ids,
    _vector_count_and_dim,
    _vector_dim,
    make_async_wrapper,
    make_sync_wrapper,
)


def _properties_text(props: Any) -> Any:
    if not isinstance(props, dict):
        return None
    for key in ("text", "content", "page_content", "chunk", "body"):
        val = props.get(key)
        if isinstance(val, str):
            return val
    return None


def _objects_to_matches(response) -> tuple:
    if response is None:
        return [], []
    objs = (
        response.get("objects")
        if isinstance(response, dict)
        else getattr(response, "objects", None)
    )
    if objs is None:
        return [], []
    items: list = []
    scores: list = []
    for o in objs:
        is_dict = isinstance(o, dict)
        oid = (
            o.get("uuid") if is_dict
            else (getattr(o, "uuid", None) or getattr(o, "id", None))
        )
        meta = o.get("metadata") if is_dict else getattr(o, "metadata", None)
        sc = None
        if meta is not None:
            sc = (
                meta.get("score") if isinstance(meta, dict)
                else (getattr(meta, "score", None) or getattr(meta, "distance", None))
            )
        if isinstance(sc, (int, float)):
            scores.append(sc)
        props = o.get("properties") if is_dict else getattr(o, "properties", None)
        items.append(_build_match(
            id=oid, score=sc, text=_properties_text(props), metadata=props,
        ))
    return items, scores


def _target(instance, kwargs) -> dict:
    name = (
        getattr(instance, "name", None)
        or getattr(getattr(instance, "_collection", None), "name", None)
        or getattr(getattr(instance, "_executor", None), "name", None)
    )
    return {"collection": name, "tenant": kwargs.get("tenant")}


def _objects_count(response) -> Any:
    if response is None:
        return None
    objs = (
        response.get("objects")
        if isinstance(response, dict)
        else getattr(response, "objects", None)
    )
    if objs is None:
        return None
    try:
        return len(objs)
    except TypeError:
        return None


def _objects_ids(response) -> list:
    if response is None:
        return []
    objs = (
        response.get("objects")
        if isinstance(response, dict)
        else getattr(response, "objects", None)
    )
    if objs is None:
        return []
    out: list = []
    for o in objs:
        oid = (
            o.get("uuid")
            if isinstance(o, dict)
            else (getattr(o, "uuid", None) or getattr(o, "id", None))
        )
        if oid is not None:
            out.append(str(oid))
    return out


def _attach_matches(out: dict, response) -> None:
    items, scores = _objects_to_matches(response)
    if items:
        out["result"] = {
            "matches": _capture_matches(items),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }


def _summarize_near_vector(args, kwargs, instance, response=None) -> dict:
    vec = kwargs.get("near_vector") or (args[0] if args else None)
    out = {
        "target": _target(instance, kwargs),
        "query": {
            "top_k": kwargs.get("limit"),
            "vector_dim": _vector_dim(vec),
            "filter": _safe_jsonable(kwargs.get("filters")),
            "distance": kwargs.get("distance"),
            "certainty": kwargs.get("certainty"),
        },
    }
    _attach_matches(out, response)
    return out


def _summarize_near_text(args, kwargs, instance, response=None) -> dict:
    q = kwargs.get("query") or (args[0] if args else None)
    out = {
        "target": _target(instance, kwargs),
        "query": {
            "top_k": kwargs.get("limit"),
            "texts": _capture_query_texts(q),
            "filter": _safe_jsonable(kwargs.get("filters")),
            "distance": kwargs.get("distance"),
            "certainty": kwargs.get("certainty"),
        },
    }
    _attach_matches(out, response)
    return out


def _summarize_hybrid(args, kwargs, instance, response=None) -> dict:
    q = kwargs.get("query") or (args[0] if args else None)
    out = {
        "target": _target(instance, kwargs),
        "query": {
            "top_k": kwargs.get("limit"),
            "alpha": kwargs.get("alpha"),
            "texts": _capture_query_texts(q),
            "vector_dim": _vector_dim(kwargs.get("vector")),
            "filter": _safe_jsonable(kwargs.get("filters")),
        },
    }
    _attach_matches(out, response)
    return out


def _summarize_bm25(args, kwargs, instance, response=None) -> dict:
    q = kwargs.get("query") or (args[0] if args else None)
    out = {
        "target": _target(instance, kwargs),
        "query": {
            "top_k": kwargs.get("limit"),
            "texts": _capture_query_texts(q),
            "filter": _safe_jsonable(kwargs.get("filters")),
        },
    }
    _attach_matches(out, response)
    return out


def _summarize_fetch_objects(args, kwargs, instance, response=None) -> dict:
    out = {
        "target": _target(instance, kwargs),
        "query": {
            "limit": kwargs.get("limit"),
            "offset": kwargs.get("offset"),
            "filter": _safe_jsonable(kwargs.get("filters")),
        },
    }
    _attach_matches(out, response)
    return out


def _summarize_insert(args, kwargs, instance, response=None) -> dict:
    vec = kwargs.get("vector")
    out = {
        "target": _target(instance, kwargs),
        "mutation": {
            "vector_dim": _vector_dim(vec),
            "has_properties": kwargs.get("properties") is not None,
        },
    }
    if response is not None:
        out["mutation"]["id"] = str(response)
    return out


def _summarize_insert_many(args, kwargs, instance, response=None) -> dict:
    objects = kwargs.get("objects") or (args[0] if args else None)
    count = None
    dim = None
    if objects is not None:
        try:
            seq = list(objects)
            count = len(seq)
            for o in seq:
                vec = (
                    o.get("vector")
                    if isinstance(o, dict)
                    else getattr(o, "vector", None)
                )
                if dim is None and isinstance(vec, (list, tuple)):
                    dim = _vector_dim(vec)
                    break
        except TypeError:
            pass
    out = {
        "target": _target(instance, kwargs),
        "mutation": {"vector_count": count, "vector_dim": dim},
    }
    if response is not None:
        has_errors = (
            getattr(response, "has_errors", None)
            if not isinstance(response, dict)
            else response.get("has_errors")
        )
        if has_errors is not None:
            out["mutation"]["has_errors"] = bool(has_errors)
    return out


def _summarize_delete_by_id(args, kwargs, instance, response=None) -> dict:
    uuid = kwargs.get("uuid") or (args[0] if args else None)
    return {
        "target": _target(instance, kwargs),
        "mutation": {"ids": _truncate_ids([uuid] if uuid else None)},
    }


def _summarize_delete_many(args, kwargs, instance, response=None) -> dict:
    return {
        "target": _target(instance, kwargs),
        "mutation": {
            "where": _safe_jsonable(kwargs.get("where") or (args[0] if args else None)),
        },
    }


def _summarize_update(args, kwargs, instance, response=None) -> dict:
    return {
        "target": _target(instance, kwargs),
        "mutation": {
            "ids": _truncate_ids([kwargs.get("uuid")] if kwargs.get("uuid") else None),
            "has_properties": kwargs.get("properties") is not None,
            "vector_dim": _vector_dim(kwargs.get("vector")),
        },
    }



_QUERY_OPS = (
    ("near_vector", "near_vector", _summarize_near_vector),
    ("near_text", "near_text", _summarize_near_text),
    ("hybrid", "hybrid", _summarize_hybrid),
    ("bm25", "bm25", _summarize_bm25),
    ("fetch_objects", "fetch_objects", _summarize_fetch_objects),
)

_DATA_OPS = (
    ("insert", "insert", _summarize_insert),
    ("insert_many", "insert_many", _summarize_insert_many),
    ("delete_by_id", "delete_by_id", _summarize_delete_by_id),
    ("delete_many", "delete_many", _summarize_delete_many),
    ("update", "update", _summarize_update),
    ("replace", "replace", _summarize_update),
)


def _import_first(candidates):
    for path, attr in candidates:
        try:
            mod = __import__(path, fromlist=[attr])
            cls = getattr(mod, attr, None)
            if cls is not None:
                return cls
        except Exception:
            continue
    return None


def _patch_ops(cls, ops, wrapper_factory):
    for attr, api, summarize in ops:
        if hasattr(cls, attr):
            setattr(
                cls, attr,
                wrapper_factory(getattr(cls, attr), TraceType.Weaviate, api, summarize),
            )


_QUERY_SYNC_PATHS = (
    ("weaviate.collections.queries.query", "_QueryCollection"),
    ("weaviate.collections.queries.query", "_Query"),
)
_QUERY_ASYNC_PATHS = (
    ("weaviate.collections.queries.query", "_QueryCollectionAsync"),
    ("weaviate.collections.queries.query", "_QueryAsync"),
)
_DATA_SYNC_PATHS = (
    ("weaviate.collections.data", "_DataCollection"),
    ("weaviate.collections.data.data", "_Data"),
)
_DATA_ASYNC_PATHS = (
    ("weaviate.collections.data", "_DataCollectionAsync"),
    ("weaviate.collections.data.data", "_DataAsync"),
)


def patch_weaviate():
    qcls = _import_first(_QUERY_SYNC_PATHS)
    if qcls is not None:
        _patch_ops(qcls, _QUERY_OPS, make_sync_wrapper)
    dcls = _import_first(_DATA_SYNC_PATHS)
    if dcls is not None:
        _patch_ops(dcls, _DATA_OPS, make_sync_wrapper)


def patch_weaviate_async():
    qcls = _import_first(_QUERY_ASYNC_PATHS)
    if qcls is not None:
        _patch_ops(qcls, _QUERY_OPS, make_async_wrapper)
    dcls = _import_first(_DATA_ASYNC_PATHS)
    if dcls is not None:
        _patch_ops(dcls, _DATA_OPS, make_async_wrapper)
