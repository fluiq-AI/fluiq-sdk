from typing import Any

from types import SimpleNamespace

from fluiq.integrations.shared.models import TraceType
from fluiq.integrations.Vectorstores.shared.utils import (
    _build_match,
    _capture_matches,
    _capture_query_texts,
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



def _collection_name(instance) -> str:
    return (
        getattr(instance, "name", None)
        or getattr(instance, "_name", None)  # _BaseExecutor stores name as _name
        or getattr(getattr(instance, "_collection", None), "name", None)
        or getattr(getattr(instance, "_executor", None), "name", None)
        or ""
    )


def _near_vector_key(args, kwargs, instance) -> str:
    vec = kwargs.get("near_vector") or (args[0] if args else None)
    target = kwargs.get("target_vector")
    return vectorstore_cache_key("weaviate", _collection_name(instance), vec, kwargs.get("limit"), (kwargs.get("filters"), target))


def _near_text_key(args, kwargs, instance) -> str:
    q = kwargs.get("query") or (args[0] if args else None)
    return vectorstore_cache_key("weaviate", _collection_name(instance), q, kwargs.get("limit"), kwargs.get("filters"))


def _hybrid_key(args, kwargs, instance) -> str:
    q = kwargs.get("query") or (args[0] if args else None)
    return vectorstore_cache_key("weaviate", _collection_name(instance), q, kwargs.get("limit"), kwargs.get("filters"))


def _bm25_key(args, kwargs, instance) -> str:
    q = kwargs.get("query") or (args[0] if args else None)
    return vectorstore_cache_key("weaviate", _collection_name(instance), q, kwargs.get("limit"), kwargs.get("filters"))


def _fetch_objects_key(args, kwargs, instance) -> str:
    return vectorstore_cache_key("weaviate", _collection_name(instance), None, kwargs.get("limit"), kwargs.get("filters"))


def _objects_mock(cached_result: dict, args, kwargs, instance):
    items = (cached_result.get("matches") or {}).get("items") or []
    objects = [
        SimpleNamespace(
            uuid=m.get("id"),
            metadata=SimpleNamespace(score=m.get("score"), distance=None),
            properties=m.get("metadata") or {},
            vector=None,
        )
        for m in items
    ]
    return SimpleNamespace(objects=objects)


_QUERY_OPS = (
    ("near_vector", "near_vector", _summarize_near_vector, _near_vector_key),
    ("near_text", "near_text", _summarize_near_text, _near_text_key),
    ("hybrid", "hybrid", _summarize_hybrid, _hybrid_key),
    ("bm25", "bm25", _summarize_bm25, _bm25_key),
    ("fetch_objects", "fetch_objects", _summarize_fetch_objects, _fetch_objects_key),
)

_DATA_OPS = (
    ("insert", "insert", _summarize_insert),
    ("insert_many", "insert_many", _summarize_insert_many),
    ("delete_by_id", "delete_by_id", _summarize_delete_by_id),
    ("delete_many", "delete_many", _summarize_delete_many),
    ("update", "update", _summarize_update),
    ("replace", "replace", _summarize_update),
)


def _import_cls(candidates):
    """Return the first importable class from a list of (module_path, class_name) pairs."""
    for path, attr in candidates:
        try:
            mod = __import__(path, fromlist=[attr])
            cls = getattr(mod, attr, None)
            if cls is not None:
                return cls
        except Exception:
            continue
    return None


# Weaviate v4 restructured each query op into its own package.
# Each op has a shared executor (holds the original method) plus separate
# sync and async concrete subclasses.  We patch the concrete subclasses so
# sync and async wrappers never overwrite each other on the shared executor.
#
# Tuple: (attr, api, summarize_fn, key_fn, base_pkg,
#         executor_cls, sync_cls, async_cls)
_QUERY_EXECUTOR_OPS = (
    ("near_vector", "near_vector", _summarize_near_vector, _near_vector_key,
     "weaviate.collections.queries.near_vector.query",
     "_NearVectorQueryExecutor", "_NearVectorQuery", "_NearVectorQueryAsync"),
    ("near_text",   "near_text",   _summarize_near_text,   _near_text_key,
     "weaviate.collections.queries.near_text.query",
     "_NearTextQueryExecutor",   "_NearTextQuery",   "_NearTextQueryAsync"),
    ("hybrid",      "hybrid",      _summarize_hybrid,      _hybrid_key,
     "weaviate.collections.queries.hybrid.query",
     "_HybridQueryExecutor",     "_HybridQuery",     "_HybridQueryAsync"),
    ("bm25",        "bm25",        _summarize_bm25,        _bm25_key,
     "weaviate.collections.queries.bm25.query",
     "_BM25QueryExecutor",       "_BM25Query",       "_BM25QueryAsync"),
    ("fetch_objects", "fetch_objects", _summarize_fetch_objects, _fetch_objects_key,
     "weaviate.collections.queries.fetch_objects.query",
     "_FetchObjectsQueryExecutor", "_FetchObjectsQuery", "_FetchObjectsQueryAsync"),
)

# Legacy single-class paths (older Weaviate v4 client builds).
_QUERY_LEGACY_SYNC_PATHS = (
    ("weaviate.collections.queries.query", "_QueryCollection"),
    ("weaviate.collections.queries.query", "_Query"),
)
_QUERY_LEGACY_ASYNC_PATHS = (
    ("weaviate.collections.queries.query", "_QueryCollectionAsync"),
    ("weaviate.collections.queries.query", "_QueryAsync"),
)

_DATA_PATHS = (
    ("weaviate.collections.data.executor", "_DataCollectionExecutor"),
    ("weaviate.collections.data", "_DataCollection"),
    ("weaviate.collections.data.data", "_Data"),
)
_DATA_ASYNC_PATHS = (
    ("weaviate.collections.data", "_DataCollectionAsync"),
    ("weaviate.collections.data.data", "_DataAsync"),
)


def _weaviate_target_fn(args, kwargs, instance) -> str:
    return _collection_name(instance) or ""


def _patch_data_ops(cls, wrapper_factory):
    inv_factory = (
        make_sync_invalidating_wrapper
        if wrapper_factory is make_sync_wrapper
        else make_async_invalidating_wrapper
    )
    for attr, api, summarize in _DATA_OPS:
        if hasattr(cls, attr):
            setattr(cls, attr, inv_factory(
                getattr(cls, attr), TraceType.Weaviate, api, summarize, _weaviate_target_fn,
            ))


def _patch_query_ops_per_class(sync: bool) -> bool:
    """Patch each query op on its dedicated concrete class (sync or async).

    Returns True if at least one op was patched successfully.
    Getting the original from the executor's own __dict__ ensures we never
    accidentally wrap an already-wrapped method.
    """
    cached_fn = make_sync_cached_wrapper if sync else make_async_cached_wrapper
    sub_mod = "sync" if sync else "async_"
    patched_any = False
    for attr, api, summarize, key_fn, base_pkg, exec_cls_name, sync_cls_name, async_cls_name in _QUERY_EXECUTOR_OPS:
        target_cls_name = sync_cls_name if sync else async_cls_name
        try:
            exec_mod = __import__(f"{base_pkg}.executor", fromlist=[exec_cls_name])
            exec_cls = getattr(exec_mod, exec_cls_name, None)
            if exec_cls is None or attr not in vars(exec_cls):
                continue
            original = vars(exec_cls)[attr]

            target_mod = __import__(f"{base_pkg}.{sub_mod}", fromlist=[target_cls_name])
            target_cls = getattr(target_mod, target_cls_name, None)
            if target_cls is None:
                continue
            setattr(target_cls, attr, cached_fn(
                original, TraceType.Weaviate, api, summarize, key_fn, _objects_mock,
            ))
            patched_any = True
        except Exception:
            continue
    return patched_any


def patch_weaviate():
    patched = _patch_query_ops_per_class(sync=True)
    if not patched:
        qcls = _import_cls(_QUERY_LEGACY_SYNC_PATHS)
        if qcls is not None:
            for attr, api, summarize, key_fn, *_ in _QUERY_EXECUTOR_OPS:
                if hasattr(qcls, attr):
                    setattr(qcls, attr, make_sync_cached_wrapper(
                        getattr(qcls, attr), TraceType.Weaviate, api, summarize, key_fn, _objects_mock,
                    ))
    dcls = _import_cls(_DATA_PATHS)
    if dcls is not None:
        _patch_data_ops(dcls, make_sync_wrapper)


def patch_weaviate_async():
    patched = _patch_query_ops_per_class(sync=False)
    if not patched:
        qcls = _import_cls(_QUERY_LEGACY_ASYNC_PATHS)
        if qcls is not None:
            for attr, api, summarize, key_fn, *_ in _QUERY_EXECUTOR_OPS:
                if hasattr(qcls, attr):
                    setattr(qcls, attr, make_async_cached_wrapper(
                        getattr(qcls, attr), TraceType.Weaviate, api, summarize, key_fn, _objects_mock,
                    ))
    dcls = _import_cls(_DATA_ASYNC_PATHS) or _import_cls(_DATA_PATHS)
    if dcls is not None:
        _patch_data_ops(dcls, make_async_wrapper)
