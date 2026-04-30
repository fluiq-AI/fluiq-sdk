from typing import Any, Optional

from fluiq.integrations.shared.models import TraceType
from fluiq.integrations.Vectorstores.shared.utils import (
    _truncate_ids,
    make_sync_wrapper,
)


def _shape(x: Any) -> Optional[tuple]:
    try:
        return tuple(x.shape)
    except AttributeError:
        return None


def _n_d(x: Any) -> tuple:
    s = _shape(x)
    if s is None:
        return (None, None)
    if len(s) == 1:
        return (1, s[0])
    if len(s) >= 2:
        return (s[0], s[1])
    return (None, None)


def _target(instance) -> dict:
    return {
        "index_type": type(instance).__name__,
        "dim": getattr(instance, "d", None),
        "ntotal": getattr(instance, "ntotal", None),
        "is_trained": getattr(instance, "is_trained", None),
    }


def _flatten_minmax(arr: Any) -> tuple:
    try:
        flat = arr.reshape(-1)
        if flat.size == 0:
            return (None, None)
        return (float(flat.min()), float(flat.max()))
    except Exception:
        return (None, None)


def _summarize_add(args, kwargs, instance, response=None) -> dict:
    x = kwargs.get("x") or (args[0] if args else None)
    n, d = _n_d(x)
    return {
        "target": _target(instance),
        "mutation": {"vector_count": n, "vector_dim": d},
    }


def _summarize_add_with_ids(args, kwargs, instance, response=None) -> dict:
    x = kwargs.get("x") or (args[0] if args else None)
    ids = kwargs.get("ids") or (args[1] if len(args) > 1 else None)
    n, d = _n_d(x)
    id_summary = None
    try:
        seq = list(ids) if ids is not None else None
        id_summary = _truncate_ids(seq)
    except TypeError:
        pass
    return {
        "target": _target(instance),
        "mutation": {
            "vector_count": n,
            "vector_dim": d,
            "ids": id_summary,
        },
    }


def _summarize_search(args, kwargs, instance, response=None) -> dict:
    x = kwargs.get("x") or (args[0] if args else None)
    k = kwargs.get("k") or (args[1] if len(args) > 1 else None)
    n, d = _n_d(x)
    out = {
        "target": _target(instance),
        "query": {
            "top_k": k,
            "vector_count": n,
            "vector_dim": d,
        },
    }
    if isinstance(response, tuple) and len(response) >= 2:
        D, I = response[0], response[1]
        d_min, d_max = _flatten_minmax(D)
        try:
            result_count = int(I.size)
        except Exception:
            result_count = None
        out["result"] = {
            "count": result_count,
            "distance_min": d_min,
            "distance_max": d_max,
        }
    return out


def _summarize_range_search(args, kwargs, instance, response=None) -> dict:
    x = kwargs.get("x") or (args[0] if args else None)
    radius = kwargs.get("radius") or (args[1] if len(args) > 1 else None)
    n, d = _n_d(x)
    out = {
        "target": _target(instance),
        "query": {
            "vector_count": n,
            "vector_dim": d,
            "radius": float(radius) if isinstance(radius, (int, float)) else None,
        },
    }
    if isinstance(response, tuple) and len(response) >= 3:
        lims, D, _I = response[0], response[1], response[2]
        try:
            total = int(lims[-1]) if lims is not None and len(lims) > 0 else None
        except Exception:
            total = None
        d_min, d_max = _flatten_minmax(D)
        out["result"] = {
            "count": total,
            "distance_min": d_min,
            "distance_max": d_max,
        }
    return out


def _summarize_remove_ids(args, kwargs, instance, response=None) -> dict:
    sel = kwargs.get("sel") or (args[0] if args else None)
    out = {
        "target": _target(instance),
        "mutation": {"selector": type(sel).__name__ if sel is not None else None},
    }
    if isinstance(response, int):
        out["mutation"]["removed_count"] = response
    return out


def _summarize_train(args, kwargs, instance, response=None) -> dict:
    x = kwargs.get("x") or (args[0] if args else None)
    n, d = _n_d(x)
    return {
        "target": _target(instance),
        "mutation": {"vector_count": n, "vector_dim": d},
    }


def _summarize_reset(args, kwargs, instance, response=None) -> dict:
    return {"target": _target(instance), "mutation": {}}


_OPS = (
    ("add", "add", _summarize_add),
    ("add_with_ids", "add_with_ids", _summarize_add_with_ids),
    ("search", "search", _summarize_search),
    ("range_search", "range_search", _summarize_range_search),
    ("remove_ids", "remove_ids", _summarize_remove_ids),
    ("train", "train", _summarize_train),
    ("reset", "reset", _summarize_reset),
)


def _patch_index_class(cls):
    for attr, api, summarize in _OPS:
        if hasattr(cls, attr):
            try:
                setattr(
                    cls, attr,
                    make_sync_wrapper(getattr(cls, attr), TraceType.FAISS, api, summarize),
                )
            except (AttributeError, TypeError):
                continue


def patch_faiss():
    try:
        import faiss
    except Exception:
        return
    for cls_name in ("Index", "IndexBinary"):
        cls = getattr(faiss, cls_name, None)
        if cls is not None:
            _patch_index_class(cls)
