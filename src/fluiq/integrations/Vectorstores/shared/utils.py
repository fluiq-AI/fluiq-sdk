import json
import time
from typing import Any, Callable, Optional

from fluiq.config import _config as _fluiq_config
from fluiq.integrations.shared.context import current_parent_id
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.tracer import log_trace


_MAX_IDS = 32
_MAX_STR = 256
_MAX_MATCHES = 10
_MAX_CHUNK_CHARS = 2000
_MAX_QUERY_CHARS = 4000
_MAX_METADATA_CHARS = 1000


def _truncate_text(value: Any, limit: int) -> Any:
    if value is None or not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _capture_query_texts(texts: Any) -> Optional[list]:
    if texts is None:
        return None
    if isinstance(texts, str):
        return [_truncate_text(texts, _MAX_QUERY_CHARS)]
    if isinstance(texts, (list, tuple)):
        return [_truncate_text(t, _MAX_QUERY_CHARS) for t in texts if t is not None]
    return [_truncate_text(str(texts), _MAX_QUERY_CHARS)]


def _capture_metadata(meta: Any) -> Any:
    if meta is None:
        return None
    return _safe_jsonable(meta, max_str=_MAX_METADATA_CHARS)


def _build_match(
    *,
    id: Any = None,
    score: Any = None,
    text: Any = None,
    metadata: Any = None,
) -> dict:
    out: dict = {}
    if id is not None:
        out["id"] = str(id)
    if isinstance(score, (int, float)):
        out["score"] = float(score)
    if text is not None:
        out["text"] = _truncate_text(text, _MAX_CHUNK_CHARS)
    if metadata is not None:
        out["metadata"] = _capture_metadata(metadata)
    return out


def _capture_matches(matches: list) -> Optional[dict]:
    if not matches:
        return None
    total = len(matches)
    head = matches[:_MAX_MATCHES]
    return {
        "items": head,
        "count": total,
        "truncated": total > _MAX_MATCHES,
    }


def _safe_jsonable(obj: Any, max_str: int = _MAX_STR) -> Any:
    if obj is None:
        return None
    try:
        s = json.dumps(obj, default=str)
    except Exception:
        try:
            s = json.dumps(repr(obj), default=str)
        except Exception:
            return {"type": type(obj).__name__}
    if len(s) > max_str:
        s = s[:max_str] + "...[truncated]"
    try:
        return json.loads(s)
    except Exception:
        return s


def _truncate_ids(ids: Any, max_items: int = _MAX_IDS) -> Optional[dict]:
    if ids is None:
        return None
    try:
        seq = list(ids)
    except TypeError:
        return None
    total = len(seq)
    return {
        "count": total,
        "sample": [str(x) for x in seq[:max_items]],
        "truncated": total > max_items,
    }


def _vector_dim(vec: Any) -> Optional[int]:
    if vec is None:
        return None
    try:
        return len(vec)
    except TypeError:
        return None


def _vector_count_and_dim(vectors: Any) -> dict:
    out: dict = {"count": None, "dim": None}
    if vectors is None:
        return out
    try:
        seq = list(vectors)
    except TypeError:
        return out
    out["count"] = len(seq)
    if seq:
        first = seq[0]
        if isinstance(first, (list, tuple)):
            out["dim"] = _vector_dim(first)
        else:
            out["dim"] = _vector_dim(seq)
            out["count"] = 1
    return out


def emit_vector_trace(
    integration: TraceType,
    api: str,
    *,
    target: Optional[dict] = None,
    query: Optional[dict] = None,
    result: Optional[dict] = None,
    mutation: Optional[dict] = None,
    start: float,
    end: float,
    success: bool = True,
    error: Optional[str] = None,
    cache_hit: Optional[bool] = None,
) -> None:
    try:
        extras: dict = {}
        if cache_hit is False:
            extras["cache_hit"] = False
        payload = LogTrace(
            type="vectorstore",
            integration=integration,
            api=api,
            latency=end - start,
            parent_id=current_parent_id(),
            success=success,
            target=target,
            query=query,
            result=result,
            mutation=mutation,
            error=error,
            **extras,
        )
        payload_dict = payload.model_dump(mode="json", exclude_none=True)
        if cache_hit is True:
            # Inject after model_dump — Pydantic v2 strips underscore-prefixed keys,
            # so passing _cache_hit via LogTrace kwargs would silently drop it.
            payload_dict["_cache_hit"] = True
        log_trace(payload_dict)
    except Exception:
        pass


def _is_vs_cache_active() -> bool:
    return (
        bool(_fluiq_config.get("optimize"))
        and _fluiq_config.get("optimize_mode", "cache") == "cache"
        and bool(_fluiq_config.get("api_key"))
    )


def make_sync_cached_wrapper(
    original: Callable,
    integration: TraceType,
    api: str,
    summarize: Callable,
    cache_key_fn: Callable,
    mock_builder: Callable,
    raw_result_fn: Optional[Callable] = None,
) -> Callable:
    """Like make_sync_wrapper but caches query results via fluiq.optimize().

    cache_key_fn(args, kwargs, instance) -> Optional[str]
    mock_builder(cached_result_dict, args, kwargs, instance) -> provider_response
    raw_result_fn(args, kwargs, instance, response) -> Optional[dict]  — override
        what gets stored; defaults to summary["result"] when None.
    """
    def wrapped(self, *args, **kwargs):
        cache_key: Optional[str] = None
        if _is_vs_cache_active():
            try:
                cache_key = cache_key_fn(args, kwargs, self)
            except Exception:
                cache_key = None

        if cache_key:
            try:
                from fluiq.optimization.client import lookup_vectorstore_cache
                cached = lookup_vectorstore_cache(cache_key)
                if cached is not None:
                    mock = mock_builder(cached.get("result", {}), args, kwargs, self)
                    ts = time.time()
                    summary = _safe_summarize(summarize, args, kwargs, self, mock) or {}
                    emit_vector_trace(integration, api, **summary, start=ts, end=ts, cache_hit=True)
                    from fluiq.integrations.shared.context import mark_inner_cache_hit
                    mark_inner_cache_hit()
                    return mock
            except Exception:
                pass

        start = time.time()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            end = time.time()
            summary = _safe_summarize(summarize, args, kwargs, self)
            emit_vector_trace(
                integration, api, **summary,
                start=start, end=end, success=False, error=type(exc).__name__,
            )
            raise
        end = time.time()
        summary = _safe_summarize(summarize, args, kwargs, self, response) or {}

        if cache_key:
            try:
                cache_result = (
                    raw_result_fn(args, kwargs, self, response)
                    if raw_result_fn is not None
                    else summary.get("result")
                )
                if cache_result:
                    from fluiq.optimization.client import populate_vectorstore_cache
                    populate_vectorstore_cache(cache_key, cache_result)
            except Exception:
                pass

        emit_vector_trace(
            integration, api, **summary,
            start=start, end=end,
            cache_hit=False if cache_key else None,
        )
        return response

    return wrapped


def make_async_cached_wrapper(
    original: Callable,
    integration: TraceType,
    api: str,
    summarize: Callable,
    cache_key_fn: Callable,
    mock_builder: Callable,
    raw_result_fn: Optional[Callable] = None,
) -> Callable:
    async def wrapped(self, *args, **kwargs):
        cache_key: Optional[str] = None
        if _is_vs_cache_active():
            try:
                cache_key = cache_key_fn(args, kwargs, self)
            except Exception:
                cache_key = None

        if cache_key:
            try:
                from fluiq.optimization.client import lookup_vectorstore_cache
                cached = lookup_vectorstore_cache(cache_key)
                if cached is not None:
                    mock = mock_builder(cached.get("result", {}), args, kwargs, self)
                    ts = time.time()
                    summary = _safe_summarize(summarize, args, kwargs, self, mock) or {}
                    emit_vector_trace(integration, api, **summary, start=ts, end=ts, cache_hit=True)
                    from fluiq.integrations.shared.context import mark_inner_cache_hit
                    mark_inner_cache_hit()
                    return mock
            except Exception:
                pass

        start = time.time()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            end = time.time()
            summary = _safe_summarize(summarize, args, kwargs, self)
            emit_vector_trace(
                integration, api, **summary,
                start=start, end=end, success=False, error=type(exc).__name__,
            )
            raise
        end = time.time()
        summary = _safe_summarize(summarize, args, kwargs, self, response) or {}

        if cache_key:
            try:
                cache_result = (
                    raw_result_fn(args, kwargs, self, response)
                    if raw_result_fn is not None
                    else summary.get("result")
                )
                if cache_result:
                    from fluiq.optimization.client import populate_vectorstore_cache
                    populate_vectorstore_cache(cache_key, cache_result)
            except Exception:
                pass

        emit_vector_trace(
            integration, api, **summary,
            start=start, end=end,
            cache_hit=False if cache_key else None,
        )
        return response

    return wrapped


def make_sync_invalidating_wrapper(
    original: Callable,
    integration: TraceType,
    api: str,
    summarize: Callable,
    target_fn: Callable,
) -> Callable:
    """Like make_sync_wrapper but invalidates the vectorstore query cache on success.

    target_fn(args, kwargs, instance) -> str  — returns the collection/index name.
    Called for mutation operations (add, upsert, update, delete) so that
    subsequent queries see fresh results.
    """
    def wrapped(self, *args, **kwargs):
        start = time.time()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            end = time.time()
            summary = _safe_summarize(summarize, args, kwargs, self)
            emit_vector_trace(
                integration, api, **summary,
                start=start, end=end, success=False, error=type(exc).__name__,
            )
            raise
        end = time.time()
        summary = _safe_summarize(summarize, args, kwargs, self, response) or {}
        emit_vector_trace(integration, api, **summary, start=start, end=end)

        if _is_vs_cache_active():
            try:
                target = target_fn(args, kwargs, self) or ""
                from fluiq.optimization.client import invalidate_vectorstore_cache
                invalidate_vectorstore_cache(integration.value if hasattr(integration, "value") else str(integration), target)
            except Exception:
                pass

        return response

    return wrapped


def make_async_invalidating_wrapper(
    original: Callable,
    integration: TraceType,
    api: str,
    summarize: Callable,
    target_fn: Callable,
) -> Callable:
    async def wrapped(self, *args, **kwargs):
        start = time.time()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            end = time.time()
            summary = _safe_summarize(summarize, args, kwargs, self)
            emit_vector_trace(
                integration, api, **summary,
                start=start, end=end, success=False, error=type(exc).__name__,
            )
            raise
        end = time.time()
        summary = _safe_summarize(summarize, args, kwargs, self, response) or {}
        emit_vector_trace(integration, api, **summary, start=start, end=end)

        if _is_vs_cache_active():
            try:
                target = target_fn(args, kwargs, self) or ""
                from fluiq.optimization.client import invalidate_vectorstore_cache
                invalidate_vectorstore_cache(integration.value if hasattr(integration, "value") else str(integration), target)
            except Exception:
                pass

        return response

    return wrapped


def make_sync_wrapper(
    original: Callable,
    integration: TraceType,
    api: str,
    summarize: Callable[[tuple, dict, Any], dict],
) -> Callable:
    def wrapped(self, *args, **kwargs):
        start = time.time()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            end = time.time()
            summary = _safe_summarize(summarize, args, kwargs, self)
            emit_vector_trace(
                integration, api, **summary,
                start=start, end=end, success=False, error=type(exc).__name__,
            )
            raise
        end = time.time()
        summary = _safe_summarize(summarize, args, kwargs, self, response)
        emit_vector_trace(integration, api, **summary, start=start, end=end)
        return response
    return wrapped


def make_async_wrapper(
    original: Callable,
    integration: TraceType,
    api: str,
    summarize: Callable[[tuple, dict, Any], dict],
) -> Callable:
    async def wrapped(self, *args, **kwargs):
        start = time.time()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            end = time.time()
            summary = _safe_summarize(summarize, args, kwargs, self)
            emit_vector_trace(
                integration, api, **summary,
                start=start, end=end, success=False, error=type(exc).__name__,
            )
            raise
        end = time.time()
        summary = _safe_summarize(summarize, args, kwargs, self, response)
        emit_vector_trace(integration, api, **summary, start=start, end=end)
        return response
    return wrapped


def _safe_summarize(fn, args, kwargs, instance, response=None) -> dict:
    try:
        return fn(args, kwargs, instance, response) or {}
    except Exception:
        return {}
