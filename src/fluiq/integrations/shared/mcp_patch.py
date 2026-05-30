import contextvars
from types import SimpleNamespace

# Set by our streamablehttp_client / http_client wrappers so that
# _get_session_url can find the URL even when the session streams
# carry no URL attribute.
_mcp_server_url_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_fluiq_mcp_server_url", default=""
)


def _dump(v):
    if v is None:
        return None
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return v.model_dump(exclude_none=True)
    return v


def _get_session_url(session) -> str:
    """Return the server URL stored on this session (set during initialize), or ''."""
    url = getattr(session, "_fluiq_server_url", None)
    if isinstance(url, str) and url:
        return url
    # ContextVar set by our streamablehttp_client / http_client wrapper.
    cv_url = _mcp_server_url_var.get("")
    if cv_url:
        return cv_url
    # Fallback: try to extract from transport streams (pre-initialize path,
    # or transports that expose the URL as an attribute).
    for attr in ("_read_stream", "_write_stream"):
        stream = getattr(session, attr, None)
        if stream is None:
            continue
        for candidate in ("_fluiq_url", "url", "_url", "endpoint", "_endpoint"):
            v = getattr(stream, candidate, None)
            if isinstance(v, str) and v:
                return v
    return ""


def _patch_mcp_transports() -> None:
    """Wrap streamablehttp_client (and http_client) to tag the server URL onto
    the read stream and into a ContextVar so _get_session_url can find it."""

    def _wrap(module_path: str, fn_name: str) -> None:
        try:
            import importlib
            mod = importlib.import_module(module_path)
        except ImportError:
            return
        orig = getattr(mod, fn_name, None)
        if orig is None or getattr(orig, "_fluiq_patched", False):
            return

        import contextlib

        @contextlib.asynccontextmanager
        async def _wrapped(url, *args, **kwargs):
            url_str = str(url)
            token = _mcp_server_url_var.set(url_str)
            try:
                async with orig(url, *args, **kwargs) as streams:
                    # streams is (read, write) or (read, write, get_session_id)
                    read = streams[0]
                    try:
                        read._fluiq_url = url_str
                    except Exception:
                        pass
                    yield streams
            finally:
                _mcp_server_url_var.reset(token)

        _wrapped._fluiq_patched = True
        setattr(mod, fn_name, _wrapped)

    _wrap("mcp.client.streamable_http", "streamablehttp_client")
    _wrap("mcp.client.streamable_http", "streamable_http_client")
    _wrap("mcp.client.sse", "sse_client")


def patch_mcp_initialize():
    try:
        from mcp import ClientSession
    except ImportError:
        return

    _patch_mcp_transports()

    if getattr(ClientSession.initialize, "_fluiq_patched", False):
        return

    original = ClientSession.initialize

    async def wrapped(self, *args, **kwargs):
        # Extract URL from transport streams before the call so it's available
        # even if initialize raises.
        server_url = _get_session_url(self)
        result = await original(self, *args, **kwargs)
        try:
            server_info = getattr(result, "serverInfo", None) or getattr(result, "server_info", None)
            if server_info is not None:
                self._fluiq_server_info = _dump(server_info)
            protocol_version = getattr(result, "protocolVersion", None) or getattr(result, "protocol_version", None)
            if protocol_version is not None:
                self._fluiq_protocol_version = protocol_version
            instructions = getattr(result, "instructions", None)
            if instructions is not None:
                self._fluiq_instructions = instructions
            # Store URL so list_tools/call_tool patches can find it without
            # re-extracting from streams every call.
            if server_url:
                self._fluiq_server_url = server_url
                # Server (re-)initialize means the tool list may have changed.
                try:
                    from fluiq.optimization.client import invalidate_mcp_tools_cache
                    invalidate_mcp_tools_cache(server_url)
                except Exception:
                    pass
        except Exception:
            pass
        return result

    wrapped._fluiq_patched = True
    ClientSession.initialize = wrapped


def patch_mcp_list_tools():
    """Cache ClientSession.list_tools() responses in Redis, keyed by server URL."""
    try:
        from mcp import ClientSession
    except ImportError:
        return

    if getattr(ClientSession.list_tools, "_fluiq_patched", False):
        return

    original = ClientSession.list_tools

    async def wrapped(self, *args, **kwargs):
        from fluiq.optimization.client import (
            lookup_mcp_tools_cache,
            populate_mcp_tools_cache,
        )
        from fluiq.tracer import log_trace

        server_url = _get_session_url(self)

        if server_url:
            cached_tools = lookup_mcp_tools_cache(server_url)
            if cached_tools is not None:
                try:
                    from mcp.types import ListToolsResult, Tool
                    result = ListToolsResult(
                        tools=[Tool.model_validate(t) for t in cached_tools]
                    )
                except Exception:
                    result = SimpleNamespace(
                        tools=[SimpleNamespace(**t) for t in cached_tools]
                    )
                try:
                    log_trace({
                        "type": "mcp",
                        "kind": "mcp_list_tools",
                        "server_url": server_url,
                        "cache_hit": True,
                    })
                except Exception:
                    pass
                return result

        result = await original(self, *args, **kwargs)

        if server_url:
            try:
                tools = []
                for t in (getattr(result, "tools", None) or []):
                    if hasattr(t, "model_dump"):
                        tools.append(t.model_dump(mode="json", exclude_none=True))
                    else:
                        tools.append({
                            "name": getattr(t, "name", None),
                            "description": getattr(t, "description", None),
                            "inputSchema": _dump(getattr(t, "inputSchema", None)),
                        })
                populate_mcp_tools_cache(server_url, tools)
            except Exception:
                pass
            try:
                log_trace({
                    "type": "mcp",
                    "kind": "mcp_list_tools",
                    "server_url": server_url,
                    "cache_hit": False,
                })
            except Exception:
                pass

        return result

    wrapped._fluiq_patched = True
    ClientSession.list_tools = wrapped


def patch_mcp_call_tool():
    """Cache ClientSession.call_tool() results in Redis, keyed by (server_url, tool_name, args)."""
    try:
        from mcp import ClientSession
    except ImportError:
        return

    if getattr(ClientSession.call_tool, "_fluiq_patched", False):
        return

    original = ClientSession.call_tool

    async def wrapped(self, name, arguments=None, *args, **kwargs):
        from fluiq.optimization.client import (
            lookup_mcp_call_cache,
            populate_mcp_call_cache,
        )
        from fluiq.tracer import log_trace

        server_url = _get_session_url(self)
        args_dict = arguments if isinstance(arguments, dict) else {}

        if server_url:
            cached = lookup_mcp_call_cache(server_url, name, args_dict)
            if cached is not None:
                cached_content = cached.get("content") or []
                cached_is_error = cached.get("isError", False)
                try:
                    from mcp.types import CallToolResult
                    result = CallToolResult.model_validate({
                        "content": cached_content,
                        "isError": cached_is_error,
                    })
                except Exception:
                    result = SimpleNamespace(
                        content=[SimpleNamespace(**c) for c in cached_content],
                        isError=cached_is_error,
                    )
                try:
                    log_trace({
                        "type": "mcp",
                        "kind": "mcp_call",
                        "server_url": server_url,
                        "tool_name": name,
                        "cache_hit": True,
                    })
                except Exception:
                    pass
                return result

        result = await original(self, name, arguments, *args, **kwargs)

        if server_url and not getattr(result, "isError", False):
            try:
                raw_content = getattr(result, "content", None) or []
                serialized = []
                for item in raw_content:
                    if hasattr(item, "model_dump"):
                        serialized.append(item.model_dump(mode="json", exclude_none=True))
                    elif isinstance(item, dict):
                        serialized.append(item)
                    else:
                        serialized.append({"type": "text", "text": str(item)})
                populate_mcp_call_cache(server_url, name, args_dict, serialized, is_error=False)
            except Exception:
                pass
            try:
                log_trace({
                    "type": "mcp",
                    "kind": "mcp_call",
                    "server_url": server_url,
                    "tool_name": name,
                    "cache_hit": False,
                })
            except Exception:
                pass

        return result

    wrapped._fluiq_patched = True
    ClientSession.call_tool = wrapped
