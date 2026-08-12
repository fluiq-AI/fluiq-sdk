import contextvars

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
        except Exception:
            pass
        return result

    wrapped._fluiq_patched = True
    ClientSession.initialize = wrapped


def patch_mcp_list_tools():
    """Trace ClientSession.list_tools() calls."""
    try:
        from mcp import ClientSession
    except ImportError:
        return

    if getattr(ClientSession.list_tools, "_fluiq_patched", False):
        return

    original = ClientSession.list_tools

    async def wrapped(self, *args, **kwargs):
        from fluiq.tracer import log_trace

        server_url = _get_session_url(self)
        result = await original(self, *args, **kwargs)

        if server_url:
            try:
                log_trace({
                    "type": "mcp",
                    "kind": "mcp_list_tools",
                    "server_url": server_url,
                })
            except Exception:
                pass

        return result

    wrapped._fluiq_patched = True
    ClientSession.list_tools = wrapped


def patch_mcp_call_tool():
    """Trace ClientSession.call_tool() invocations."""
    try:
        from mcp import ClientSession
    except ImportError:
        return

    if getattr(ClientSession.call_tool, "_fluiq_patched", False):
        return

    original = ClientSession.call_tool

    async def wrapped(self, name, arguments=None, *args, **kwargs):
        from fluiq.tracer import log_trace

        server_url = _get_session_url(self)
        result = await original(self, name, arguments, *args, **kwargs)

        if server_url:
            try:
                log_trace({
                    "type": "mcp",
                    "kind": "mcp_call",
                    "server_url": server_url,
                    "tool_name": name,
                })
            except Exception:
                pass

        return result

    wrapped._fluiq_patched = True
    ClientSession.call_tool = wrapped
