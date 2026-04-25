def _dump(v):
    if v is None:
        return None
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return v.model_dump(exclude_none=True)
    return v


def _transport_url(obj):
    for attr in ("_read_stream", "_write_stream"):
        stream = getattr(obj, attr, None)
        for candidate in ("url", "_url", "endpoint", "_endpoint"):
            url = getattr(stream, candidate, None)
            if isinstance(url, str):
                return url
    return None


def _session_descriptor(obj):
    cls = type(obj)
    desc = {
        "type": "mcp_session",
        "class": f"{cls.__module__}.{cls.__qualname__}",
    }
    client_info = _dump(getattr(obj, "_client_info", None))
    server_caps = _dump(getattr(obj, "_server_capabilities", None))
    url = _transport_url(obj)
    tools_cache = getattr(obj, "_fluiq_tools_cache", None)
    server_info = getattr(obj, "_fluiq_server_info", None)
    if client_info:
        desc["client_info"] = client_info
    if server_info:
        desc["server_info"] = server_info
    if server_caps:
        desc["server_capabilities"] = server_caps
    if url:
        desc["url"] = url
    if tools_cache:
        desc["tools"] = tools_cache
    return desc


def _is_mcp_session(obj):
    cls = type(obj)
    mod = (cls.__module__ or "")
    return "mcp" in mod.lower() and "session" in cls.__qualname__.lower()


def _locate_tools(kwargs):
    config = kwargs.get("config")
    tools = kwargs.get("tools")
    if config is not None and not tools:
        tools = (
            config.get("tools") if isinstance(config, dict)
            else getattr(config, "tools", None)
        )
    return tools or []


async def _enrich_mcp_sessions(kwargs):
    for tool in _locate_tools(kwargs):
        if not _is_mcp_session(tool):
            continue
        if getattr(tool, "_fluiq_tools_cache", None):
            continue
        try:
            result = await tool.list_tools()
            tools = []
            for t in (getattr(result, "tools", None) or []):
                tools.append({
                    "name": getattr(t, "name", None),
                    "description": getattr(t, "description", None),
                    "input_schema": _dump(getattr(t, "inputSchema", None)),
                })
            tool._fluiq_tools_cache = tools
        except Exception:
            tool._fluiq_tools_cache = []


def _extract_mcp_servers(kwargs):
    found = []
    for tool in _locate_tools(kwargs):
        if _is_mcp_session(tool):
            found.append(_session_descriptor(tool))
            continue
        mcp = (
            tool.get("mcp_servers") if isinstance(tool, dict)
            else getattr(tool, "mcp_servers", None)
        )
        if mcp:
            for srv in mcp:
                found.append(
                    srv if isinstance(srv, dict)
                    else getattr(srv, "model_dump", lambda: srv)()
                )
    return found or None