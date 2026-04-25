def _session_descriptor(obj):
    cls = type(obj)
    return {
        "type": "mcp_session",
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "repr": repr(obj)[:200],
    }


def _is_mcp_session(obj):
    cls = type(obj)
    mod = (cls.__module__ or "")
    return "mcp" in mod.lower() and "session" in cls.__qualname__.lower()


def _extract_mcp_servers(kwargs):
    config = kwargs.get("config")
    tools = kwargs.get("tools")
    if config is not None and not tools:
        tools = (
            config.get("tools") if isinstance(config, dict)
            else getattr(config, "tools", None)
        )

    found = []
    for tool in tools or []:
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