def _dump(v):
    if v is None:
        return None
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return v.model_dump(exclude_none=True)
    return v


def patch_mcp_initialize():
    try:
        from mcp import ClientSession
    except ImportError:
        return

    if getattr(ClientSession.initialize, "_fluiq_patched", False):
        return

    original = ClientSession.initialize

    async def wrapped(self, *args, **kwargs):
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
        except Exception:
            pass
        return result

    wrapped._fluiq_patched = True
    ClientSession.initialize = wrapped
