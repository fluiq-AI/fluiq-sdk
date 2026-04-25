from fluiq.integrations.Anthropic.helper.utils import _to_jsonable

MCP_BLOCK_TYPES = {"mcp_tool_use", "mcp_tool_result"}

def _extract_mcp_servers(kwargs):
    servers = kwargs.get("mcp_servers")
    return _to_jsonable(servers) if servers else None

def _extract_mcp_blocks(content):
    if not isinstance(content, list):
        return None
    
    blocks = []
    for block in content:
        btype = (
            block.get("type") if isinstance(block, dict)
            else getattr(block, "type", None)
        )
        if btype in MCP_BLOCK_TYPES:
            blocks.append(_to_jsonable(block))
    return blocks or None

def _extract_mcp_results_from_messages(messages):

    if not messages:
        return None
    results = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg,dict) else getattr(msg, "role", None)
        content = (
            msg.get("content") if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if role != "user" or not isinstance(content, list):
            continue
        
        for block in content:
            btype = (
                block.get("type") if isinstance(block, dict)
                else getattr(block, "type", None)
            )
            if btype == "mcp_tool_result":
                results.append(_to_jsonable(block))
    
    return results or None