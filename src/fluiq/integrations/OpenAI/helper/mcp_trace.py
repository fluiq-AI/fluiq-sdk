from fluiq.integrations.OpenAI.helper.utils import _to_jsonable

MCP_TOOL_TYPE = "mcp"
MCP_OUTPUT_TYPES = {"mcp_list_tools", "mcp_call", "mcp_approval_request"}


def _extract_mcp_servers_from_tools(tools):
    if not tools:
        return None
    out = []
    for tool in tools:
        ttype = (
            tool.get("type") if isinstance(tool, dict)
            else getattr(tool, "type", None)
        )
        if ttype == MCP_TOOL_TYPE:
            out.append(_to_jsonable(tool))
    return out or None


def _extract_mcp_calls_from_output(output_items):
    if not output_items:
        return None
    out = []
    for item in output_items:
        itype = (
            item.get("type") if isinstance(item, dict)
            else getattr(item, "type", None)
        )
        if itype in MCP_OUTPUT_TYPES:
            out.append(_to_jsonable(item))
    return out or None