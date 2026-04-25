from fluiq.integrations.Anthropic.helper.utils import _to_jsonable

THINKING_BLOCK_TYPES = {
    "thinking",
    "redacted_thinking"
}

def _extract_thinking(content):
    
    if not isinstance(content, list):
        return None

    blocks = []
    for block in content:
        if isinstance(block, dict):
            btype = block.get("type")
        else:
            btype = getattr(block,"type",None)

        if btype not in THINKING_BLOCK_TYPES:
            continue

        dumped = _to_jsonable(block)

        if isinstance(dumped, dict):
            blocks.append({
                "type": dumped.get("type"),
                "thinking": dumped.get("thinking"),
                "signature": dumped.get("signature"),
                "data": dumped.get("data")
            })
        else:
            blocks.append(dumped)
    
    return blocks or None