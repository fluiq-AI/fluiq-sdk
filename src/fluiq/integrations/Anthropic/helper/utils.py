MEDIA_PART_TYPES = {"image", "document", "thinking", "redacted_thinking"}

def _to_jsonable(part):
    if part is None:
        return None
    if hasattr(part, "model_dump"):
        return part.model_dump()
    if isinstance(part, list):
        return [_to_jsonable(x) for x in part]
    if isinstance(part, dict):
        return {k: _to_jsonable(v) for k, v in part.items()}
    return part

def _strip_media(content):
    if content is None:
        return None
    if isinstance(content, list):
        kept = []
        for part in content:
            ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if ptype in MEDIA_PART_TYPES:
                continue
            kept.append(_to_jsonable(part))
        return kept or None
    return content