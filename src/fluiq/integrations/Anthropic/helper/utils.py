from fluiq.integrations.shared.media import media_reference

# Media parts get a payload-free reference; reasoning blocks are dropped (they
# are captured separately by the thinking-trace helper and are not media).
MEDIA_PART_TYPES = {"image", "document"}
_DROP_PART_TYPES = {"thinking", "redacted_thinking"}

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
    """Replace media parts (image/document) with a payload-free reference and
    drop reasoning blocks; keep text parts as-is."""
    if content is None:
        return None
    if isinstance(content, list):
        kept = []
        for part in content:
            ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if ptype in _DROP_PART_TYPES:
                continue
            if ptype in MEDIA_PART_TYPES:
                kept.append(media_reference(part))
            else:
                kept.append(_to_jsonable(part))
        return kept or None
    return content