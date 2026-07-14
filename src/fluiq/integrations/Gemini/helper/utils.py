
def _to_jsonable(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    return obj

from fluiq.integrations.shared.media import media_reference


def _is_media_part(part):
    if isinstance(part,dict):
        return bool(part.get("inline_data") or part.get("file_data"))
    return bool(
        getattr(part, "inline_data", None) or getattr(part,"file_data",None)
    )


def _strip_media(candidates):
    """Replace media parts (inline_data / file_data) with a payload-free
    reference; keep text parts as-is."""
    if not candidates:
        return None
    kept_candidates = []
    for cand in candidates:
        if isinstance(cand, dict):
            content = cand.get("content")
            parts = (content or {}).get("parts") if content else None
        else:
            content = getattr(cand,"content",None)
            parts = getattr(content, "parts", None) if content else None

        if not parts:
            continue

        kept_parts = [
            media_reference(p) if _is_media_part(p) else _to_jsonable(p)
            for p in parts
        ]

        if kept_parts:
            kept_candidates.append(kept_parts)

    return kept_candidates or None