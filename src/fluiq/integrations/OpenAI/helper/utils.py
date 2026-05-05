from fluiq.integrations.shared.safety import _fail_open

MEDIA_PART_TYPES = {
    "image_url", "image", "input_image", "output_image",
    "input_audio", "audio", "output_audio", "video",
}

@_fail_open
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

@_fail_open
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