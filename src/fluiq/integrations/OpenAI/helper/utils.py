from fluiq.integrations.shared.safety import _fail_open
from fluiq.integrations.shared.media import media_reference

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
    """Replace media parts with a payload-free reference; keep text parts as-is.

    The raw image/audio/video bytes are never stored, but a compact
    ``_media_ref`` (kind / mime / size / sha256 / url) is kept so the trace still
    records that the call was multimodal."""
    if content is None:
        return None
    if isinstance(content, list):
        kept = []
        for part in content:
            ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if ptype in MEDIA_PART_TYPES:
                kept.append(media_reference(part))
            else:
                kept.append(_to_jsonable(part))
        return kept or None
    return content