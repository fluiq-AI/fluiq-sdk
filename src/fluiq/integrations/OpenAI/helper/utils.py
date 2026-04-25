MEDIA_PART_TYPES = {
    "image_url", "image", "input_image", "output_image",
    "input_audio", "audio", "output_audio", "video",
}

def _to_jsonable(part):
    if hasattr(part, "model_dump"):
        return part.model_dump()
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