"""Compact media references for traces.

The SDK never stores raw image/audio/video/document payloads in a trace — they
are large, often sensitive, and would blow past the Kafka message ceiling. But
*dropping* them entirely hid the fact that a call was multimodal at all, which
left the evaluator and security scanners blind to media.

:func:`media_reference` replaces a media content part with a small, bounded
**reference** — kind, mime, source (url / base64 / file), decoded byte size, and
a stable ``sha256`` content id — carrying **no payload**. Downstream can now see
media is present, dedupe by hash, and (in future) fetch by url, without the trace
ever holding the bytes.

Pure stdlib (hashlib / base64); safe to call in the hot path.
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Dict, Optional

# Small base64 media is kept **inline** in the reference so the evaluator and
# security scanners can actually use it (vision judging, OCR injection); larger
# payloads keep only the hash. Threshold in decoded bytes (env-overridable, 0
# disables inline retention entirely). Default 256 KiB.
try:
    _INLINE_MAX_BYTES = int(os.getenv("FLUIQ_MEDIA_INLINE_MAX_BYTES", str(256 * 1024)))
except (TypeError, ValueError):
    _INLINE_MAX_BYTES = 256 * 1024

# OpenAI/Anthropic content-part ``type`` → media kind.
_TYPE_KIND = {
    "image_url": "image", "image": "image", "input_image": "image", "output_image": "image",
    "input_audio": "audio", "audio": "audio", "output_audio": "audio",
    "video": "video", "input_video": "video",
    "document": "document", "file": "document",
}

_MIME_TOP_KIND = {
    "image": "image", "audio": "audio", "video": "video",
    "application": "document", "text": "document",
}


def _coerce_dict(part: Any) -> Dict[str, Any]:
    if isinstance(part, dict):
        return part
    if hasattr(part, "model_dump"):
        try:
            return part.model_dump(mode="json")
        except Exception:
            try:
                return part.model_dump()
            except Exception:
                return {}
    if hasattr(part, "to_dict"):
        try:
            return part.to_dict()
        except Exception:
            return {}
    return {}


def _kind_from_mime(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None
    return _MIME_TOP_KIND.get(str(mime).split("/", 1)[0].lower())


def _hash(value: Any) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()[:16]


def _b64_bytes(data: Any) -> Optional[int]:
    if isinstance(data, bytes):
        return len(data)
    if isinstance(data, str) and data:
        try:
            return len(base64.b64decode(data + "===", validate=False))
        except Exception:
            return (len(data) * 3) // 4  # cheap estimate
    return None


def is_media_type(ptype: Any) -> bool:
    return ptype in _TYPE_KIND


def media_reference(part: Any, kind: Optional[str] = None) -> Dict[str, Any]:
    """Build a payload-free reference dict for a media content part.

    Returns ``{"type": <original>, "_media_ref": {kind, mime?, source, bytes?,
    sha256?, url?}}``. Recognizes OpenAI (``image_url`` / ``input_audio``),
    Anthropic (``source``), and Gemini (``inline_data`` / ``file_data``) shapes.
    """
    p = _coerce_dict(part)
    ptype = p.get("type")
    mime: Optional[str] = None
    data: Any = None       # base64 string or bytes
    url: Optional[str] = None

    # Anthropic: {"source": {"type": "base64"|"url", "media_type", "data"|"url"}}
    src = p.get("source")
    if isinstance(src, dict):
        mime = src.get("media_type") or src.get("mime_type")
        url = url or src.get("url")
        data = data or src.get("data")

    # OpenAI: {"image_url": {"url": ...}} or {"image_url": "..."}
    iu = p.get("image_url")
    if isinstance(iu, dict):
        url = url or iu.get("url")
    elif isinstance(iu, str):
        url = url or iu

    # OpenAI: {"input_audio"|"audio"|"output_audio": {"data": ..., "format": ...}}
    for k in ("input_audio", "audio", "output_audio"):
        av = p.get(k)
        if isinstance(av, dict):
            data = data or av.get("data")
            fmt = av.get("format")
            if fmt and not mime:
                mime = f"audio/{fmt}"

    # Gemini: {"inline_data": {"mime_type", "data"}} / {"file_data": {"mime_type", "file_uri"}}
    idata = p.get("inline_data") or p.get("inlineData")
    if isinstance(idata, dict):
        mime = mime or idata.get("mime_type") or idata.get("mimeType")
        data = data or idata.get("data")
    fdata = p.get("file_data") or p.get("fileData")
    if isinstance(fdata, dict):
        mime = mime or fdata.get("mime_type") or fdata.get("mimeType")
        url = url or fdata.get("file_uri") or fdata.get("fileUri")

    # data: URLs carry base64 inline — unwrap to hash the content, not the blob.
    if isinstance(url, str) and url.startswith("data:"):
        try:
            header, b64 = url.split(",", 1)
            if not mime:
                mime = header[5:].split(";", 1)[0] or None
            data = data or b64
            url = None
        except ValueError:
            pass

    ref: Dict[str, Any] = {
        "kind": kind or _kind_from_mime(mime) or _TYPE_KIND.get(ptype) or "media",
    }
    if mime:
        ref["mime"] = mime

    if isinstance(data, (str, bytes)) and data:
        ref["source"] = "base64"
        nbytes = _b64_bytes(data)
        if nbytes is not None:
            ref["bytes"] = nbytes
        ref["sha256"] = _hash(data)
        # Keep the payload inline for small media so downstream eval / security
        # can use it; large media keeps only the hash (privacy + Kafka ceiling).
        if _INLINE_MAX_BYTES > 0 and (nbytes is None or nbytes <= _INLINE_MAX_BYTES):
            ref["data"] = data if isinstance(data, str) else base64.b64encode(data).decode("ascii")
    elif isinstance(url, str) and url:
        ref["source"] = "url"
        ref["url"] = url
        ref["sha256"] = _hash(url)
    else:
        ref["source"] = "unknown"

    return {"type": ptype or ref["kind"], "_media_ref": ref}
