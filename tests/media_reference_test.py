"""Media parts are preserved as payload-free references (not dropped).

Covers the shared helper + OpenAI / Anthropic / Gemini _strip_media.

Run:  ../.venv/Scripts/python.exe tests/media_reference_test.py
"""
import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fluiq.integrations.shared.media import media_reference
from fluiq.integrations.OpenAI.helper.utils import _strip_media as openai_strip
from fluiq.integrations.Anthropic.helper.utils import _strip_media as anthropic_strip
from fluiq.integrations.Gemini.helper.utils import _strip_media as gemini_strip

_PNG = base64.b64encode(b"\x89PNG\r\n" + b"x" * 500).decode()


def test_reference_base64_image():
    ref = media_reference({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}})
    assert ref["type"] == "image_url"
    m = ref["_media_ref"]
    assert m["kind"] == "image" and m["mime"] == "image/png"
    assert m["source"] == "base64" and m["bytes"] > 400 and len(m["sha256"]) == 16
    # small media (<256KB) keeps the base64 inline so eval/security can use it
    assert m.get("data") == _PNG


def test_reference_large_base64_hash_only():
    # >256KB decoded → payload dropped, hash kept
    big = base64.b64encode(b"x" * (300 * 1024)).decode()
    m = media_reference({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": big}})["_media_ref"]
    assert m["source"] == "base64" and m["sha256"] and m.get("data") is None


def test_reference_url_image():
    ref = media_reference({"type": "image_url", "image_url": {"url": "https://cdn.example.com/cat.png"}})
    m = ref["_media_ref"]
    assert m["source"] == "url" and m["url"] == "https://cdn.example.com/cat.png" and m["sha256"]


def test_reference_audio():
    ref = media_reference({"type": "input_audio", "input_audio": {"data": _PNG, "format": "wav"}})
    m = ref["_media_ref"]
    assert m["kind"] == "audio" and m["mime"] == "audio/wav" and m["source"] == "base64"


def test_openai_strip_keeps_text_and_refs():
    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}},
    ]
    out = openai_strip(content)
    assert len(out) == 2
    assert out[0]["text"] == "What is in this image?"      # text preserved
    assert out[1]["_media_ref"]["kind"] == "image"          # media → ref
    assert "image_url" not in out[1]                        # raw part replaced by ref
    assert out[1]["_media_ref"].get("data") == _PNG         # small payload kept inline


def test_anthropic_image_ref_thinking_dropped():
    content = [
        {"type": "text", "text": "Describe it"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _PNG}},
        {"type": "thinking", "thinking": "internal reasoning"},
    ]
    out = anthropic_strip(content)
    types = [p.get("type") for p in out]
    assert "text" in types and "image" in types
    assert "thinking" not in types                         # reasoning still dropped
    img = next(p for p in out if p.get("type") == "image")
    assert img["_media_ref"]["mime"] == "image/jpeg"
    assert "source" not in img                              # raw source replaced by ref


def test_gemini_inline_data_ref():
    candidates = [{"content": {"parts": [
        {"text": "hello"},
        {"inline_data": {"mime_type": "image/png", "data": _PNG}},
    ]}}]
    out = gemini_strip(candidates)
    parts = out[0]
    assert parts[0]["text"] == "hello"
    assert parts[1]["_media_ref"]["kind"] == "image"
    assert "inline_data" not in parts[1] and parts[1]["_media_ref"].get("data") == _PNG


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
