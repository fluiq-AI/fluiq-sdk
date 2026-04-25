def _extract_thinking(candidates):
    if not candidates:
        return None
    thoughts = []
    for cand in candidates:
        if isinstance(cand, dict):
            content = cand.get("content")
            parts = (content or {}).get("parts") if content else None
        else:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
        for part in parts or []:
            if isinstance(part, dict):
                is_thought = part.get("thought")
                text = part.get("text")
                signature = part.get("thought_signature")
            else:
                is_thought = getattr(part, "thought", None)
                text = getattr(part, "text", None)
                signature = getattr(part, "thought_signature", None)
            if is_thought:
                thoughts.append({
                    "text": text,
                    "signature": signature,
                })
    return thoughts or None
