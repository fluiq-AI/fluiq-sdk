from fluiq.integrations.Anthropic.helper.utils import _to_jsonable


class _MessageStreamAccumulator:
    def __init__(self):
        self.blocks = {}
        self.stop_reason = None
        self.usage = None
        self.model = None

    def feed(self, event):
        etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)

        if etype == "message_start":
            message = getattr(event, "message", None) or (event.get("message") if isinstance(event, dict) else None)
            if message is not None:
                self.model = getattr(message, "model", None) or (message.get("model") if isinstance(message, dict) else None)
                u = getattr(message, "usage", None) or (message.get("usage") if isinstance(message, dict) else None)
                if u:
                    self.usage = _to_jsonable(u)
            return

        if etype == "content_block_start":
            idx = getattr(event, "index", None)
            if idx is None and isinstance(event, dict):
                idx = event.get("index")
            block = getattr(event, "content_block", None)
            if block is None and isinstance(event, dict):
                block = event.get("content_block")
            if idx is not None and block is not None:
                btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                bucket = {"type": btype}
                if btype == "text":
                    bucket["text"] = ""
                elif btype == "tool_use":
                    bucket["id"] = getattr(block, "id", None) or (block.get("id") if isinstance(block, dict) else None)
                    bucket["name"] = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
                    bucket["input_json"] = ""
                elif btype in ("thinking", "redacted_thinking"):
                    bucket["thinking"] = ""
                self.blocks[idx] = bucket
            return

        if etype == "content_block_delta":
            idx = getattr(event, "index", None)
            if idx is None and isinstance(event, dict):
                idx = event.get("index")
            delta = getattr(event, "delta", None)
            if delta is None and isinstance(event, dict):
                delta = event.get("delta")
            if idx is None or delta is None:
                return
            bucket = self.blocks.get(idx)
            if bucket is None:
                return
            dtype = getattr(delta, "type", None) or (delta.get("type") if isinstance(delta, dict) else None)
            if dtype == "text_delta":
                bucket["text"] = bucket.get("text", "") + (
                    getattr(delta, "text", None) or (delta.get("text") if isinstance(delta, dict) else "") or ""
                )
            elif dtype == "input_json_delta":
                bucket["input_json"] = bucket.get("input_json", "") + (
                    getattr(delta, "partial_json", None) or (delta.get("partial_json") if isinstance(delta, dict) else "") or ""
                )
            elif dtype in ("thinking_delta",):
                bucket["thinking"] = bucket.get("thinking", "") + (
                    getattr(delta, "thinking", None) or (delta.get("thinking") if isinstance(delta, dict) else "") or ""
                )
            return

        if etype == "message_delta":
            delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else None)
            if delta is not None:
                sr = getattr(delta, "stop_reason", None) or (delta.get("stop_reason") if isinstance(delta, dict) else None)
                if sr:
                    self.stop_reason = sr
            usage = getattr(event, "usage", None) or (event.get("usage") if isinstance(event, dict) else None)
            if usage:
                u = _to_jsonable(usage)
                if isinstance(self.usage, dict) and isinstance(u, dict):
                    self.usage.update(u)
                else:
                    self.usage = u

    def assemble(self):
        ordered = [self.blocks[i] for i in sorted(self.blocks.keys())]
        text_parts = []
        thinking = []
        tool_uses = []
        content = []
        for b in ordered:
            btype = b.get("type")
            if btype == "text":
                text_parts.append(b.get("text", ""))
                content.append({"type": "text", "text": b.get("text", "")})
            elif btype == "tool_use":
                import json
                input_json = b.get("input_json") or ""
                try:
                    parsed = json.loads(input_json) if input_json else {}
                except Exception:
                    parsed = {"_raw": input_json}
                entry = {"type": "tool_use", "id": b.get("id"), "name": b.get("name"), "input": parsed}
                tool_uses.append(entry)
                content.append(entry)
            elif btype in ("thinking", "redacted_thinking"):
                thinking.append({"type": btype, "thinking": b.get("thinking", "")})
        usage = self.usage or {}
        prompt = usage.get("input_tokens") if isinstance(usage, dict) else None
        completion = usage.get("output_tokens") if isinstance(usage, dict) else None
        total = (prompt or 0) + (completion or 0) if (prompt or completion) else None
        cache_read = usage.get("cache_read_input_tokens") if isinstance(usage, dict) else None
        cache_creation = usage.get("cache_creation_input_tokens") if isinstance(usage, dict) else None
        return {
            "response": content or None,
            "text": "".join(text_parts) or None,
            "thinking": thinking or None,
            "tool_uses": tool_uses or None,
            "stop_reason": self.stop_reason,
            "model": self.model,
            "tokens": {"prompt": prompt, "completion": completion, "total": total} if (prompt or completion) else None,
            "prompt_cache_read_tokens": cache_read,
            "prompt_cache_creation_tokens": cache_creation,
        }
