from fluiq.integrations.OpenAI.helper.utils import _to_jsonable


class _ChatStreamAccumulator:
    def __init__(self):
        self.content_parts = []
        self.tool_calls = {}
        self.finish_reasons = []
        self.usage = None
        self.model = None
        self.thinking_parts = []

    def feed(self, chunk):
        model = getattr(chunk, "model", None)
        if model:
            self.model = model
        usage = getattr(chunk, "usage", None)
        if usage:
            self.usage = usage
        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                self.content_parts.append(content)
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if reasoning:
                self.thinking_parts.append(reasoning)
            tcs = getattr(delta, "tool_calls", None)
            for tc in tcs or []:
                idx = getattr(tc, "index", 0)
                bucket = self.tool_calls.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    bucket["id"] = tc_id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", None)
                    if name:
                        bucket["function"]["name"] = name
                    args = getattr(fn, "arguments", None)
                    if args:
                        bucket["function"]["arguments"] += args
            fr = getattr(choice, "finish_reason", None)
            if fr:
                self.finish_reasons.append(fr)

    def assemble(self):
        text = "".join(self.content_parts) or None
        tool_calls = [self.tool_calls[k] for k in sorted(self.tool_calls.keys())] or None
        thinking = self.thinking_parts or None
        usage = None
        if self.usage:
            usage = {
                "prompt": getattr(self.usage, "prompt_tokens", None),
                "completion": getattr(self.usage, "completion_tokens", None),
                "total": getattr(self.usage, "total_tokens", None),
            }
        return {
            "response": text,
            "tool_calls": tool_calls,
            "thinking": thinking,
            "finish_reasons": self.finish_reasons or None,
            "usage": usage,
            "model": self.model,
        }


class _ResponsesStreamAccumulator:
    def __init__(self):
        self.events = []
        self.final_response = None

    def feed(self, event):
        etype = getattr(event, "type", None)
        if etype == "response.completed":
            self.final_response = getattr(event, "response", None)
        self.events.append(_to_jsonable(event))

    def assemble(self):
        output = None
        usage = None
        model = None
        if self.final_response is not None:
            output = getattr(self.final_response, "output", None)
            usage = getattr(self.final_response, "usage", None)
            model = getattr(self.final_response, "model", None)
        return {
            "response": _to_jsonable(output) if output is not None else self.events,
            "usage": _to_jsonable(usage),
            "model": model,
            "raw_response": self.final_response,
        }


class _StreamProxy:
    """Wraps a sync stream iterator; calls on_chunk(chunk) for each chunk and on_end() at completion."""
    def __init__(self, stream, on_chunk, on_end):
        self._stream = stream
        self._on_chunk = on_chunk
        self._on_end = on_end
        self._ended = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finalize()
            raise
        self._on_chunk(chunk)
        return chunk

    def _finalize(self):
        if self._ended:
            return
        self._ended = True
        try:
            self._on_end()
        except Exception:
            pass

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if hasattr(self._stream, "__exit__"):
                self._stream.__exit__(exc_type, exc, tb)
        finally:
            self._finalize()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _AsyncStreamProxy:
    def __init__(self, stream, on_chunk, on_end):
        self._stream = stream
        self._on_chunk = on_chunk
        self._on_end = on_end
        self._ended = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        self._on_chunk(chunk)
        return chunk

    def _finalize(self):
        if self._ended:
            return
        self._ended = True
        try:
            self._on_end()
        except Exception:
            pass

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if hasattr(self._stream, "__aexit__"):
                await self._stream.__aexit__(exc_type, exc, tb)
        finally:
            self._finalize()

    def __getattr__(self, name):
        return getattr(self._stream, name)
