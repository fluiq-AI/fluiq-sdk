"""
Contract: the OpenAI integration in fluiq.integrations.OpenAI.trace must fail
open when FluiqAI's ingest endpoint is over quota or unreachable. The user's
OpenAI call must return its real response, and any error from OpenAI itself
must propagate untouched (never replaced or masked by an SDK-internal error).

Run:  python -m unittest tests.basic.test_openai_trace_fail_open
"""
import sys
import types
import unittest
from unittest.mock import patch

import requests

from fluiq import instrument
from ..keys import FLUIQ_API_KEY

instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")

def _install_fake_openai():
    """Plant the minimum module tree that patch_openai() imports from."""
    openai = types.ModuleType("openai")
    resources = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")

    class _Msg:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)
            self.finish_reason = "stop"
            self.index = 0

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 2
        total_tokens = 3

    class FakeResponse:
        model = "gpt-4o"
        usage = _Usage()
        def __init__(self):
            self.choices = [_Choice("ok")]

    class Completions:
        def create(self, *a, **kw):
            return FakeResponse()

    completions.Completions = Completions
    sys.modules.update({
        "openai": openai,
        "openai.resources": resources,
        "openai.resources.chat": chat,
        "openai.resources.chat.completions": completions,
    })
    return Completions, FakeResponse


class OpenAITraceFailOpen(unittest.TestCase):
    _MOD_KEYS = (
        "openai",
        "openai.resources",
        "openai.resources.chat",
        "openai.resources.chat.completions",
    )

    def setUp(self):
        self._orig = {k: sys.modules.get(k) for k in self._MOD_KEYS}

    def tearDown(self):
        for k in self._MOD_KEYS:
            v = self._orig[k]
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def _setup(self, create_impl=None):
        Completions, FakeResponse = _install_fake_openai()
        if create_impl is not None:
            Completions.create = create_impl
        # Re-import patch_openai fresh so it captures the just-installed
        # Completions.create as `original`.
        if "fluiq.integrations.OpenAI.trace" in sys.modules:
            del sys.modules["fluiq.integrations.OpenAI.trace"]
        from fluiq.integrations.OpenAI.trace import patch_openai
        patch_openai()
        return Completions, FakeResponse

    def _call(self, Completions):
        return Completions().create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

    def test_ingest_unreachable_does_not_surface(self):
        Completions, FakeResponse = self._setup()
        with patch(
            "fluiq.client.requests.post",
            side_effect=requests.exceptions.ConnectionError("ingest unreachable"),
        ):
            resp = self._call(Completions)
        self.assertIsInstance(resp, FakeResponse)

    def test_ingest_quota_exceeded_does_not_surface(self):
        Completions, FakeResponse = self._setup()

        def _raise_429():
            raise requests.exceptions.HTTPError("429 quota exceeded")

        fake_http_resp = types.SimpleNamespace(
            status_code=429,
            raise_for_status=_raise_429,
        )
        with patch("fluiq.client.requests.post", return_value=fake_http_resp):
            resp = self._call(Completions)
        self.assertIsInstance(resp, FakeResponse)

    def test_openai_error_propagates_unmasked_even_when_ingest_down(self):
        # Errors from OpenAI itself MUST reach the user's app, even if the
        # SDK's own ingest is simultaneously failing — the SDK must never
        # replace or swallow the user's API error.
        class OpenAIRateLimit(Exception):
            pass

        def boom(self, *a, **kw):
            raise OpenAIRateLimit("openai over quota")

        Completions, _ = self._setup(create_impl=boom)
        with patch(
            "fluiq.client.requests.post",
            side_effect=requests.exceptions.ConnectionError("ingest unreachable"),
        ):
            with self.assertRaises(OpenAIRateLimit):
                self._call(Completions)

    def test_emit_body_fault_does_not_surface(self):
        # Realistic case: a helper used inside _emit_chat_trace raises (e.g.
        # tool-call extraction choking on an unusual response shape, a
        # Pydantic validation error, etc.). The @_fail_open decorator on the
        # emitter must absorb it so the user's call still returns the real
        # response.
        Completions, FakeResponse = self._setup()
        import fluiq.integrations.OpenAI.trace as t
        with patch.object(
            t, "_extract_tool_calls",
            side_effect=RuntimeError("synthetic emitter-internal bug"),
        ):
            resp = self._call(Completions)
        self.assertIsInstance(resp, FakeResponse)

    def test_log_trace_fault_does_not_surface(self):
        # Belt-and-suspenders: log_trace itself is also wrapped, so a fault
        # in compute_chain_id / send_event paths is absorbed.
        Completions, FakeResponse = self._setup()
        with patch(
            "fluiq.tracer.send_event",
            side_effect=RuntimeError("synthetic send_event bug"),
        ):
            resp = self._call(Completions)
        self.assertIsInstance(resp, FakeResponse)


class TraceDecoratorFailOpen(unittest.TestCase):
    def test_user_function_result_unaffected_by_emit_failure(self):
        from fluiq import trace

        @trace
        def add(a, b):
            return a + b

        with patch(
            "fluiq.client.requests.post",
            side_effect=requests.exceptions.ConnectionError("ingest down"),
        ):
            self.assertEqual(add(2, 3), 5)

    def test_user_function_exception_propagates_unmasked(self):
        from fluiq import trace

        class MyDomainError(Exception):
            pass

        @trace
        def boom():
            raise MyDomainError("user-land")

        with patch(
            "fluiq.client.requests.post",
            side_effect=requests.exceptions.ConnectionError("ingest down"),
        ):
            with self.assertRaises(MyDomainError):
                boom()


if __name__ == "__main__":
    unittest.main()
