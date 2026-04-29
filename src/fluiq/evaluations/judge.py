import os
from typing import Any, Callable, Dict, Optional

import requests

from fluiq.evaluations.base import _parse_json_object


JudgeFn = Callable[[str], str]

PROVIDERS = ("openai", "anthropic", "gemini", "fluiq")

DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-2.5-flash",
    "fluiq": "fluiq-judge",
}

_SYSTEM_PROMPT = (
    "You are a strict evaluator. Always respond with a single valid JSON "
    "object and nothing else."
)


class LLMJudge:
    """LLM-as-judge with pluggable providers.

    `provider` selects the backend: ``openai`` (default), ``anthropic``,
    ``gemini``, or ``fluiq``. The first three call the respective vendor
    SDK directly using the user's own API key. ``fluiq`` proxies the call
    through the Fluiq API so judging is billed and metered server-side.

    Pass ``judge_fn(prompt: str) -> str`` to short-circuit the provider
    path entirely (useful for offline testing and self-hosted models).
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        judge_fn: Optional[JudgeFn] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unsupported judge provider: {provider!r}. "
                f"Use one of: {PROVIDERS}"
            )
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.temperature = temperature
        self._judge_fn = judge_fn
        self._api_key = api_key
        self._client = None

    def __call__(self, prompt: str) -> str:
        if self._judge_fn is not None:
            return self._judge_fn(prompt)
        if self.provider == "openai":
            return self._call_openai(prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        if self.provider == "fluiq":
            return self._call_fluiq(prompt)
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def judge_json(self, prompt: str) -> Dict[str, Any]:
        return _parse_json_object(self(prompt))

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "judge provider 'openai' requires the `openai` package"
            ) from exc
        if self._client is None:
            key = self._api_key or os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key) if key else OpenAI()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or "{}"

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "judge provider 'anthropic' requires the `anthropic` package"
            ) from exc
        if self._client is None:
            key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                return text
        return "{}"

    def _call_gemini(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "judge provider 'gemini' requires the `google-genai` package"
            ) from exc
        if self._client is None:
            key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key) if key else genai.Client()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )
        return getattr(resp, "text", None) or "{}"

    def _call_fluiq(self, prompt: str) -> str:
        from fluiq.config import _config

        api_key = self._api_key or _config.get("api_key")
        if not api_key:
            raise RuntimeError(
                "judge provider 'fluiq' requires fluiq.instrument(api_key=...) "
                "or LLMJudge(api_key=...)"
            )
        endpoint = _config.get("endpoint")
        version = _config.get("version", "v1")
        url = f"{endpoint}/{version}/judge"
        resp = requests.post(
            url,
            json={
                "api_key": api_key,
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json() or {}
        return body.get("content") or "{}"



