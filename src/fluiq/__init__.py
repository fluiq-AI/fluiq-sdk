from fluiq.config import init as _init, ENDPOINT, API_KEY, VERSION
from fluiq.decorator import trace
from fluiq.exceptions import FluiqSecurityError, FluiqEvalError
from fluiq.prompts import Prompt
from fluiq.integrations.shared.context import declare_parents as join_parents
# Code-first surface: declare evals in your repo, then `python -m fluiq.cli eval`.
# Exported here so the docs' first line isn't a deep import path.
#
# The declarative Prompt/Scorer/Dataset deliberately stay in `fluiq.resources`:
# `fluiq.Prompt` already means the *fetched* prompt object returned by
# fetch_prompt, and re-binding it here would silently change what that name
# refers to for every existing caller.
from fluiq.evals.definition import Eval

import re as _re

# Mirrors the API's tag rule (routes/trace.normalize_tags). Duplicated because
# the SDK cannot import from the API, and a tag the SDK accepts but the server
# drops would fail silently.
_TAG_RE = _re.compile(r"^[a-z0-9][a-z0-9._\-/]{0,62}$")

def instrument(
    api_key:  str = API_KEY,
    *,
    endpoint: str = ENDPOINT,
    version:  str = VERSION,
) -> None:
    """Start Fluiq instrumentation.

    Parameters
    ----------
    api_key:
        Your Fluiq API key.  Defaults to the ``FLUIQ_API_KEY`` env var.
    endpoint:
        Override the ingest endpoint (useful for local dev / self-hosting).
        Defaults to ``FLUIQ_API_ENDPOINT`` env var or ``https://api.getfluiq.com/api``.
    version:
        Trace schema version.  Pin in production so server-side schema bumps are opt-in.
    """
    _init(api_key=api_key, version=version, endpoint=endpoint)


def eval(
    thresholds: dict | None = None,
    metrics: list[str] | None = None,
    mode: str = "warn",
    judge_model: str = "claude-haiku-4-5-20251001",
    custom_judges: dict | None = None,
) -> None:
    """Activate server-side LLM response evaluation.

    Must be called after ``fluiq.instrument()``.

    After each LLM call Fluiq runs an LLM-as-judge on the response, scores
    each requested metric (0 = worst, 1 = best), stores the results in your
    dashboard, and — depending on ``mode`` — either warns or blocks when a
    score falls below its threshold.

    Parameters
    ----------
    thresholds : dict, optional
        Per-metric pass/fail thresholds, e.g.
        ``{"hallucination": 0.8, "relevance": 0.7}``.
        Supported: ``hallucination``, ``faithfulness``, ``relevance``,
        ``toxicity``, ``coherence``, ``completeness``.
    metrics : list[str], optional
        Which metrics to evaluate. Defaults to
        ``["hallucination", "relevance"]`` when omitted.
    mode : "warn" | "block"
        ``"warn"``  (default) — evaluate in the background and log a warning
        when any metric falls below its threshold.  LLM calls are never
        interrupted.

        ``"block"`` — evaluate synchronously after each LLM call and raise
        ``FluiqEvalError`` when any metric fails, preventing the response
        from reaching your application.
    judge_model : str
        The model Fluiq uses as judge. Defaults to ``"claude-haiku-4-5-20251001"``.
    custom_judges : dict, optional
        Your own LLM-as-judge prompts, mapping a saved judge prompt's slug to
        its pass/fail threshold, e.g. ``{"refund-policy": 0.8}``.

        Create a judge prompt in the dashboard (Prompts → Save as *Judge*).
        Its template uses ``$question``, ``$answer`` and ``$context``
        placeholders and should ask the model to return a JSON object with a
        numeric ``score`` (0–1) and a ``reason``. Each judge is scored on every
        LLM response just like a built-in metric, and (in ``block`` mode) raises
        ``FluiqEvalError`` when the score falls below its threshold.
    """
    if mode not in ("warn", "block"):
        raise ValueError(f"fluiq.eval() mode must be 'warn' or 'block', got {mode!r}")
    from fluiq.config import _config
    _config["eval"]              = True
    _config["eval_mode"]         = mode
    _config["eval_thresholds"]   = dict(thresholds) if thresholds else {}
    _config["eval_metrics"]      = list(metrics) if metrics else ["hallucination", "relevance"]
    _config["eval_judge_model"]  = judge_model
    _config["eval_custom_judges"] = {str(k): float(v) for k, v in (custom_judges or {}).items()}


def feedback(
    value,
    trace_id: str | None = None,
    name: str = "user_feedback",
    comment: str | None = None,
) -> None:
    """Record end-user feedback (a thumbs verdict or 0–1 rating) for a trace.

    Call it when your user reacts to an AI response — the score lands next to
    the automated evaluation results for that trace in the dashboard.

    Parameters
    ----------
    value : bool | float
        ``True``/``False`` for thumbs up/down, or a 0–1 rating.
    trace_id : str, optional
        The trace to attach the feedback to. Defaults to the most recent LLM
        call's trace in the current context (e.g. right after the call that
        produced the response the user is rating).
    name : str
        Feedback channel name shown in the dashboard, e.g. ``"thumbs"`` or
        ``"csat"``. Defaults to ``"user_feedback"``.
    comment : str, optional
        Free-text comment from the user.

    Fire-and-forget: never raises; a network failure only logs locally.
    """
    from fluiq.config import _config, auth_headers
    if not _config.get("enabled"):
        return
    if trace_id is None:
        from fluiq.integrations.shared.context import (
            current_llm_trace_id, current_parent_id,
        )
        trace_id = current_llm_trace_id() or current_parent_id()
    if not trace_id:
        print("[fluiq] feedback skipped: no trace_id (pass one explicitly)")
        return
    try:
        import requests
        requests.post(
            f"{_config['endpoint']}/{_config['version']}/feedback",
            json={
                "trace_id": str(trace_id),
                "name":     name,
                "value":    value,
                "comment":  comment,
            },
            headers=auth_headers(),
            timeout=5,
        ).raise_for_status()
    except Exception as e:
        print("[fluiq] feedback failed: ", repr(e))


def tag(*tags: str, replace: bool = False) -> None:
    """Label every trace from here on, so you can slice production traffic later.

    The case this exists for: shipping two prompts side by side and wanting to
    know which one scored better in the wild. Tag each cohort, then filter the
    dashboard by tag and compare.

        fluiq.tag("prompt-b", "canary")

    Parameters
    ----------
    *tags : str
        Lowercase letters, digits, and ``. _ - /``. Anything else is dropped
        rather than raising — a stray character in a label must never cost you
        the trace it was attached to.
    replace : bool
        By default tags accumulate, so a request handler can add one without
        knowing what a caller already set. Pass ``replace=True`` to reset.

    Tags can also be added and removed from the dashboard afterwards.
    """
    from fluiq.config import _config
    cleaned = [
        t for t in (str(x or "").strip().lower() for x in tags)
        if t and _TAG_RE.match(t)
    ]
    current = [] if replace else list(_config.get("tags") or [])
    _config["tags"] = sorted({*current, *cleaned})


def clear_tags() -> None:
    """Drop every tag set by :func:`tag`."""
    from fluiq.config import _config
    _config["tags"] = []


def set_metadata(**values) -> None:
    """Attach scalar key/values to every trace from here on.

        fluiq.set_metadata(tenant="acme", release="2026.8.1")

    Where a tag answers "which cohort is this?", metadata answers "what else was
    true at the time?". Only strings, numbers, and booleans are kept: nested
    structures are not filterable, so storing them would imply a capability that
    isn't there.
    """
    from fluiq.config import _config
    scalars = {
        str(k): v for k, v in values.items()
        if isinstance(v, (str, int, float, bool))
    }
    _config["metadata"] = {**(_config.get("metadata") or {}), **scalars}


def secure(mode: str = "warn", *, guardrail: str = "default") -> None:
    """Activate server-side security scanning (requires Team plan or above).

    Must be called after ``fluiq.instrument()``.

    Parameters
    ----------
    mode : "warn" | "block"
        ``"warn"``  (default) — post-call scan only.  Security fields are
        written into the stored trace; HIGH-risk content is redacted before
        persistence.  Your LLM calls are never interrupted.

        ``"block"`` — pre-call guard enabled.  Every prompt is checked
        against attack patterns *before* the LLM API call is made.  If the
        check returns ``allow=False``, a ``FluiqSecurityError`` is raised
        and the LLM call is never executed.  Post-call scanning still runs
        on allowed calls.

    guardrail : str
        Slug of the named guardrail policy to use (configured in the dashboard).
        Defaults to ``"default"``.  Unknown slugs fall back to ``"default"``
        on the server side.

    Raises ``FluiqSecurityError`` at LLM call time when mode is ``"block"``
    and an attack is detected.  Free-tier keys receive a 402 and fall back
    to warn behaviour automatically.
    """
    if mode not in ("warn", "block"):
        raise ValueError(f"fluiq.secure() mode must be 'warn' or 'block', got {mode!r}")
    from fluiq.config import _config
    _config["secure"]           = True
    _config["secure_mode"]      = mode
    _config["secure_guardrail"] = guardrail


def fetch_prompt(slug: str, env: str = "production") -> Prompt:
    """Fetch a deployed prompt template from the Fluiq dashboard.

    Must be called after ``fluiq.instrument()``.

    Parameters
    ----------
    slug : str
        The prompt's URL-safe identifier as set in the dashboard
        (e.g. ``"support-reply"``).
    env : "production" | "staging" | "development"
        Which environment snapshot to load. Defaults to ``"production"``.

    Returns
    -------
    Prompt
        A :class:`Prompt` object whose ``.render(**variables)`` method
        substitutes ``{variable}`` placeholders and returns the final string.

    Raises
    ------
    requests.HTTPError
        404 if the prompt is not deployed to the requested environment.
        401 if the API key is invalid.
    """
    import requests
    from fluiq.config import _config, auth_headers

    r = requests.get(
        f"{_config['endpoint']}/{_config['version']}/prompts/fetch/{slug}",
        params={"env": env},
        headers=auth_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return Prompt(r.json())
