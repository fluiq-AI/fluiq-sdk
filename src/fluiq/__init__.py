from fluiq.config import init as _init, ENDPOINT, API_KEY, VERSION
from fluiq.decorator import trace
from fluiq.exceptions import FluiqSecurityError, FluiqEvalError
from fluiq.prompts import Prompt

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


def optimize(mode: str = "cache") -> None:
    """Activate trace-driven Redis caching (requires paid plan).

    Must be called after ``fluiq.instrument()``.

    Fluiq's backend analyses your historical traces to determine which LLM
    calls are repeated most often and provisions a dedicated Redis instance
    for your account.  On the first call after ``optimize()`` the SDK
    fetches that cache profile, connects to your Redis instance, and begins
    serving repeated prompts from cache — saving both latency and LLM cost.

    Parameters
    ----------
    mode : "cache" | "observe"
        ``"cache"``   (default) — full caching enabled.  Repeated LLM calls
        that match the backend profile are intercepted before the API call
        and served from Redis.  Real responses are stored automatically.

        ``"observe"`` — no interception.  The SDK still records what *would*
        have been a cache hit so you can review savings before opting in.
    """
    if mode not in ("cache", "observe"):
        raise ValueError(f"fluiq.optimize() mode must be 'cache' or 'observe', got {mode!r}")
    from fluiq.config import _config
    _config["optimize"]      = True
    _config["optimize_mode"] = mode


def eval(
    thresholds: dict | None = None,
    metrics: list[str] | None = None,
    mode: str = "warn",
    judge_model: str = "claude-haiku-4-5-20251001",
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
    """
    if mode not in ("warn", "block"):
        raise ValueError(f"fluiq.eval() mode must be 'warn' or 'block', got {mode!r}")
    from fluiq.config import _config
    _config["eval"]             = True
    _config["eval_mode"]        = mode
    _config["eval_thresholds"]  = dict(thresholds) if thresholds else {}
    _config["eval_metrics"]     = list(metrics) if metrics else ["hallucination", "relevance"]
    _config["eval_judge_model"] = judge_model


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


def lookup_tool_result(tool_name: str, args):
    """Return a cached tool result, or ``None`` if not in cache.

    *args* can be a dict or a JSON string.  Keys are sorted before hashing so
    argument order does not matter.

    Typical usage inside a tool execution function::

        result = fluiq.lookup_tool_result("get_weather", {"location": "London"})
        if result is not None:
            return result
        return call_weather_api("London")
    """
    try:
        from fluiq.optimization.client import lookup_tool_cache
        return lookup_tool_cache(tool_name, args)
    except Exception:
        return None