from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("FLUIQ_API_KEY")
ENDPOINT = os.getenv("FLUIQ_API_ENDPOINT","https://api.getfluiq.com/api")
VERSION = "v1"

_config={
    "api_key": None,
    "enabled": True,
    "version": "v1",
    "endpoint": ENDPOINT,
    "secure": False,
    "secure_mode": "warn",      # "warn" | "block"
    "optimize": False,
    "optimize_mode": "cache",    # "cache" | "observe"
    "eval": False,
    "eval_mode": "warn",         # "warn" | "block"
    "eval_metrics": None,        # None → SDK default; or explicit list
    "eval_thresholds": {},       # {"hallucination": 0.8, ...}
    "eval_judge_model": "claude-haiku-4-5-20251001",
    "eval_custom_judges": {},    # {judge_prompt_slug: threshold}
}

def auth_headers() -> dict:
    """Return the Authorization header carrying the configured API key.

    The API key is transmitted as an HTTP ``Authorization: Bearer`` token on
    every SDK → fluiq-api request. Returns an empty dict when no key is
    configured so callers can spread it unconditionally.
    """
    key = _config.get("api_key")
    return {"Authorization": f"Bearer {key}"} if key else {}


def init(api_key: str = API_KEY, version: str = VERSION, endpoint: str = ENDPOINT):
    _config["api_key"]  = api_key
    _config["version"]  = version
    _config["endpoint"] = endpoint

    from fluiq.integrations import init as _integration_init_
    _integration_init_()