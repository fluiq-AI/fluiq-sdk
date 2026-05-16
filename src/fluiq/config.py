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
    "eval_judge_model": "gpt-4o-mini",
}

def init(api_key: str = API_KEY, version: str = VERSION, endpoint: str = ENDPOINT):
    _config["api_key"]  = api_key
    _config["version"]  = version
    _config["endpoint"] = endpoint

    from fluiq.integrations import init as _integration_init_
    _integration_init_()