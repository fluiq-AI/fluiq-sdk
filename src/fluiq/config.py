from dotenv import load_dotenv
import os

load_dotenv()

ENDPOINT = os.getenv("FLUIQ_API_ENDPOINT","https://api.getfluiq.com/api")

_config={
    "api_key": None,
    "enabled": True,
    "version": "v1",
    "endpoint": ENDPOINT
}

def init(api_key: str, version="v1", endpoint=ENDPOINT):
    _config["api_key"] = api_key
    _config["version"] = version
    _config["endpoint"] = endpoint

    from fluiq.integrations import init as _integration_init_
    _integration_init_()