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
    "security_scan": True,
}

def init(
        api_key: str=API_KEY, version=VERSION, 
        endpoint: str=ENDPOINT, security_scan: bool = True
    ):
    _config["api_key"] = api_key
    _config["version"] = version
    _config["endpoint"] = endpoint
    _config["security_scan"] = security_scan

    from fluiq.integrations import init as _integration_init_
    _integration_init_()