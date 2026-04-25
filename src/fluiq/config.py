
_config={
    "api_key": None,
    "enabled": True,
    "version": "v1"
}

def init(api_key: str, version="v1"):
    _config["api_key"] = api_key
    _config["version"] = version

    from fluiq.integrations import init as _integration_init_
    _integration_init_()