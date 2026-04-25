import requests
from fluiq.config import _config

ENDPOINT = "http://localhost:8000/api"

def send_event(data):

    if not _config["enabled"]:
        return
    
    try:
        r = requests.post(
            ENDPOINT+f"/{_config['version']}/ingest",
            json={
                "api_key": _config["api_key"],
                "event":data
            },
            timeout=1
        )
        r.raise_for_status()
    except Exception as e:
        print("[fluiq] send event failed: ",repr(e))