import requests
from fluiq.config import _config, auth_headers

def send_event(data):

    if not _config["enabled"]:
        return

    try:
        r = requests.post(
            f"{_config['endpoint']}/{_config['version']}/ingest",
            json={
                "event":data
            },
            headers=auth_headers(),
            timeout=5
        )
        r.raise_for_status()
    except Exception as e:
        print("[fluiq] send event failed: ",repr(e))


def send_event_gated(data) -> dict:
    """Send event to /ingest and return the parsed response body.

    Used when ``fluiq.secure(mode='block')`` is active so the caller can
    inspect ``response_blocked`` before returning the LLM output to the user.
    Falls back to ``{}`` on any network or parse error (fail-open).
    """
    if not _config["enabled"]:
        return {}
    try:
        r = requests.post(
            f"{_config['endpoint']}/{_config['version']}/ingest",
            json={
                "event": data,
            },
            headers=auth_headers(),
            timeout=5,
        )
        r.raise_for_status()
        return r.json() if r.content else {}
    except Exception as e:
        print("[fluiq] send event failed: ", repr(e))
        return {}