from fluiq.config import init as _init, ENDPOINT, API_KEY, VERSION
from fluiq.decorator import trace

def instrument(
    api_key:       str = API_KEY,
    *,
    endpoint:      str  = ENDPOINT,
    version:       str  = VERSION,
    security_scan: bool = True,
) -> None:
    """Start Fluiq instrumentation.
 
    Parameters
    ----------
    api_key:
        Your Fluiq API key.
    endpoint:
        Override the ingest endpoint (useful for local dev).
    security_scan:
        Set to ``False`` to disable PII / injection / secret scanning
        globally for this process.  Default is ``True``.
    """
    _init(
        api_key=api_key,
        version=version,
        endpoint=endpoint,
        security_scan=security_scan,
    )