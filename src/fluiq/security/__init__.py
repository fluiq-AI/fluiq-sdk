"""fluiq.security — server-side security scanning via fluiq.secure().

Local scanners have been removed. All detection runs on the Fluiq backend
so patterns are never shipped in the public SDK.  Call fluiq.secure() after
fluiq.instrument() to activate scanning (requires Team plan or above).
"""
from fluiq.security.client import call_secure, pre_call_check

__all__ = ["call_secure", "pre_call_check"]
