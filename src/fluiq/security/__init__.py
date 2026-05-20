"""fluiq.security — server-side security scanning via fluiq.secure().

All detection runs on the Fluiq backend (evaluator worker) so patterns are
never shipped in the public SDK.  Call fluiq.secure() after fluiq.instrument()
to activate scanning (requires Team plan or above).

Pre-call block-mode check is synchronous (/secure/check endpoint).
Post-call full scan is async via the Kafka → evaluator worker pipeline.
"""
from fluiq.security.client import pre_call_check

__all__ = ["pre_call_check"]
