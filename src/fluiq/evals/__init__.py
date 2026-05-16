"""fluiq.evals — server-side evaluation via fluiq.eval().

All evaluation runs on the Fluiq backend (LLM-as-judge) so no vendor
API keys are required in the SDK.  Call fluiq.eval() after
fluiq.instrument() to activate.
"""
from fluiq.evals.client import call_evaluate

__all__ = ["call_evaluate"]